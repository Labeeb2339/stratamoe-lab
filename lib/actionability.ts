import {
  DEFAULT_SIMULATION_CONTROLS,
  runPlannedSimulation,
  runSimulation,
  type ComparisonConfig,
  type RouterTrace,
  type SimulationConfig,
  type SimulationControls,
  type SimulationExecutionPlan,
  type SimulationResult,
} from "./simulator";

export const SHADOW_OBSERVATION_TOKENS = 12;
export const SHADOW_MINIMUM_SAVING_FRACTION = 0.05;
export const ACTION_HORIZON_TOKENS = 64;

export const NO_ACTION_CONTROLS: SimulationControls = Object.freeze({
  ...DEFAULT_SIMULATION_CONTROLS,
  shiftCachePrefetch: false,
  shiftCacheJsdReweighting: false,
  shiftCacheTransitionRetention: false,
  shiftCachePersistentDetector: true,
  shiftCacheTriggeredReweighting: false,
});

export const FROZEN_ACTION_CONTROLS: SimulationControls = Object.freeze({
  ...NO_ACTION_CONTROLS,
  shiftCacheTriggeredReweighting: true,
});

export type ShadowDecisionReason =
  | "passed"
  | "below-threshold"
  | "no-event"
  | "incomplete-window"
  | "zero-baseline-traffic";

export interface ShadowDecision {
  detectedEventToken: number | null;
  observationStartToken: number | null;
  observationEndTokenExclusive: number | null;
  decisionToken: number | null;
  noActionObservationBytes: number;
  frozenActionObservationBytes: number;
  estimatedSavingFraction: number | null;
  minimumSavingFraction: number;
  act: boolean;
  reason: ShadowDecisionReason;
  appliedActionStartToken: number | null;
  appliedActionEndTokenExclusive: number | null;
}

export interface ActionabilityTrafficWindows {
  primaryWindowBytes: number;
  postBoundaryBytes: number;
  wholeTraceBytes: number;
}

export interface ActionabilityArm extends ActionabilityTrafficWindows {
  result: SimulationResult;
}

export type FiniteOracleChoice =
  | "no-action"
  | "frozen-detected-action"
  | "perfect-boundary-action";

export interface FiniteActionOracle {
  choice: FiniteOracleChoice;
  primaryWindowBytes: number;
  headroomFraction: number;
}

export interface DetectorSummary {
  eventTokens: number[];
  firstEventToken: number | null;
  firstPostBoundaryEventToken: number | null;
  firstPostBoundaryDelayTokens: number | null;
  preBoundaryEvents: number;
  detectedWithin64Tokens: boolean;
}

export interface ActionabilityInvariants {
  traceFingerprintsMatch: boolean;
  exactRoutingSelectionsMatch: boolean;
  semanticRoutingChangesZero: boolean;
}

export interface ActionabilityCellResult {
  knownBoundaryToken: number;
  primaryWindow: {
    startToken: number;
    endTokenExclusive: number;
  };
  detector: DetectorSummary;
  shadowDecision: ShadowDecision;
  references: {
    lru: ActionabilityArm;
    lfu: ActionabilityArm;
  };
  arms: {
    noAction: ActionabilityArm;
    frozenDetectedAction: ActionabilityArm;
    perfectBoundaryAction: ActionabilityArm;
    trafficShadowGatedAction: ActionabilityArm;
  };
  finiteActionOracle: FiniteActionOracle;
  gatedPrimaryPercentChangeVsNoAction: number;
  frozenPrimaryPercentChangeVsNoAction: number;
  falseAction: boolean;
  harmfulAction: boolean;
  firstPermanentBreakEvenToken: number | null;
  regretBytesAgainstFiniteOracle: number;
  invariants: ActionabilityInvariants;
}

export interface RunActionabilityCellInput {
  config: ComparisonConfig | SimulationConfig;
  trace: RouterTrace;
  knownBoundaryToken: number;
}

function assertWindow(
  result: SimulationResult,
  startToken: number,
  endTokenExclusive: number,
): void {
  if (!Number.isSafeInteger(startToken) || !Number.isSafeInteger(endTokenExclusive)) {
    throw new RangeError("Traffic-window bounds must be safe integers.");
  }
  if (
    startToken < 0 ||
    endTokenExclusive < startToken ||
    endTokenExclusive > result.timeline.length
  ) {
    throw new RangeError(
      `Traffic window [${startToken}, ${endTokenExclusive}) is outside a ${result.timeline.length}-token result.`,
    );
  }
}

export function modeledBytes(
  result: SimulationResult,
  startToken: number,
  endTokenExclusive: number,
): number {
  assertWindow(result, startToken, endTokenExclusive);
  let total = 0;
  for (let token = startToken; token < endTokenExclusive; token += 1) {
    total += result.timeline[token].bytesTransferred;
  }
  return total;
}

function abstention(
  detectedEventToken: number | null,
  reason: Exclude<ShadowDecisionReason, "passed" | "below-threshold">,
  observationStartToken: number | null = null,
  observationEndTokenExclusive: number | null = null,
  decisionToken: number | null = null,
): ShadowDecision {
  return {
    detectedEventToken,
    observationStartToken,
    observationEndTokenExclusive,
    decisionToken,
    noActionObservationBytes: 0,
    frozenActionObservationBytes: 0,
    estimatedSavingFraction: null,
    minimumSavingFraction: SHADOW_MINIMUM_SAVING_FRACTION,
    act: false,
    reason,
    appliedActionStartToken: null,
    appliedActionEndTokenExclusive: null,
  };
}

export function decideTrafficShadow(
  noAction: SimulationResult,
  frozenAction: SimulationResult,
  detectedEventToken: number | null,
  horizonEndExclusive = Math.min(
    noAction.timeline.length,
    frozenAction.timeline.length,
  ),
): ShadowDecision {
  if (noAction.traceFingerprint !== frozenAction.traceFingerprint) {
    throw new RangeError("Shadow arms must replay the same router trace.");
  }
  if (noAction.timeline.length !== frozenAction.timeline.length) {
    throw new RangeError("Shadow arms must have identical timeline lengths.");
  }
  if (
    !Number.isSafeInteger(horizonEndExclusive) ||
    horizonEndExclusive < 0 ||
    horizonEndExclusive > noAction.timeline.length
  ) {
    throw new RangeError("horizonEndExclusive must be within both shadow timelines.");
  }
  if (detectedEventToken === null) {
    return abstention(null, "no-event");
  }
  if (
    !Number.isSafeInteger(detectedEventToken) ||
    detectedEventToken < 0 ||
    detectedEventToken >= noAction.timeline.length
  ) {
    throw new RangeError("detectedEventToken must identify a token in the trace.");
  }

  const observationStartToken = detectedEventToken + 1;
  const observationEndTokenExclusive =
    observationStartToken + SHADOW_OBSERVATION_TOKENS;
  const decisionToken = observationEndTokenExclusive - 1;
  const originalActionEndTokenExclusive =
    detectedEventToken + 1 + ACTION_HORIZON_TOKENS;
  if (
    observationEndTokenExclusive > horizonEndExclusive ||
    originalActionEndTokenExclusive > horizonEndExclusive
  ) {
    return abstention(
      detectedEventToken,
      "incomplete-window",
      observationStartToken,
      observationEndTokenExclusive,
      decisionToken,
    );
  }

  const noActionObservationBytes = modeledBytes(
    noAction,
    observationStartToken,
    observationEndTokenExclusive,
  );
  const frozenActionObservationBytes = modeledBytes(
    frozenAction,
    observationStartToken,
    observationEndTokenExclusive,
  );
  if (noActionObservationBytes === 0) {
    return {
      ...abstention(
        detectedEventToken,
        "zero-baseline-traffic",
        observationStartToken,
        observationEndTokenExclusive,
        decisionToken,
      ),
      noActionObservationBytes,
      frozenActionObservationBytes,
    };
  }

  const estimatedSavingFraction =
    (noActionObservationBytes - frozenActionObservationBytes) /
    noActionObservationBytes;
  const act =
    frozenActionObservationBytes < noActionObservationBytes &&
    estimatedSavingFraction >= SHADOW_MINIMUM_SAVING_FRACTION;

  return {
    detectedEventToken,
    observationStartToken,
    observationEndTokenExclusive,
    decisionToken,
    noActionObservationBytes,
    frozenActionObservationBytes,
    estimatedSavingFraction,
    minimumSavingFraction: SHADOW_MINIMUM_SAVING_FRACTION,
    act,
    reason: act ? "passed" : "below-threshold",
    appliedActionStartToken: act ? observationEndTokenExclusive : null,
    appliedActionEndTokenExclusive: act
      ? originalActionEndTokenExclusive
      : null,
  };
}

function comparisonConfig(
  input: ComparisonConfig | SimulationConfig,
): ComparisonConfig {
  return {
    scenario: input.scenario,
    seed: input.seed,
    tokens: input.tokens,
    layers: input.layers,
    expertsPerLayer: input.expertsPerLayer,
    topK: input.topK,
    gpuSlots: input.gpuSlots,
    ramSlots: input.ramSlots,
    expertSizeMB: input.expertSizeMB,
    pcieGBps: input.pcieGBps,
    nvmeGBps: input.nvmeGBps,
    computeMsPerToken: input.computeMsPerToken,
  };
}

function evaluateArm(
  result: SimulationResult,
  boundary: number,
): ActionabilityArm {
  const primaryEnd = boundary + ACTION_HORIZON_TOKENS;
  return {
    result,
    primaryWindowBytes: modeledBytes(result, boundary, primaryEnd),
    postBoundaryBytes: modeledBytes(result, boundary, result.timeline.length),
    wholeTraceBytes: modeledBytes(result, 0, result.timeline.length),
  };
}

function percentChange(after: number, before: number): number {
  if (before === 0) return after === 0 ? 0 : Number.POSITIVE_INFINITY;
  return ((after - before) / before) * 100;
}

function sameSelections(
  left: SimulationResult,
  right: SimulationResult,
): boolean {
  const leftSelections = left.trace.selections;
  const rightSelections = right.trace.selections;
  if (leftSelections.length !== rightSelections.length) return false;
  for (let token = 0; token < leftSelections.length; token += 1) {
    if (leftSelections[token].length !== rightSelections[token].length) return false;
    for (let layer = 0; layer < leftSelections[token].length; layer += 1) {
      const leftExperts = leftSelections[token][layer];
      const rightExperts = rightSelections[token][layer];
      if (leftExperts.length !== rightExperts.length) return false;
      for (let rank = 0; rank < leftExperts.length; rank += 1) {
        if (leftExperts[rank] !== rightExperts[rank]) return false;
      }
    }
  }
  return true;
}

export function firstPermanentBreakEvenToken(
  noAction: SimulationResult,
  gatedAction: SimulationResult,
  primaryStart: number,
  primaryEndExclusive: number,
  actionStartToken: number | null,
): number | null {
  if (actionStartToken === null || actionStartToken >= primaryEndExclusive) {
    return null;
  }
  const firstCandidate = Math.max(primaryStart, actionStartToken);
  const cumulativeDifferences: number[] = [];
  let gatedMinusNoAction = 0;
  for (let token = primaryStart; token < primaryEndExclusive; token += 1) {
    gatedMinusNoAction +=
      gatedAction.timeline[token].bytesTransferred -
      noAction.timeline[token].bytesTransferred;
    cumulativeDifferences.push(gatedMinusNoAction);
  }

  for (let token = firstCandidate; token < primaryEndExclusive; token += 1) {
    const index = token - primaryStart;
    if (cumulativeDifferences[index] > 0) continue;
    let remainsBrokenEven = true;
    for (let later = index + 1; later < cumulativeDifferences.length; later += 1) {
      if (cumulativeDifferences[later] > 0) {
        remainsBrokenEven = false;
        break;
      }
    }
    if (remainsBrokenEven) return token;
  }
  return null;
}

function plan(
  windows: SimulationExecutionPlan["forcedReweightingWindows"],
): SimulationExecutionPlan {
  return { forcedReweightingWindows: windows };
}

export function runActionabilityCell({
  config: inputConfig,
  trace,
  knownBoundaryToken,
}: RunActionabilityCellInput): ActionabilityCellResult {
  const baseConfig = comparisonConfig(inputConfig);
  if (
    !Number.isSafeInteger(knownBoundaryToken) ||
    knownBoundaryToken < 0 ||
    knownBoundaryToken + ACTION_HORIZON_TOKENS > baseConfig.tokens
  ) {
    throw new RangeError(
      "knownBoundaryToken must leave a complete 64-token primary window.",
    );
  }

  const shiftConfig = { ...baseConfig, policy: "shift-cache" } as const;
  const lru = runSimulation(
    { ...baseConfig, policy: "lru" },
    trace,
    NO_ACTION_CONTROLS,
  );
  const lfu = runSimulation(
    { ...baseConfig, policy: "lfu" },
    trace,
    NO_ACTION_CONTROLS,
  );
  const noAction = runSimulation(
    shiftConfig,
    trace,
    NO_ACTION_CONTROLS,
  );
  const frozenDetectedAction = runSimulation(
    shiftConfig,
    trace,
    FROZEN_ACTION_CONTROLS,
  );
  const perfectBoundaryAction = runPlannedSimulation(
    shiftConfig,
    trace,
    NO_ACTION_CONTROLS,
    plan([
      {
        startToken: knownBoundaryToken,
        endTokenExclusive: knownBoundaryToken + ACTION_HORIZON_TOKENS,
      },
    ]),
  );

  const detectorEventTokens = noAction.timeline
    .filter((point) => point.shiftDetected)
    .map((point) => point.token);
  const firstEventToken = detectorEventTokens[0] ?? null;
  const firstPostBoundaryEventToken =
    detectorEventTokens.find((token) => token >= knownBoundaryToken) ?? null;
  const shadowDecision = decideTrafficShadow(
    noAction,
    frozenDetectedAction,
    firstEventToken,
    baseConfig.tokens,
  );
  const gatedWindows: SimulationExecutionPlan["forcedReweightingWindows"] =
    shadowDecision.act &&
    shadowDecision.appliedActionStartToken !== null &&
    shadowDecision.appliedActionEndTokenExclusive !== null
      ? [
          {
            startToken: shadowDecision.appliedActionStartToken,
            endTokenExclusive: shadowDecision.appliedActionEndTokenExclusive,
          },
        ]
      : [];
  const trafficShadowGatedAction = runPlannedSimulation(
    shiftConfig,
    trace,
    NO_ACTION_CONTROLS,
    plan(gatedWindows),
  );

  const references = {
    lru: evaluateArm(lru, knownBoundaryToken),
    lfu: evaluateArm(lfu, knownBoundaryToken),
  };
  const arms = {
    noAction: evaluateArm(noAction, knownBoundaryToken),
    frozenDetectedAction: evaluateArm(
      frozenDetectedAction,
      knownBoundaryToken,
    ),
    perfectBoundaryAction: evaluateArm(
      perfectBoundaryAction,
      knownBoundaryToken,
    ),
    trafficShadowGatedAction: evaluateArm(
      trafficShadowGatedAction,
      knownBoundaryToken,
    ),
  };

  const oracleCandidates: Array<{
    choice: FiniteOracleChoice;
    bytes: number;
  }> = [
    { choice: "no-action", bytes: arms.noAction.primaryWindowBytes },
    {
      choice: "frozen-detected-action",
      bytes: arms.frozenDetectedAction.primaryWindowBytes,
    },
    {
      choice: "perfect-boundary-action",
      bytes: arms.perfectBoundaryAction.primaryWindowBytes,
    },
  ];
  let oracle = oracleCandidates[0];
  for (const candidate of oracleCandidates.slice(1)) {
    if (candidate.bytes < oracle.bytes) oracle = candidate;
  }
  const finiteActionOracle: FiniteActionOracle = {
    choice: oracle.choice,
    primaryWindowBytes: oracle.bytes,
    headroomFraction:
      arms.noAction.primaryWindowBytes === 0
        ? 0
        : (arms.noAction.primaryWindowBytes - oracle.bytes) /
          arms.noAction.primaryWindowBytes,
  };

  const allResults = [
    lru,
    lfu,
    noAction,
    frozenDetectedAction,
    perfectBoundaryAction,
    trafficShadowGatedAction,
  ];
  const invariants: ActionabilityInvariants = {
    traceFingerprintsMatch: allResults.every(
      (result) => result.traceFingerprint === noAction.traceFingerprint,
    ),
    exactRoutingSelectionsMatch: allResults.every((result) =>
      sameSelections(result, noAction),
    ),
    semanticRoutingChangesZero: allResults.every(
      (result) => result.metrics.semanticRoutingChanges === 0,
    ),
  };
  if (
    !invariants.traceFingerprintsMatch ||
    !invariants.exactRoutingSelectionsMatch ||
    !invariants.semanticRoutingChangesZero
  ) {
    throw new Error("Actionability arms violated trace or semantic invariants.");
  }

  const primaryEnd = knownBoundaryToken + ACTION_HORIZON_TOKENS;
  return {
    knownBoundaryToken,
    primaryWindow: {
      startToken: knownBoundaryToken,
      endTokenExclusive: primaryEnd,
    },
    detector: {
      eventTokens: detectorEventTokens,
      firstEventToken,
      firstPostBoundaryEventToken,
      firstPostBoundaryDelayTokens:
        firstPostBoundaryEventToken === null
          ? null
          : firstPostBoundaryEventToken - knownBoundaryToken,
      preBoundaryEvents: detectorEventTokens.filter(
        (token) => token < knownBoundaryToken,
      ).length,
      detectedWithin64Tokens:
        firstPostBoundaryEventToken !== null &&
        firstPostBoundaryEventToken - knownBoundaryToken <=
          ACTION_HORIZON_TOKENS,
    },
    shadowDecision,
    references,
    arms,
    finiteActionOracle,
    gatedPrimaryPercentChangeVsNoAction: percentChange(
      arms.trafficShadowGatedAction.primaryWindowBytes,
      arms.noAction.primaryWindowBytes,
    ),
    frozenPrimaryPercentChangeVsNoAction: percentChange(
      arms.frozenDetectedAction.primaryWindowBytes,
      arms.noAction.primaryWindowBytes,
    ),
    falseAction:
      shadowDecision.act && finiteActionOracle.choice === "no-action",
    harmfulAction:
      shadowDecision.act &&
      arms.trafficShadowGatedAction.primaryWindowBytes >
        arms.noAction.primaryWindowBytes,
    firstPermanentBreakEvenToken: firstPermanentBreakEvenToken(
      noAction,
      trafficShadowGatedAction,
      knownBoundaryToken,
      primaryEnd,
      shadowDecision.appliedActionStartToken,
    ),
    regretBytesAgainstFiniteOracle:
      arms.trafficShadowGatedAction.primaryWindowBytes -
      finiteActionOracle.primaryWindowBytes,
    invariants,
  };
}
