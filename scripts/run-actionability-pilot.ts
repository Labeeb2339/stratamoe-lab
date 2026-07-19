import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  existsSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { fileURLToPath } from "node:url";
import {
  ACTION_HORIZON_TOKENS,
  FROZEN_ACTION_CONTROLS,
  NO_ACTION_CONTROLS,
  SHADOW_MINIMUM_SAVING_FRACTION,
  SHADOW_OBSERVATION_TOKENS,
  decideTrafficShadow,
  modeledBytes,
  runActionabilityCell,
  type ActionabilityArm,
  type ActionabilityCellResult,
} from "../lib/actionability";
import {
  SHIFT_CACHE_PARAMETERS,
  generateRouterTrace,
  type ComparisonConfig,
  type ScenarioId,
  type SimulationExecutionPlan,
} from "../lib/simulator";

type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

interface ActionabilityProtocol {
  protocol: string;
  status: string;
  evidenceBoundary: string;
  excludedSeeds: Record<string, number[]>;
  confirmatorySeeds: number[];
  configuration: {
    tokens: number;
    knownBoundaryToken: number;
    layers: number;
    expertsPerLayer: number;
    topK: number;
    gpuSlots: number[];
    stationaryGpuSlots: number;
    ramSlots: number;
    expertSizeMB: number;
    pcieGBps: number;
    nvmeGBps: number;
    computeMsPerToken: number;
  };
  detector: {
    shortWindowTokens: number;
    longWindowTokens: number;
    shiftThresholdBits: number;
    rearmThresholdBits: number;
    persistentThresholdTokens: number;
    cooldownTokens: number;
    actionHorizonTokens: number;
  };
  shadowGate: {
    observationTokens: number;
    minimumPrefixSavingPercent: number;
    ties: string;
    incompleteObservationWindow: string;
  };
  primaryWindow: {
    startToken: number;
    endTokenExclusive: number;
  };
  bootstrap: {
    unit: string;
    resamples: number;
    prng: string;
    seed: number;
    lowerPercentile: number;
    upperPercentile: number;
  };
  gates: {
    completeCells: number;
    minimumDetectedWithin64: number;
    maximumPreBoundaryEvents: number;
    maximumStationaryEventsPer10000Tokens: number;
    minimumOracleHeadroomPercent: number;
    minimumOracleActionableCells: number;
    minimumOracleActionableCapacities: number;
    minimumOracleActionableMedianImprovementPercent: number;
    maximumAnyCellRegressionPercent: number;
    maximumHarmfulExecutedActionRate: number;
    requireBootstrapUpperBelowZero: boolean;
    requireGatedMedianNoWorseThanFrozen: boolean;
    requireByteIdenticalRerun: boolean;
    requireSemanticRoutingChangesZero: boolean;
  };
  externalPilotOutsideGates: Record<string, JsonValue>;
}

interface ArmRecord {
  primaryWindowBytes: number;
  postBoundaryBytes: number;
  wholeTraceBytes: number;
  resultSha256: string;
  semanticRoutingChanges: number;
}

interface CellRecord {
  scenario: ScenarioId;
  seed: number;
  gpuSlots: number;
  traceFingerprint: string;
  traceSha256: string;
  configurationSha256: string;
  actionabilityResultSha256: string;
  executionPlans: {
    perfectBoundaryAction: SimulationExecutionPlan;
    trafficShadowGatedAction: SimulationExecutionPlan;
  };
  detector: ActionabilityCellResult["detector"];
  shadowDecision: ActionabilityCellResult["shadowDecision"];
  references: {
    lru: ArmRecord;
    lfu: ArmRecord;
  };
  arms: {
    noAction: ArmRecord;
    frozenDetectedAction: ArmRecord;
    perfectBoundaryAction: ArmRecord;
    trafficShadowGatedAction: ArmRecord;
  };
  finiteActionOracle: ActionabilityCellResult["finiteActionOracle"];
  gatedPrimaryPercentChangeVsNoAction: number;
  frozenPrimaryPercentChangeVsNoAction: number;
  falseAction: boolean;
  harmfulAction: boolean;
  firstPermanentBreakEvenToken: number | null;
  regretBytesAgainstFiniteOracle: number;
  invariants: ActionabilityCellResult["invariants"];
}

interface BootstrapEndpoint {
  unit: "seed-cluster";
  statistic: "median-paired-percent-change";
  estimate: number | null;
  resamplesRequested: number;
  validResamples: number;
  emptyResamples: number;
  prng: "xorshift32";
  seed: number;
  lowerPercentile: number;
  upperPercentile: number;
  lower: number | null;
  upper: number | null;
}

interface GateRecord {
  requirement: string;
  observed: JsonValue;
  passed: boolean;
}

const protocolUrl = new URL(
  "../evidence/actionability-v1/protocol.json",
  import.meta.url,
);
const resultsUrl = new URL(
  "../evidence/actionability-v1/results.json",
  import.meta.url,
);
const repositoryRoot = fileURLToPath(new URL("../", import.meta.url));
const resultsRepositoryPath = "evidence/actionability-v1/results.json";
const frozenProtocolCanonicalSha256 =
  "abd2dda6ec914927df58ffc3b647bb6c5dfead8cf9e35ea284d4548563ebc186";
const internalEmitMode = "--emit-core";
const implementationPaths = [
  "lib/simulator.ts",
  "lib/actionability.ts",
  "scripts/run-actionability-pilot.ts",
  "lib/index.ts",
  "package.json",
  "tests/simulator.actionability.test.ts",
  "tests/simulator.execution-plan.test.ts",
] as const;

function canonicalize(value: unknown): JsonValue {
  if (value === null) return null;
  if (typeof value === "boolean" || typeof value === "string") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new RangeError("Canonical JSON cannot contain a non-finite number.");
    }
    return Object.is(value, -0) ? 0 : value;
  }
  if (Array.isArray(value)) return value.map((entry) => canonicalize(entry));
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const output: Record<string, JsonValue> = {};
    for (const key of Object.keys(record).sort()) {
      if (record[key] === undefined) {
        throw new TypeError(`Canonical JSON field ${key} is undefined.`);
      }
      output[key] = canonicalize(record[key]);
    }
    return output;
  }
  throw new TypeError(`Unsupported canonical JSON value: ${typeof value}.`);
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(canonicalize(value));
}

function canonicalFileJson(value: unknown): string {
  return `${JSON.stringify(canonicalize(value), null, 2)}\n`;
}

function sha256Canonical(value: unknown): string {
  return createHash("sha256").update(canonicalJson(value), "utf8").digest("hex");
}

function git(args: string[]): string {
  return execFileSync("git", args, {
    cwd: repositoryRoot,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  }).trim();
}

function latestCommitFor(paths: readonly string[]): string {
  const commit = git(["log", "-1", "--format=%H", "--", ...paths]);
  if (!/^[0-9a-f]{40}$/.test(commit)) {
    throw new Error(`Could not resolve a full commit for ${paths.join(", ")}.`);
  }
  return commit;
}

function assertWritePreconditions(): void {
  if (existsSync(resultsUrl)) {
    throw new Error(`${resultsRepositoryPath} already exists; refusing to overwrite it.`);
  }
  const status = git(["status", "--porcelain=v1", "--untracked-files=all"]);
  if (status.length > 0) {
    throw new Error("--write requires a clean Git working tree before confirmatory execution.");
  }
}

function assertVerifyPreconditions(): void {
  if (!existsSync(resultsUrl)) {
    throw new Error(`${resultsRepositoryPath} does not exist; there is nothing to verify.`);
  }
  const unexpectedChanges = git([
    "status",
    "--porcelain=v1",
    "--untracked-files=all",
  ])
    .split(/\r?\n/u)
    .filter((line) => line.length > 0)
    .filter((line) => !line.endsWith(resultsRepositoryPath));
  if (unexpectedChanges.length > 0) {
    throw new Error(
      "--verify permits only the generated results file to differ from Git.",
    );
  }
}

function loadProtocol(): ActionabilityProtocol {
  const parsed = JSON.parse(readFileSync(protocolUrl, "utf8")) as unknown;
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new TypeError("Actionability protocol must be a JSON object.");
  }
  return parsed as ActionabilityProtocol;
}

function equalNumbers(left: readonly number[], right: readonly number[]): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function assertProtocol(protocol: ActionabilityProtocol): void {
  const protocolSha256 = sha256Canonical(protocol);
  if (protocolSha256 !== frozenProtocolCanonicalSha256) {
    throw new Error(
      `Protocol canonical SHA-256 changed: expected ${frozenProtocolCanonicalSha256}, received ${protocolSha256}.`,
    );
  }
  const expectedSeeds = Array.from({ length: 30 }, (_, index) => 4100 + index);
  if (!equalNumbers(protocol.confirmatorySeeds, expectedSeeds)) {
    throw new Error("Protocol confirmatory seeds are not exactly 4100 through 4129.");
  }
  if (!equalNumbers(protocol.configuration.gpuSlots, [8, 16, 32, 64, 96])) {
    throw new Error("Protocol GPU capacities are not exactly 8, 16, 32, 64, 96.");
  }
  if (
    protocol.configuration.stationaryGpuSlots !== 32 ||
    protocol.configuration.tokens !== 512 ||
    protocol.configuration.knownBoundaryToken !== 256 ||
    protocol.primaryWindow.startToken !== 256 ||
    protocol.primaryWindow.endTokenExclusive !== 320
  ) {
    throw new Error("Protocol trace length, boundary, primary window, or stationary capacity drifted.");
  }
  if (
    protocol.detector.shortWindowTokens !==
      SHIFT_CACHE_PARAMETERS.maximumShortWindowTokens ||
    protocol.detector.longWindowTokens !==
      SHIFT_CACHE_PARAMETERS.maximumShortWindowTokens *
        SHIFT_CACHE_PARAMETERS.longWindowMultiplier ||
    protocol.detector.shiftThresholdBits !==
      SHIFT_CACHE_PARAMETERS.shiftThresholdBits ||
    protocol.detector.rearmThresholdBits !==
      SHIFT_CACHE_PARAMETERS.rearmThresholdBits ||
    protocol.detector.persistentThresholdTokens !==
      SHIFT_CACHE_PARAMETERS.persistentThresholdTokens ||
    protocol.detector.cooldownTokens !==
      SHIFT_CACHE_PARAMETERS.detectorCooldownTokens ||
    protocol.detector.actionHorizonTokens !== ACTION_HORIZON_TOKENS
  ) {
    throw new Error("Protocol detector settings do not match the frozen implementation.");
  }
  if (
    protocol.shadowGate.observationTokens !== SHADOW_OBSERVATION_TOKENS ||
    protocol.shadowGate.minimumPrefixSavingPercent !==
      SHADOW_MINIMUM_SAVING_FRACTION * 100 ||
    protocol.shadowGate.ties !== "abstain" ||
    protocol.shadowGate.incompleteObservationWindow !== "abstain"
  ) {
    throw new Error("Protocol shadow-gate settings do not match the frozen implementation.");
  }
  if (
    NO_ACTION_CONTROLS.shiftCachePrefetch ||
    NO_ACTION_CONTROLS.shiftCacheJsdReweighting ||
    NO_ACTION_CONTROLS.shiftCacheTransitionRetention ||
    !NO_ACTION_CONTROLS.shiftCachePersistentDetector ||
    NO_ACTION_CONTROLS.shiftCacheTriggeredReweighting ||
    !FROZEN_ACTION_CONTROLS.shiftCacheTriggeredReweighting
  ) {
    throw new Error("Frozen actionability controls no longer match the protocol.");
  }
  const excludedSeeds = new Set(Object.values(protocol.excludedSeeds).flat());
  if (protocol.confirmatorySeeds.some((seed) => excludedSeeds.has(seed))) {
    throw new Error("A confirmatory seed appears in a protocol exclusion set.");
  }
  if (
    protocol.bootstrap.unit !== "seed-cluster" ||
    protocol.bootstrap.resamples !== 10_000 ||
    protocol.bootstrap.prng !== "xorshift32" ||
    protocol.bootstrap.seed !== 2339 ||
    protocol.bootstrap.lowerPercentile !== 2.5 ||
    protocol.bootstrap.upperPercentile !== 97.5
  ) {
    throw new Error("Protocol bootstrap settings drifted from the frozen design.");
  }
}

function comparisonConfig(
  protocol: ActionabilityProtocol,
  scenario: ScenarioId,
  seed: number,
  gpuSlots: number,
): ComparisonConfig {
  const config = protocol.configuration;
  return {
    scenario,
    seed,
    tokens: config.tokens,
    layers: config.layers,
    expertsPerLayer: config.expertsPerLayer,
    topK: config.topK,
    gpuSlots,
    ramSlots: config.ramSlots,
    expertSizeMB: config.expertSizeMB,
    pcieGBps: config.pcieGBps,
    nvmeGBps: config.nvmeGBps,
    computeMsPerToken: config.computeMsPerToken,
  };
}

function assertEqualCanonical(
  actual: unknown,
  expected: unknown,
  label: string,
): void {
  if (canonicalJson(actual) !== canonicalJson(expected)) {
    throw new Error(`${label} did not match its independently recomputed value.`);
  }
}

function armRecord(
  arm: ActionabilityArm,
  boundary: number,
  primaryEndExclusive: number,
): ArmRecord {
  const independentlyMeasured = {
    primaryWindowBytes: modeledBytes(
      arm.result,
      boundary,
      primaryEndExclusive,
    ),
    postBoundaryBytes: modeledBytes(
      arm.result,
      boundary,
      arm.result.timeline.length,
    ),
    wholeTraceBytes: modeledBytes(
      arm.result,
      0,
      arm.result.timeline.length,
    ),
  };
  assertEqualCanonical(
    independentlyMeasured,
    {
      primaryWindowBytes: arm.primaryWindowBytes,
      postBoundaryBytes: arm.postBoundaryBytes,
      wholeTraceBytes: arm.wholeTraceBytes,
    },
    "Actionability traffic windows",
  );
  return {
    ...independentlyMeasured,
    resultSha256: sha256Canonical(arm.result),
    semanticRoutingChanges: arm.result.metrics.semanticRoutingChanges,
  };
}

function buildCellRecord(
  protocol: ActionabilityProtocol,
  scenario: ScenarioId,
  seed: number,
  gpuSlots: number,
): CellRecord {
  const config = comparisonConfig(protocol, scenario, seed, gpuSlots);
  const trace = generateRouterTrace({
    scenario,
    seed,
    tokens: config.tokens,
    layers: config.layers,
    expertsPerLayer: config.expertsPerLayer,
    topK: config.topK,
  });
  const result = runActionabilityCell({
    config,
    trace,
    knownBoundaryToken: protocol.configuration.knownBoundaryToken,
  });
  const independentlyDecidedShadow = decideTrafficShadow(
    result.arms.noAction.result,
    result.arms.frozenDetectedAction.result,
    result.detector.firstEventToken,
    config.tokens,
  );
  assertEqualCanonical(
    independentlyDecidedShadow,
    result.shadowDecision,
    "Traffic-shadow decision",
  );

  const primaryEndExclusive = protocol.primaryWindow.endTokenExclusive;
  const perfectBoundaryAction: SimulationExecutionPlan = {
    forcedReweightingWindows: [
      {
        startToken: protocol.configuration.knownBoundaryToken,
        endTokenExclusive:
          protocol.configuration.knownBoundaryToken + ACTION_HORIZON_TOKENS,
      },
    ],
  };
  const trafficShadowGatedAction: SimulationExecutionPlan = {
    forcedReweightingWindows:
      result.shadowDecision.act &&
      result.shadowDecision.appliedActionStartToken !== null &&
      result.shadowDecision.appliedActionEndTokenExclusive !== null
        ? [
            {
              startToken: result.shadowDecision.appliedActionStartToken,
              endTokenExclusive:
                result.shadowDecision.appliedActionEndTokenExclusive,
            },
          ]
        : [],
  };
  const configurationEvidence = {
    simulation: config,
    knownBoundaryToken: protocol.configuration.knownBoundaryToken,
    primaryWindow: protocol.primaryWindow,
    detector: protocol.detector,
    shadowGate: protocol.shadowGate,
    controls: {
      noAction: NO_ACTION_CONTROLS,
      frozenAction: FROZEN_ACTION_CONTROLS,
    },
    executionPlanRules: {
      perfectBoundaryAction,
      trafficShadowGatedAction:
        "Use the exact half-open range recorded for this cell after the causal shadow decision; otherwise use an empty plan.",
    },
  };

  return {
    scenario,
    seed,
    gpuSlots,
    traceFingerprint: result.arms.noAction.result.traceFingerprint,
    traceSha256: sha256Canonical(result.arms.noAction.result.trace),
    configurationSha256: sha256Canonical(configurationEvidence),
    actionabilityResultSha256: sha256Canonical(result),
    executionPlans: {
      perfectBoundaryAction,
      trafficShadowGatedAction,
    },
    detector: result.detector,
    shadowDecision: result.shadowDecision,
    references: {
      lru: armRecord(
        result.references.lru,
        protocol.primaryWindow.startToken,
        primaryEndExclusive,
      ),
      lfu: armRecord(
        result.references.lfu,
        protocol.primaryWindow.startToken,
        primaryEndExclusive,
      ),
    },
    arms: {
      noAction: armRecord(
        result.arms.noAction,
        protocol.primaryWindow.startToken,
        primaryEndExclusive,
      ),
      frozenDetectedAction: armRecord(
        result.arms.frozenDetectedAction,
        protocol.primaryWindow.startToken,
        primaryEndExclusive,
      ),
      perfectBoundaryAction: armRecord(
        result.arms.perfectBoundaryAction,
        protocol.primaryWindow.startToken,
        primaryEndExclusive,
      ),
      trafficShadowGatedAction: armRecord(
        result.arms.trafficShadowGatedAction,
        protocol.primaryWindow.startToken,
        primaryEndExclusive,
      ),
    },
    finiteActionOracle: result.finiteActionOracle,
    gatedPrimaryPercentChangeVsNoAction:
      result.gatedPrimaryPercentChangeVsNoAction,
    frozenPrimaryPercentChangeVsNoAction:
      result.frozenPrimaryPercentChangeVsNoAction,
    falseAction: result.falseAction,
    harmfulAction: result.harmfulAction,
    firstPermanentBreakEvenToken: result.firstPermanentBreakEvenToken,
    regretBytesAgainstFiniteOracle: result.regretBytesAgainstFiniteOracle,
    invariants: result.invariants,
  };
}

function median(values: readonly number[]): number {
  if (values.length === 0) {
    throw new RangeError("Median requires at least one value.");
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function createXorshift32(seed: number): () => number {
  let state = seed >>> 0;
  if (state === 0) {
    throw new RangeError("xorshift32 requires a non-zero seed.");
  }
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4_294_967_296;
  };
}

function percentileBounds(
  sorted: readonly number[],
  lowerPercentile: number,
  upperPercentile: number,
): { lower: number; upper: number } {
  if (sorted.length === 0) {
    throw new RangeError("Percentile bounds require at least one value.");
  }
  const last = sorted.length - 1;
  const lowerIndex = Math.floor((lowerPercentile / 100) * last);
  const upperIndex = Math.ceil((upperPercentile / 100) * last);
  return {
    lower: sorted[lowerIndex],
    upper: sorted[upperIndex],
  };
}

function clusterBootstrap(
  records: readonly CellRecord[],
  seeds: readonly number[],
  protocol: ActionabilityProtocol,
): BootstrapEndpoint {
  const valuesBySeed = new Map<number, number[]>();
  for (const seed of seeds) valuesBySeed.set(seed, []);
  for (const record of records) {
    const values = valuesBySeed.get(record.seed);
    if (!values) {
      throw new Error(`Bootstrap record used unexpected seed ${record.seed}.`);
    }
    values.push(record.gatedPrimaryPercentChangeVsNoAction);
  }

  const observedValues = [...valuesBySeed.values()].flat();
  const random = createXorshift32(protocol.bootstrap.seed);
  const estimates: number[] = [];
  let emptyResamples = 0;
  for (let sample = 0; sample < protocol.bootstrap.resamples; sample += 1) {
    const sampleValues: number[] = [];
    for (let cluster = 0; cluster < seeds.length; cluster += 1) {
      const chosenSeed = seeds[Math.floor(random() * seeds.length)];
      sampleValues.push(...(valuesBySeed.get(chosenSeed) ?? []));
    }
    if (sampleValues.length === 0) {
      emptyResamples += 1;
    } else {
      estimates.push(median(sampleValues));
    }
  }
  estimates.sort((left, right) => left - right);
  const bounds =
    estimates.length === 0
      ? null
      : percentileBounds(
          estimates,
          protocol.bootstrap.lowerPercentile,
          protocol.bootstrap.upperPercentile,
        );
  return {
    unit: "seed-cluster",
    statistic: "median-paired-percent-change",
    estimate: observedValues.length === 0 ? null : median(observedValues),
    resamplesRequested: protocol.bootstrap.resamples,
    validResamples: estimates.length,
    emptyResamples,
    prng: "xorshift32",
    seed: protocol.bootstrap.seed,
    lowerPercentile: protocol.bootstrap.lowerPercentile,
    upperPercentile: protocol.bootstrap.upperPercentile,
    lower: bounds?.lower ?? null,
    upper: bounds?.upper ?? null,
  };
}

function gate(
  requirement: string,
  observed: JsonValue,
  passed: boolean,
): GateRecord {
  return { requirement, observed, passed };
}

function assertAbruptDetectorConsistency(records: readonly CellRecord[]): void {
  const bySeed = new Map<number, CellRecord[]>();
  for (const record of records) {
    const group = bySeed.get(record.seed) ?? [];
    group.push(record);
    bySeed.set(record.seed, group);
  }
  for (const [seed, group] of bySeed) {
    const reference = group[0];
    for (const record of group.slice(1)) {
      assertEqualCanonical(
        record.detector,
        reference.detector,
        `Detector summary for seed ${seed}`,
      );
      if (record.traceSha256 !== reference.traceSha256) {
        throw new Error(`Trace SHA-256 changed across capacities for seed ${seed}.`);
      }
    }
  }
}

function allInvariantsHold(record: CellRecord): boolean {
  return (
    record.invariants.traceFingerprintsMatch &&
    record.invariants.exactRoutingSelectionsMatch &&
    record.invariants.semanticRoutingChangesZero
  );
}

function buildCoreEvidence(
  protocol: ActionabilityProtocol,
  provenance: JsonValue,
) {
  const abruptRecords: CellRecord[] = [];
  const stationaryRecords: CellRecord[] = [];
  for (const seed of protocol.confirmatorySeeds) {
    for (const gpuSlots of protocol.configuration.gpuSlots) {
      abruptRecords.push(
        buildCellRecord(protocol, "domain-shift", seed, gpuSlots),
      );
    }
    stationaryRecords.push(
      buildCellRecord(
        protocol,
        "steady",
        seed,
        protocol.configuration.stationaryGpuSlots,
      ),
    );
  }
  assertAbruptDetectorConsistency(abruptRecords);

  const uniqueAbruptBySeed = protocol.confirmatorySeeds.map((seed) => {
    const record = abruptRecords.find((candidate) => candidate.seed === seed);
    if (!record) throw new Error(`Missing abrupt record for seed ${seed}.`);
    return record;
  });
  const detectedWithin64Count = uniqueAbruptBySeed.filter(
    (record) => record.detector.detectedWithin64Tokens,
  ).length;
  const preBoundaryEvents = uniqueAbruptBySeed.reduce(
    (sum, record) => sum + record.detector.preBoundaryEvents,
    0,
  );
  const stationaryDetectorEvents = stationaryRecords.reduce(
    (sum, record) => sum + record.detector.eventTokens.length,
    0,
  );
  const stationaryGatedActions = stationaryRecords.filter(
    (record) => record.shadowDecision.act,
  ).length;
  const stationaryTokens =
    stationaryRecords.length * protocol.configuration.tokens;
  const stationaryEventsPer10000Tokens =
    (stationaryDetectorEvents / stationaryTokens) * 10_000;
  const stationaryActionsPer10000Tokens =
    (stationaryGatedActions / stationaryTokens) * 10_000;

  const oracleThreshold = protocol.gates.minimumOracleHeadroomPercent / 100;
  const oracleActionableRecords = abruptRecords.filter(
    (record) => record.finiteActionOracle.headroomFraction >= oracleThreshold,
  );
  const oracleActionableCapacities = [
    ...new Set(oracleActionableRecords.map((record) => record.gpuSlots)),
  ].sort((left, right) => left - right);
  const allCellBootstrap = clusterBootstrap(
    abruptRecords,
    protocol.confirmatorySeeds,
    protocol,
  );
  const oracleActionableBootstrap = clusterBootstrap(
    oracleActionableRecords,
    protocol.confirmatorySeeds,
    protocol,
  );
  const maximumRegressionPercent = Math.max(
    ...abruptRecords.map(
      (record) => record.gatedPrimaryPercentChangeVsNoAction,
    ),
  );
  const executedActions = abruptRecords.filter(
    (record) => record.shadowDecision.act,
  );
  const harmfulExecutedActions = executedActions.filter(
    (record) => record.harmfulAction,
  ).length;
  const harmfulExecutedActionRate =
    executedActions.length === 0
      ? 0
      : harmfulExecutedActions / executedActions.length;
  const falseActions = executedActions.filter(
    (record) => record.falseAction,
  ).length;
  const medianGatedPrimaryBytes = median(
    abruptRecords.map(
      (record) => record.arms.trafficShadowGatedAction.primaryWindowBytes,
    ),
  );
  const medianFrozenPrimaryBytes = median(
    abruptRecords.map(
      (record) => record.arms.frozenDetectedAction.primaryWindowBytes,
    ),
  );
  const semanticAndRoutingInvariantsHold = [
    ...abruptRecords,
    ...stationaryRecords,
  ].every(allInvariantsHold);

  const gatesWithoutRerun: Record<string, GateRecord> = {
    completeCellsAndRouting: gate(
      `Exactly ${protocol.gates.completeCells} abrupt cells complete with identical routing in every arm.`,
      canonicalize({
        expectedCells: protocol.gates.completeCells,
        observedCells: abruptRecords.length,
        allRoutingInvariantsHold: semanticAndRoutingInvariantsHold,
      }),
      abruptRecords.length === protocol.gates.completeCells &&
        semanticAndRoutingInvariantsHold,
    ),
    detector: gate(
      `At least ${protocol.gates.minimumDetectedWithin64} of 30 shifts are detected within 64 tokens with at most ${protocol.gates.maximumPreBoundaryEvents} pre-boundary events.`,
      canonicalize({ detectedWithin64Count, preBoundaryEvents }),
      detectedWithin64Count >= protocol.gates.minimumDetectedWithin64 &&
        preBoundaryEvents <= protocol.gates.maximumPreBoundaryEvents,
    ),
    stationaryControls: gate(
      `Detector events and gated actions are each at most ${protocol.gates.maximumStationaryEventsPer10000Tokens} per 10,000 stationary tokens.`,
      canonicalize({
        stationaryTokens,
        stationaryDetectorEvents,
        stationaryGatedActions,
        stationaryEventsPer10000Tokens,
        stationaryActionsPer10000Tokens,
      }),
      stationaryEventsPer10000Tokens <=
        protocol.gates.maximumStationaryEventsPer10000Tokens &&
        stationaryActionsPer10000Tokens <=
          protocol.gates.maximumStationaryEventsPer10000Tokens,
    ),
    oracleCoverage: gate(
      `At least ${protocol.gates.minimumOracleActionableCells} cells across ${protocol.gates.minimumOracleActionableCapacities} capacities have at least ${protocol.gates.minimumOracleHeadroomPercent}% finite-action-oracle headroom.`,
      canonicalize({
        actionableCells: oracleActionableRecords.length,
        actionableCapacities: oracleActionableCapacities,
      }),
      oracleActionableRecords.length >=
        protocol.gates.minimumOracleActionableCells &&
        oracleActionableCapacities.length >=
          protocol.gates.minimumOracleActionableCapacities,
    ),
    oracleActionableEffect: gate(
      `Oracle-actionable cells improve median primary traffic by at least ${protocol.gates.minimumOracleActionableMedianImprovementPercent}% and the cluster-bootstrap upper bound is below zero.`,
      canonicalize(oracleActionableBootstrap),
      oracleActionableBootstrap.estimate !== null &&
        oracleActionableBootstrap.estimate <=
          -protocol.gates.minimumOracleActionableMedianImprovementPercent &&
        oracleActionableBootstrap.validResamples ===
          protocol.bootstrap.resamples &&
        (!protocol.gates.requireBootstrapUpperBelowZero ||
          (oracleActionableBootstrap.upper !== null &&
            oracleActionableBootstrap.upper < 0)),
    ),
    maximumRegression: gate(
      `No abrupt cell regresses by more than ${protocol.gates.maximumAnyCellRegressionPercent}% versus no action.`,
      maximumRegressionPercent,
      maximumRegressionPercent <=
        protocol.gates.maximumAnyCellRegressionPercent,
    ),
    harmfulExecutedActions: gate(
      `At most ${protocol.gates.maximumHarmfulExecutedActionRate} of executed gated actions are harmful.`,
      canonicalize({
        executedActions: executedActions.length,
        harmfulExecutedActions,
        harmfulExecutedActionRate,
        falseActions,
      }),
      harmfulExecutedActionRate <=
        protocol.gates.maximumHarmfulExecutedActionRate,
    ),
    gatedMedianNoWorseThanFrozen: gate(
      "The gated arm's median primary traffic is no worse than the frozen detected action.",
      canonicalize({ medianGatedPrimaryBytes, medianFrozenPrimaryBytes }),
      !protocol.gates.requireGatedMedianNoWorseThanFrozen ||
        medianGatedPrimaryBytes <= medianFrozenPrimaryBytes,
    ),
    semanticRoutingChangesZero: gate(
      "Every arm preserves the exact trace and reports zero semantic routing changes.",
      semanticAndRoutingInvariantsHold,
      !protocol.gates.requireSemanticRoutingChangesZero ||
        semanticAndRoutingInvariantsHold,
    ),
  };

  return {
    schemaVersion: 1,
    protocol: {
      id: protocol.protocol,
      status: protocol.status,
      canonicalSha256: sha256Canonical(protocol),
    },
    provenance,
    evidenceBoundary: protocol.evidenceBoundary,
    configuration: protocol.configuration,
    detector: protocol.detector,
    shadowGate: protocol.shadowGate,
    primaryWindow: protocol.primaryWindow,
    bootstrap: {
      allCells: allCellBootstrap,
      oracleActionableCells: oracleActionableBootstrap,
      percentileIndexRule:
        "lower=floor(p*(B-1)); upper=ceil(p*(B-1))",
    },
    records: {
      abrupt: abruptRecords,
      stationary: stationaryRecords,
    },
    summary: {
      abruptCells: abruptRecords.length,
      stationaryCells: stationaryRecords.length,
      detectedWithin64Count,
      preBoundaryEvents,
      stationaryTokens,
      stationaryDetectorEvents,
      stationaryGatedActions,
      stationaryEventsPer10000Tokens,
      stationaryActionsPer10000Tokens,
      oracleActionableCells: oracleActionableRecords.length,
      oracleActionableCapacities,
      maximumRegressionPercent,
      executedActions: executedActions.length,
      harmfulExecutedActions,
      harmfulExecutedActionRate,
      falseActions,
      medianGatedPrimaryBytes,
      medianFrozenPrimaryBytes,
    },
    gatesWithoutRerun,
  };
}

function validatedFullCommit(value: string, label: string): string {
  if (!/^[0-9a-f]{40}$/.test(value)) {
    throw new Error(`${label} must be a full 40-character Git commit.`);
  }
  return value;
}

function buildCoreDocument(
  protocol: ActionabilityProtocol,
  executionCommit = validatedFullCommit(
    git(["rev-parse", "HEAD"]),
    "Execution commit",
  ),
) {
  const provenance = {
    protocolCommit: latestCommitFor([
      "evidence/actionability-v1/protocol.json",
    ]),
    executionCommit: validatedFullCommit(
      executionCommit,
      "Execution commit",
    ),
    implementationCommit: latestCommitFor(implementationPaths),
    implementationFileCommits: Object.fromEntries(
      implementationPaths.map((path) => [path, latestCommitFor([path])]),
    ),
    cleanTreeRequiredByRunner: true,
  };
  return buildCoreEvidence(protocol, canonicalize(provenance));
}

function finalizeEvidence(
  protocol: ActionabilityProtocol,
  coreWithGates: ReturnType<typeof buildCoreDocument>,
  byteIdenticalCleanProcessRerun: boolean,
) {
  const gates = {
    ...coreWithGates.gatesWithoutRerun,
    byteIdenticalRerun: gate(
      "A fresh Node.js process on the same clean committed tree produces byte-identical canonical core evidence.",
      canonicalize({
        method: "fresh-node-process-clean-tree-core-comparison-v1",
        canonicalCoreSha256: sha256Canonical(coreWithGates),
        byteIdentical: byteIdenticalCleanProcessRerun,
      }),
      !protocol.gates.requireByteIdenticalRerun ||
        byteIdenticalCleanProcessRerun,
    ),
  };
  const { gatesWithoutRerun: _omitted, ...core } = coreWithGates;
  void _omitted;
  const payload = {
    ...core,
    gates,
    carryForward: Object.values(gates).every((entry) => entry.passed),
  };
  return {
    canonicalization: "recursive-lexicographic-object-keys-v1",
    canonicalPayloadSha256: sha256Canonical(payload),
    payload,
  };
}

function recordedExecutionCommit(): string {
  const parsed = JSON.parse(readFileSync(resultsUrl, "utf8")) as {
    payload?: { provenance?: { executionCommit?: unknown } };
  };
  const executionCommit = parsed.payload?.provenance?.executionCommit;
  if (typeof executionCommit !== "string") {
    throw new Error(
      `${resultsRepositoryPath} does not record an execution commit.`,
    );
  }
  return validatedFullCommit(executionCommit, "Recorded execution commit");
}

function emitCleanCoreFromChildProcess(): string {
  return execFileSync(
    process.execPath,
    [
      "--import",
      "tsx",
      fileURLToPath(import.meta.url),
      internalEmitMode,
    ],
    {
      cwd: repositoryRoot,
      encoding: "utf8",
      env: {
        ...process.env,
        STRATAMOE_ACTIONABILITY_INTERNAL: "1",
      },
      maxBuffer: 64 * 1024 * 1024,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
}

function main(): void {
  const [mode, ...extraArguments] = process.argv.slice(2);
  if (
    extraArguments.length > 0 ||
    (mode !== "--write" && mode !== "--verify" && mode !== internalEmitMode)
  ) {
    throw new Error(
      "Usage: tsx scripts/run-actionability-pilot.ts --write|--verify",
    );
  }

  if (mode === internalEmitMode) {
    if (process.env.STRATAMOE_ACTIONABILITY_INTERNAL !== "1") {
      throw new Error("The internal core-emission mode is runner-only.");
    }
    assertWritePreconditions();
    const protocol = loadProtocol();
    assertProtocol(protocol);
    process.stdout.write(canonicalFileJson(buildCoreDocument(protocol)));
    return;
  }

  if (mode === "--write") assertWritePreconditions();
  else assertVerifyPreconditions();

  const protocol = loadProtocol();
  assertProtocol(protocol);

  if (mode === "--write") {
    const core = buildCoreDocument(protocol);
    const firstCore = canonicalFileJson(core);
    const secondCore = emitCleanCoreFromChildProcess();
    const serialized = canonicalFileJson(
      finalizeEvidence(protocol, core, firstCore === secondCore),
    );
    writeFileSync(resultsUrl, serialized, { encoding: "utf8", flag: "wx" });
    process.stdout.write(`Wrote ${resultsRepositoryPath}.\n`);
    return;
  }

  const core = buildCoreDocument(protocol, recordedExecutionCommit());
  const serialized = canonicalFileJson(finalizeEvidence(protocol, core, true));
  const existing = readFileSync(resultsUrl, "utf8");
  if (existing !== serialized) {
    throw new Error(
      `${resultsRepositoryPath} is not byte-identical to the recomputed evidence.`,
    );
  }
  process.stdout.write(`Verified ${resultsRepositoryPath} byte-for-byte.\n`);
}

try {
  main();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`${message}\n`);
  process.exitCode = 1;
}
