import assert from "node:assert/strict";
import test from "node:test";

import {
  ACTION_HORIZON_TOKENS,
  FROZEN_ACTION_CONTROLS,
  NO_ACTION_CONTROLS,
  SHADOW_MINIMUM_SAVING_FRACTION,
  SHADOW_OBSERVATION_TOKENS,
  decideTrafficShadow,
  firstPermanentBreakEvenToken,
  modeledBytes,
  runActionabilityCell,
} from "../lib/actionability";
import {
  DEFAULT_CONFIG,
  generateRouterTrace,
  runSimulation,
  type RouterTraceV2,
} from "../lib/simulator";

const DEV_SEED = 0;
const TOKENS = 512;
const BOUNDARY = 256;

function devTrace() {
  return generateRouterTrace({
    scenario: "domain-shift",
    seed: DEV_SEED,
    tokens: TOKENS,
    layers: DEFAULT_CONFIG.layers,
    expertsPerLayer: DEFAULT_CONFIG.expertsPerLayer,
    topK: DEFAULT_CONFIG.topK,
  });
}

function devConfig(gpuSlots = 16) {
  return {
    ...DEFAULT_CONFIG,
    scenario: "domain-shift" as const,
    seed: DEV_SEED,
    tokens: TOKENS,
    gpuSlots,
  };
}

test("actionability controls preserve the frozen detector intervention", () => {
  assert.deepEqual(NO_ACTION_CONTROLS, {
    shiftCachePrefetch: false,
    shiftCacheJsdReweighting: false,
    shiftCacheTransitionRetention: false,
    shiftCachePersistentDetector: true,
    shiftCacheTriggeredReweighting: false,
  });
  assert.deepEqual(FROZEN_ACTION_CONTROLS, {
    ...NO_ACTION_CONTROLS,
    shiftCacheTriggeredReweighting: true,
  });
  assert.equal(SHADOW_OBSERVATION_TOKENS, 12);
  assert.equal(SHADOW_MINIMUM_SAVING_FRACTION, 0.05);
  assert.equal(ACTION_HORIZON_TOKENS, 64);
});

test("modeledBytes sums an exact half-open token window", () => {
  const trace = devTrace();
  const result = runSimulation(
    { ...devConfig(), policy: "shift-cache" },
    trace,
    NO_ACTION_CONTROLS,
  );
  const expected = result.timeline
    .slice(BOUNDARY, BOUNDARY + ACTION_HORIZON_TOKENS)
    .reduce((sum, point) => sum + point.bytesTransferred, 0);
  assert.equal(
    modeledBytes(result, BOUNDARY, BOUNDARY + ACTION_HORIZON_TOKENS),
    expected,
  );
  assert.throws(() => modeledBytes(result, -1, 2), /outside/i);
  assert.throws(
    () => modeledBytes(result, 0, result.timeline.length + 1),
    /outside/i,
  );
});

test("the shadow observes exactly 12 causal tokens and preserves the horizon remainder", () => {
  const trace = devTrace();
  const config = { ...devConfig(8), policy: "shift-cache" } as const;
  const noAction = runSimulation(config, trace, NO_ACTION_CONTROLS);
  const frozenAction = runSimulation(config, trace, FROZEN_ACTION_CONTROLS);
  const detectedEventToken = noAction.timeline.find(
    (point) => point.shiftDetected,
  )?.token;
  assert.notEqual(detectedEventToken, undefined);

  const decision = decideTrafficShadow(
    noAction,
    frozenAction,
    detectedEventToken ?? null,
  );
  assert.equal(decision.observationStartToken, (detectedEventToken ?? 0) + 1);
  assert.equal(
    decision.observationEndTokenExclusive,
    (detectedEventToken ?? 0) + 13,
  );
  assert.equal(decision.decisionToken, (detectedEventToken ?? 0) + 12);
  if (decision.act) {
    assert.equal(
      decision.appliedActionStartToken,
      decision.observationEndTokenExclusive,
    );
    assert.equal(
      decision.appliedActionEndTokenExclusive,
      (detectedEventToken ?? 0) + 65,
    );
    assert.equal(
      (decision.appliedActionEndTokenExclusive ?? 0) -
        (decision.appliedActionStartToken ?? 0),
      52,
    );
  } else {
    assert.match(decision.reason, /below-threshold|zero-baseline-traffic/);
  }
});

test("an incomplete shadow window abstains", () => {
  const trace = devTrace();
  const config = { ...devConfig(), policy: "shift-cache" } as const;
  const noAction = runSimulation(config, trace, NO_ACTION_CONTROLS);
  const frozenAction = runSimulation(config, trace, FROZEN_ACTION_CONTROLS);
  const detectedEventToken = TOKENS - 5;
  const decision = decideTrafficShadow(
    noAction,
    frozenAction,
    detectedEventToken,
    TOKENS,
  );
  assert.equal(decision.act, false);
  assert.equal(decision.reason, "incomplete-window");
});

test("a late event abstains when the frozen horizon would exceed the trace", () => {
  const trace = devTrace();
  const config = { ...devConfig(), policy: "shift-cache" } as const;
  const noAction = runSimulation(config, trace, NO_ACTION_CONTROLS);
  const frozenAction = runSimulation(config, trace, FROZEN_ACTION_CONTROLS);
  const detectedEventToken = TOKENS - ACTION_HORIZON_TOKENS;
  const decision = decideTrafficShadow(
    noAction,
    frozenAction,
    detectedEventToken,
    TOKENS,
  );

  assert.ok(
    detectedEventToken + 1 + SHADOW_OBSERVATION_TOKENS <= TOKENS,
  );
  assert.equal(decision.act, false);
  assert.equal(decision.reason, "incomplete-window");
});

test("permanent break-even includes exact repayment", () => {
  const trace = devTrace();
  const config = { ...devConfig(), policy: "shift-cache" } as const;
  const noAction = runSimulation(config, trace, NO_ACTION_CONTROLS);
  const actionStart = BOUNDARY;
  const gatedAction = {
    ...noAction,
    timeline: noAction.timeline.map((point, token) => {
      if (token === actionStart) {
        return { ...point, bytesTransferred: point.bytesTransferred + 100 };
      }
      if (token === actionStart + 1) {
        return { ...point, bytesTransferred: point.bytesTransferred - 100 };
      }
      return { ...point };
    }),
  };

  assert.equal(
    firstPermanentBreakEvenToken(
      noAction,
      gatedAction,
      BOUNDARY,
      BOUNDARY + ACTION_HORIZON_TOKENS,
      actionStart,
    ),
    actionStart + 1,
  );
});

test("the shadow acts at exactly five percent saving and abstains on a tie", () => {
  const trace = devTrace();
  const config = { ...devConfig(), policy: "shift-cache" } as const;
  const base = runSimulation(config, trace, NO_ACTION_CONTROLS);
  const withObservationBytes = (bytes: number) => ({
    ...base,
    timeline: base.timeline.map((point, token) => ({
      ...point,
      bytesTransferred: token >= 1 && token < 13 ? bytes : point.bytesTransferred,
    })),
  });

  const thresholdDecision = decideTrafficShadow(
    withObservationBytes(100),
    withObservationBytes(95),
    0,
  );
  assert.equal(thresholdDecision.estimatedSavingFraction, 0.05);
  assert.equal(thresholdDecision.act, true);
  assert.equal(thresholdDecision.reason, "passed");

  const tieDecision = decideTrafficShadow(
    withObservationBytes(100),
    withObservationBytes(100),
    0,
  );
  assert.equal(tieDecision.estimatedSavingFraction, 0);
  assert.equal(tieDecision.act, false);
  assert.equal(tieDecision.reason, "below-threshold");
});

test("actionability cell shares routing and reports the frozen finite oracle", () => {
  const cell = runActionabilityCell({
    config: devConfig(32),
    trace: devTrace(),
    knownBoundaryToken: BOUNDARY,
  });

  assert.deepEqual(cell.primaryWindow, {
    startToken: BOUNDARY,
    endTokenExclusive: BOUNDARY + ACTION_HORIZON_TOKENS,
  });
  assert.equal(cell.detector.preBoundaryEvents, 0);
  assert.equal(cell.detector.detectedWithin64Tokens, true);
  assert.ok(cell.detector.firstPostBoundaryDelayTokens !== null);
  assert.ok((cell.detector.firstPostBoundaryDelayTokens ?? 65) <= 64);
  assert.equal(cell.invariants.traceFingerprintsMatch, true);
  assert.equal(cell.invariants.exactRoutingSelectionsMatch, true);
  assert.equal(cell.invariants.semanticRoutingChangesZero, true);
  assert.equal(cell.shadowDecision.act, true);
  assert.equal(cell.finiteActionOracle.choice, "perfect-boundary-action");
  assert.ok(cell.finiteActionOracle.headroomFraction >= 0);
  assert.equal(cell.falseAction, false);
  assert.equal(cell.harmfulAction, true);
  assert.equal(cell.firstPermanentBreakEvenToken, null);
  assert.equal(cell.references.lru.result.metrics.semanticRoutingChanges, 0);
  assert.equal(cell.references.lfu.result.metrics.semanticRoutingChanges, 0);
});

test("changing only the post-observation suffix cannot change the shadow decision", () => {
  const trace = devTrace();
  const config = { ...devConfig(8), policy: "shift-cache" } as const;
  const originalNoAction = runSimulation(config, trace, NO_ACTION_CONTROLS);
  const originalFrozen = runSimulation(config, trace, FROZEN_ACTION_CONTROLS);
  const detectedEventToken = originalNoAction.timeline.find(
    (point) => point.shiftDetected,
  )?.token;
  assert.notEqual(detectedEventToken, undefined);
  const suffixStart = (detectedEventToken ?? 0) + 13;

  const changedTrace: RouterTraceV2 = {
    ...trace,
    selections: trace.selections.map((tokenSelections, token) =>
      token < suffixStart
        ? tokenSelections.map((layer) => [...layer])
        : tokenSelections.map((layer) =>
            layer.map(
              (expert) => (expert + 1) % trace.expertsPerLayer,
            ),
          ),
    ),
  };
  const changedNoAction = runSimulation(
    config,
    changedTrace,
    NO_ACTION_CONTROLS,
  );
  const changedFrozen = runSimulation(
    config,
    changedTrace,
    FROZEN_ACTION_CONTROLS,
  );
  const changedEventToken = changedNoAction.timeline.find(
    (point) => point.shiftDetected,
  )?.token;

  assert.equal(changedEventToken, detectedEventToken);
  const originalDecision = decideTrafficShadow(
    originalNoAction,
    originalFrozen,
    detectedEventToken ?? null,
  );
  const changedDecision = decideTrafficShadow(
    changedNoAction,
    changedFrozen,
    changedEventToken ?? null,
  );
  assert.deepEqual(
    {
      act: changedDecision.act,
      reason: changedDecision.reason,
      observationStartToken: changedDecision.observationStartToken,
      observationEndTokenExclusive:
        changedDecision.observationEndTokenExclusive,
      noActionObservationBytes: changedDecision.noActionObservationBytes,
      frozenActionObservationBytes: changedDecision.frozenActionObservationBytes,
      estimatedSavingFraction: changedDecision.estimatedSavingFraction,
      appliedActionStartToken: changedDecision.appliedActionStartToken,
      appliedActionEndTokenExclusive:
        changedDecision.appliedActionEndTokenExclusive,
    },
    {
      act: originalDecision.act,
      reason: originalDecision.reason,
      observationStartToken: originalDecision.observationStartToken,
      observationEndTokenExclusive:
        originalDecision.observationEndTokenExclusive,
      noActionObservationBytes: originalDecision.noActionObservationBytes,
      frozenActionObservationBytes: originalDecision.frozenActionObservationBytes,
      estimatedSavingFraction: originalDecision.estimatedSavingFraction,
      appliedActionStartToken: originalDecision.appliedActionStartToken,
      appliedActionEndTokenExclusive:
        originalDecision.appliedActionEndTokenExclusive,
    },
  );
});
