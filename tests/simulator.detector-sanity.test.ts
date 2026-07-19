import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DEFAULT_CONFIG,
  generateRouterTrace,
  runSimulation,
  type SimulationControls,
} from "../lib/simulator";

const evidenceUrl = new URL(
  "../evidence/synthetic/detector-sanity.json",
  import.meta.url,
);

interface DetectorEvidence {
  benchmark: string;
  evidenceBoundary: string;
  preregistration: { publicPlanCommit: string };
  configuration: {
    seeds: number[];
    tokens: number;
    knownShiftToken: number;
    postShiftWindowTokens: number;
    shiftCacheParameters: {
      persistentThresholdTokens: number;
      detectorCooldownTokens: number;
      triggeredReweightingTokens: number;
    };
    fixedControls: SimulationControls;
    triggeredControls: SimulationControls;
  };
  records: Array<{
    seed: number;
    domainShift: {
      traceFingerprint: string;
      detectedEventTokens: number[];
      preShiftEvents: number;
      detectionDelayTokens: number | null;
      detectedWithin64Tokens: boolean;
      fixedPostShiftModeledLinkBytes: number;
      triggeredPostShiftModeledLinkBytes: number;
      postShiftModeledLinkBytesPercentChange: number;
    };
    steady: {
      detectedEventTokens: number[];
      wholeRunModeledLinkBytesPercentChange: number;
    };
  }>;
  summary: {
    detectedWithin64Count: number;
    medianDetectionDelayTokens: number;
    stationaryFalseTriggersPer10000Tokens: number;
    medianPostShiftModeledLinkBytesPercentChange: number;
    postShiftMedianBootstrap95PercentInterval: {
      lower: number;
      upper: number;
    };
    medianStationaryWholeRunModeledLinkBytesPercentChange: number;
  };
  gates: Record<string, { requirement: string; passed: boolean }>;
  carryForward: boolean;
}

test("preregistered detector evidence preserves the failed carry-forward gate", () => {
  const evidence = JSON.parse(
    readFileSync(evidenceUrl, "utf8"),
  ) as DetectorEvidence;

  assert.equal(evidence.benchmark, "synthetic-persistent-detector-sanity-v1");
  assert.match(evidence.evidenceBoundary, /deterministic synthetic fixtures/i);
  assert.equal(
    evidence.preregistration.publicPlanCommit,
    "fe6042b6cd628d4976c2069a94d2739614ce9ce9",
  );
  assert.deepEqual(evidence.configuration.seeds, [
    ...Array.from({ length: 30 }, (_, index) => 2300 + index),
  ]);
  assert.equal(evidence.configuration.tokens, 1024);
  assert.equal(evidence.configuration.knownShiftToken, 512);
  assert.equal(evidence.configuration.postShiftWindowTokens, 64);
  assert.deepEqual(evidence.configuration.shiftCacheParameters, {
    minimumShortWindowTokens: 4,
    maximumShortWindowTokens: 12,
    longWindowMultiplier: 4,
    shiftThresholdBits: 0.28,
    rearmThresholdBits: 0.12,
    minimumTransitionObservations: 2,
    maximumPrefetchesPerToken: 2,
    persistentThresholdTokens: 3,
    detectorCooldownTokens: 64,
    triggeredReweightingTokens: 64,
  });

  assert.equal(evidence.records.length, 30);
  assert.ok(
    evidence.records.every(
      (record) =>
        record.domainShift.detectedWithin64Tokens &&
        record.domainShift.detectionDelayTokens === 6 &&
        record.domainShift.preShiftEvents === 0 &&
        record.steady.detectedEventTokens.length === 0 &&
        record.domainShift.postShiftModeledLinkBytesPercentChange > 0,
    ),
  );
  assert.equal(evidence.summary.detectedWithin64Count, 30);
  assert.equal(evidence.summary.medianDetectionDelayTokens, 6);
  assert.equal(evidence.summary.stationaryFalseTriggersPer10000Tokens, 0);
  assert.equal(
    evidence.summary.medianPostShiftModeledLinkBytesPercentChange,
    11.52386217,
  );
  assert.deepEqual(
    evidence.summary.postShiftMedianBootstrap95PercentInterval,
    {
      resamples: 10000,
      seed: 2339,
      lowerPercentile: 2.5,
      upperPercentile: 97.5,
      lower: 10.78035747,
      upper: 12.63942461,
    },
  );
  assert.equal(
    evidence.summary.medianStationaryWholeRunModeledLinkBytesPercentChange,
    0,
  );
  assert.equal(evidence.gates.detection.passed, true);
  assert.equal(evidence.gates.stationaryFalseTriggers.passed, true);
  assert.equal(evidence.gates.postShiftTraffic.passed, false);
  assert.equal(evidence.gates.stationaryTraffic.passed, true);
  assert.equal(evidence.carryForward, false);
});

test("one checked-in detector record replays without routing or prefetch changes", () => {
  const evidence = JSON.parse(
    readFileSync(evidenceUrl, "utf8"),
  ) as DetectorEvidence;
  const record = evidence.records[0];
  const traceConfig = {
    scenario: "domain-shift",
    seed: record.seed,
    tokens: evidence.configuration.tokens,
    layers: DEFAULT_CONFIG.layers,
    expertsPerLayer: DEFAULT_CONFIG.expertsPerLayer,
    topK: DEFAULT_CONFIG.topK,
  } as const;
  const trace = generateRouterTrace(traceConfig);
  const config = { ...DEFAULT_CONFIG, ...traceConfig, policy: "shift-cache" } as const;
  const fixed = runSimulation(
    config,
    trace,
    evidence.configuration.fixedControls,
  );
  const triggered = runSimulation(
    config,
    trace,
    evidence.configuration.triggeredControls,
  );
  const start = evidence.configuration.knownShiftToken;
  const end = start + evidence.configuration.postShiftWindowTokens;
  const modeledBytes = (result: typeof fixed) =>
    result.timeline
      .slice(start, end)
      .reduce((sum, point) => sum + point.bytesTransferred, 0);

  assert.equal(triggered.traceFingerprint, record.domainShift.traceFingerprint);
  assert.deepEqual(triggered.trace.selections, fixed.trace.selections);
  assert.deepEqual(
    triggered.timeline
      .filter((point) => point.shiftDetected)
      .map((point) => point.token),
    record.domainShift.detectedEventTokens,
  );
  assert.equal(
    modeledBytes(fixed),
    record.domainShift.fixedPostShiftModeledLinkBytes,
  );
  assert.equal(
    modeledBytes(triggered),
    record.domainShift.triggeredPostShiftModeledLinkBytes,
  );
  assert.equal(triggered.metrics.prefetchesIssued, 0);
  assert.equal(triggered.metrics.prefetchBytesTransferred, 0);
  assert.equal(triggered.metrics.semanticRoutingChanges, 0);
});
