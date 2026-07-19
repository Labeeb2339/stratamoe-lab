import { mkdirSync, writeFileSync } from "node:fs";
import {
  DEFAULT_CONFIG,
  DEFAULT_SIMULATION_CONTROLS,
  SHIFT_CACHE_PARAMETERS,
  generateRouterTrace,
  runSimulation,
  type ScenarioId,
  type SimulationControls,
  type SimulationResult,
} from "../lib/simulator";

const evidenceDirectory = new URL("../evidence/synthetic/", import.meta.url);
const outputUrl = new URL("detector-sanity.json", evidenceDirectory);

const seeds = Array.from({ length: 30 }, (_, index) => 2300 + index);
const tokens = 1024;
const knownShiftToken = tokens / 2;
const postShiftWindowTokens = 64;
const bootstrapResamples = 10_000;
const bootstrapSeed = 2339;

const fixedControls: SimulationControls = {
  ...DEFAULT_SIMULATION_CONTROLS,
  shiftCachePrefetch: false,
  shiftCacheJsdReweighting: false,
  shiftCacheTransitionRetention: false,
  shiftCachePersistentDetector: true,
  shiftCacheTriggeredReweighting: false,
};
const triggeredControls: SimulationControls = {
  ...fixedControls,
  shiftCacheTriggeredReweighting: true,
};

function round(value: number, digits = 8): number {
  return Number(value.toFixed(digits));
}

function median(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4_294_967_296;
  };
}

function pairedMedianBootstrap(values: readonly number[]) {
  const random = createRandom(bootstrapSeed);
  const estimates: number[] = [];
  for (let sample = 0; sample < bootstrapResamples; sample += 1) {
    const resampled: number[] = [];
    for (let index = 0; index < values.length; index += 1) {
      resampled.push(values[Math.floor(random() * values.length)]);
    }
    estimates.push(median(resampled));
  }
  estimates.sort((left, right) => left - right);
  return {
    resamples: bootstrapResamples,
    seed: bootstrapSeed,
    lowerPercentile: 2.5,
    upperPercentile: 97.5,
    lower: round(estimates[Math.floor(0.025 * (estimates.length - 1))]),
    upper: round(estimates[Math.ceil(0.975 * (estimates.length - 1))]),
  };
}

function percentChange(after: number, before: number): number {
  return before === 0 ? 0 : round(((after - before) / before) * 100);
}

function modeledBytes(
  result: SimulationResult,
  start = 0,
  endExclusive = result.timeline.length,
): number {
  return result.timeline
    .slice(start, endExclusive)
    .reduce((sum, point) => sum + point.bytesTransferred, 0);
}

function simulation(seed: number, scenario: ScenarioId, controls: SimulationControls) {
  const traceConfig = {
    scenario,
    seed,
    tokens,
    layers: DEFAULT_CONFIG.layers,
    expertsPerLayer: DEFAULT_CONFIG.expertsPerLayer,
    topK: DEFAULT_CONFIG.topK,
  };
  const trace = generateRouterTrace(traceConfig);
  return runSimulation(
    {
      ...DEFAULT_CONFIG,
      ...traceConfig,
      policy: "shift-cache",
    },
    trace,
    controls,
  );
}

const records = seeds.map((seed) => {
  const domainFixed = simulation(seed, "domain-shift", fixedControls);
  const domainTriggered = simulation(seed, "domain-shift", triggeredControls);
  const steadyFixed = simulation(seed, "steady", fixedControls);
  const steadyTriggered = simulation(seed, "steady", triggeredControls);

  const domainEvents = domainTriggered.timeline
    .filter((point) => point.shiftDetected)
    .map((point) => point.token);
  const firstPostShiftEvent = domainEvents.find(
    (token) => token >= knownShiftToken,
  );
  const detectionDelay =
    firstPostShiftEvent === undefined
      ? null
      : firstPostShiftEvent - knownShiftToken;
  const postShiftEnd = knownShiftToken + postShiftWindowTokens;
  const fixedPostShiftBytes = modeledBytes(
    domainFixed,
    knownShiftToken,
    postShiftEnd,
  );
  const triggeredPostShiftBytes = modeledBytes(
    domainTriggered,
    knownShiftToken,
    postShiftEnd,
  );

  return {
    seed,
    domainShift: {
      traceFingerprint: domainTriggered.traceFingerprint,
      detectedEventTokens: domainEvents,
      preShiftEvents: domainEvents.filter((token) => token < knownShiftToken)
        .length,
      firstPostShiftEvent: firstPostShiftEvent ?? null,
      detectionDelayTokens: detectionDelay,
      detectedWithin64Tokens:
        detectionDelay !== null &&
        detectionDelay >= 0 &&
        detectionDelay <= postShiftWindowTokens,
      fixedPostShiftModeledLinkBytes: fixedPostShiftBytes,
      triggeredPostShiftModeledLinkBytes: triggeredPostShiftBytes,
      postShiftModeledLinkBytesPercentChange: percentChange(
        triggeredPostShiftBytes,
        fixedPostShiftBytes,
      ),
    },
    steady: {
      traceFingerprint: steadyTriggered.traceFingerprint,
      detectedEventTokens: steadyTriggered.timeline
        .filter((point) => point.shiftDetected)
        .map((point) => point.token),
      fixedWholeRunModeledLinkBytes: steadyFixed.metrics.totalBytesTransferred,
      triggeredWholeRunModeledLinkBytes:
        steadyTriggered.metrics.totalBytesTransferred,
      wholeRunModeledLinkBytesPercentChange: percentChange(
        steadyTriggered.metrics.totalBytesTransferred,
        steadyFixed.metrics.totalBytesTransferred,
      ),
    },
  };
});

const detectedWithin64Count = records.filter(
  (record) => record.domainShift.detectedWithin64Tokens,
).length;
const stationaryFalseTriggers = records.reduce(
  (sum, record) => sum + record.steady.detectedEventTokens.length,
  0,
);
const domainPreShiftTriggers = records.reduce(
  (sum, record) => sum + record.domainShift.preShiftEvents,
  0,
);
const stationaryTokens = seeds.length * tokens;
const postShiftPercentChanges = records.map(
  (record) => record.domainShift.postShiftModeledLinkBytesPercentChange,
);
const stationaryPercentChanges = records.map(
  (record) => record.steady.wholeRunModeledLinkBytesPercentChange,
);
const bootstrap = pairedMedianBootstrap(postShiftPercentChanges);
const summary = {
  detectedWithin64Count,
  missedOrLateCount: seeds.length - detectedWithin64Count,
  medianDetectionDelayTokens: round(
    median(
      records
        .map((record) => record.domainShift.detectionDelayTokens)
        .filter((delay): delay is number => delay !== null),
    ),
  ),
  stationaryFalseTriggers,
  stationaryFalseTriggersPer10000Tokens: round(
    (stationaryFalseTriggers / stationaryTokens) * 10_000,
  ),
  domainPreShiftTriggers,
  domainPreShiftTriggersPer10000Tokens: round(
    (domainPreShiftTriggers / (seeds.length * knownShiftToken)) * 10_000,
  ),
  medianPostShiftModeledLinkBytesPercentChange: round(
    median(postShiftPercentChanges),
  ),
  postShiftMedianBootstrap95PercentInterval: bootstrap,
  medianStationaryWholeRunModeledLinkBytesPercentChange: round(
    median(stationaryPercentChanges),
  ),
  maximumStationaryWholeRunModeledLinkBytesPercentChange: round(
    Math.max(...stationaryPercentChanges),
  ),
};

const gates = {
  detection: {
    requirement: "At least 27 of 30 abrupt shifts detected within 64 tokens.",
    passed: summary.detectedWithin64Count >= 27,
  },
  stationaryFalseTriggers: {
    requirement: "At most one false trigger per 10,000 stationary tokens.",
    passed: summary.stationaryFalseTriggersPer10000Tokens <= 1,
  },
  postShiftTraffic: {
    requirement:
      "Median first-64-token post-shift modeled-link-byte change is at most -5%, with paired-bootstrap 95% interval entirely below zero.",
    passed:
      summary.medianPostShiftModeledLinkBytesPercentChange <= -5 &&
      summary.postShiftMedianBootstrap95PercentInterval.upper < 0,
  },
  stationaryTraffic: {
    requirement:
      "Median stationary whole-run modeled-link-byte regression is no greater than 2%.",
    passed:
      summary.medianStationaryWholeRunModeledLinkBytesPercentChange <= 2,
  },
};

const output = {
  benchmark: "synthetic-persistent-detector-sanity-v1",
  preregistration: {
    publicPlanCommit: "fe6042b6cd628d4976c2069a94d2739614ce9ce9",
    publicPlanPath: "docs/SHIFTQ_MOE_RESEARCH_PLAN.md",
    note:
      "Seed range, token count, persistence/cooldown requirement, and all four gates were merged before this sweep was executed.",
  },
  evidenceBoundary:
    "All traces in this sweep are deterministic synthetic fixtures with known generator boundaries. The result can reject a broken mechanism but cannot validate real semantic shifts, model quality, hardware performance, or ShiftQ-MoE novelty.",
  intervention:
    "After three consecutive JSD scores at or above the fixed threshold, use maximum short-window/recency reweighting for 64 tokens, then require cooldown and rearming. Prefetch and transition retention stay disabled in both arms.",
  configuration: {
    seeds,
    tokens,
    knownShiftToken,
    postShiftWindowTokens,
    base: {
      layers: DEFAULT_CONFIG.layers,
      expertsPerLayer: DEFAULT_CONFIG.expertsPerLayer,
      topK: DEFAULT_CONFIG.topK,
      gpuSlots: DEFAULT_CONFIG.gpuSlots,
      ramSlots: DEFAULT_CONFIG.ramSlots,
      expertSizeMB: DEFAULT_CONFIG.expertSizeMB,
      pcieGBps: DEFAULT_CONFIG.pcieGBps,
      nvmeGBps: DEFAULT_CONFIG.nvmeGBps,
      computeMsPerToken: DEFAULT_CONFIG.computeMsPerToken,
    },
    shiftCacheParameters: SHIFT_CACHE_PARAMETERS,
    fixedControls,
    triggeredControls,
  },
  analysis: {
    pairedUnit: "seed",
    postShiftMetric:
      "Percent change in total modeled link bytes across tokens 512 through 575, triggered versus fixed.",
    stationaryMetric:
      "Percent change in whole-run total modeled link bytes, triggered versus fixed.",
    bootstrapStatistic: "Median paired percent change.",
    bootstrap,
  },
  records,
  summary,
  gates,
  carryForward: Object.values(gates).every((gate) => gate.passed),
};

mkdirSync(evidenceDirectory, { recursive: true });
const serialized = `${JSON.stringify(output, null, 2)}\n`;
writeFileSync(outputUrl, serialized, "utf8");
process.stdout.write(serialized);
