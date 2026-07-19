import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_CONFIG,
  POLICY_IDS,
  SIMULATION_LIMITS,
  analyzeRouterTrace,
  exportRouterTrace,
  fingerprintRouterTrace,
  generateRouterTrace,
  importRouterTrace,
  jensenShannonDivergence,
  runComparison,
  runSimulation,
  validateRouterTrace,
  validateSimulationConfig,
  validateSimulationControls,
  type CapturedRouterTraceSource,
  type ComparisonConfig,
  type RouterTrace,
  type RouterTraceV2,
  type SimulationConfig,
  type SimulationResult,
} from "../lib/simulator";

function comparisonConfig(
  overrides: Partial<ComparisonConfig> = {},
): ComparisonConfig {
  const defaults: ComparisonConfig = {
    scenario: DEFAULT_CONFIG.scenario,
    seed: DEFAULT_CONFIG.seed,
    tokens: DEFAULT_CONFIG.tokens,
    layers: DEFAULT_CONFIG.layers,
    expertsPerLayer: DEFAULT_CONFIG.expertsPerLayer,
    topK: DEFAULT_CONFIG.topK,
    gpuSlots: DEFAULT_CONFIG.gpuSlots,
    ramSlots: DEFAULT_CONFIG.ramSlots,
    expertSizeMB: DEFAULT_CONFIG.expertSizeMB,
    pcieGBps: DEFAULT_CONFIG.pcieGBps,
    nvmeGBps: DEFAULT_CONFIG.nvmeGBps,
    computeMsPerToken: DEFAULT_CONFIG.computeMsPerToken,
  };
  return { ...defaults, ...overrides };
}

function simulationConfig(
  overrides: Partial<SimulationConfig> = {},
): SimulationConfig {
  return { ...DEFAULT_CONFIG, ...overrides };
}

function traceConfig(config: ComparisonConfig) {
  return {
    scenario: config.scenario,
    seed: config.seed,
    tokens: config.tokens,
    layers: config.layers,
    expertsPerLayer: config.expertsPerLayer,
    topK: config.topK,
  } as const;
}

function assertFiniteJson(value: unknown): void {
  const json = JSON.stringify(value);
  assert.ok(json.length > 0);
  assert.doesNotMatch(json, /NaN|Infinity/);
  assert.deepEqual(JSON.parse(json), value);
}

function assertResultInvariants(result: SimulationResult): void {
  const { config, metrics, finalResidency, timeline } = result;
  const expectedAccesses = config.tokens * config.layers * config.topK;
  assert.equal(metrics.totalAccesses, expectedAccesses);
  assert.equal(metrics.gpuHits + metrics.ramHits + metrics.nvmeMisses, expectedAccesses);
  assert.ok(Math.abs(metrics.gpuHitRate + metrics.ramHitRate + metrics.nvmeMissRate - 1) < 1e-7);
  assert.equal(metrics.semanticRoutingChanges, 0);
  assert.equal(timeline.length, config.tokens);
  assert.ok(finalResidency.gpu.length <= config.gpuSlots);
  assert.ok(finalResidency.ram.length <= config.ramSlots);

  const allResident = [
    ...finalResidency.gpu,
    ...finalResidency.ram,
    ...finalResidency.nvme,
  ];
  assert.equal(allResident.length, config.layers * config.expertsPerLayer);
  assert.equal(new Set(allResident).size, allResident.length);
  assert.deepEqual([...finalResidency.gpu].sort(), [...new Set(finalResidency.gpu)].sort());
  assert.ok(metrics.bytesPerToken >= 0);
  assert.ok(metrics.transferStallMsPerToken >= 0);
  assert.ok(metrics.tokensPerSecond > 0);
  assert.ok(metrics.prefetchUsefulness >= 0 && metrics.prefetchUsefulness <= 1);
  assert.equal(
    metrics.totalBytesTransferred,
    metrics.demandBytesTransferred +
      metrics.prefetchBytesTransferred +
      metrics.nvmeBytesRead,
  );
  assert.equal(
    timeline.reduce((sum, point) => sum + point.bytesTransferred, 0),
    metrics.totalBytesTransferred,
  );
  assert.equal(
    metrics.prefetchesIssued,
    metrics.prefetchesUseful + metrics.prefetchesWasted + metrics.prefetchesPending,
  );

  for (const [index, point] of timeline.entries()) {
    assert.equal(point.token, index);
    assert.ok(
      Math.abs(point.gpuHitRate + point.ramHitRate + point.nvmeMissRate - 1) < 1e-7,
    );
    assert.ok(point.gpuResident <= config.gpuSlots);
    assert.ok(point.ramResident <= config.ramSlots);
    assert.ok(point.shiftScore >= 0 && point.shiftScore <= 1);
  }
  assertFiniteJson(result);
}

test("router trace generation and JSON round-trip are deterministic", () => {
  const config = comparisonConfig({ tokens: 48, layers: 4, seed: 17 });
  const first = generateRouterTrace(traceConfig(config));
  const second = generateRouterTrace(traceConfig(config));
  assert.deepEqual(first, second);
  assert.equal(exportRouterTrace(first), exportRouterTrace(second));
  assert.deepEqual(importRouterTrace(exportRouterTrace(first)), first);
  assert.equal(fingerprintRouterTrace(first), fingerprintRouterTrace(second));
  assert.equal(first.version, 2);
  assert.deepEqual(first.source, {
    kind: "synthetic",
    generator: "stratamoe-lab/router-trace-v2",
  });
});

test("different seeds produce different deterministic traces", () => {
  const config = comparisonConfig({ tokens: 64, layers: 3 });
  const first = generateRouterTrace(traceConfig({ ...config, seed: 1 }));
  const second = generateRouterTrace(traceConfig({ ...config, seed: 2 }));
  assert.notEqual(fingerprintRouterTrace(first), fingerprintRouterTrace(second));
  assert.notDeepEqual(first.selections, second.selections);
});

test("trace fingerprints include scalar fields, selections, and provenance", () => {
  const source: CapturedRouterTraceSource = {
    kind: "captured",
    model: {
      id: "allenai/OLMoE-1B-7B-0924-Instruct",
      revision: "a".repeat(40),
    },
    tokenizer: { revision: "b".repeat(40) },
    software: {
      transformersVersion: "4.55.2",
      pytorchVersion: "2.8.0+cu128",
    },
    workload: {
      kind: "dataset",
      datasetId: "openai/gsm8k",
      split: "test",
      exampleIds: ["17"],
    },
    capture: {
      seed: 0,
      device: "cuda:0",
      dtype: "torch.bfloat16",
    },
  };
  const base: RouterTraceV2 = {
    version: 2,
    source,
    scenario: "steady",
    seed: 0,
    tokens: 1,
    layers: 1,
    expertsPerLayer: 2,
    topK: 1,
    selections: [[[0]]],
  };
  assert.notEqual(
    fingerprintRouterTrace(base),
    fingerprintRouterTrace({
      ...base,
      seed: 65_536,
      source: {
        ...source,
        capture: { ...source.capture, seed: 65_536 },
      },
    }),
  );
  assert.notEqual(
    fingerprintRouterTrace(base),
    fingerprintRouterTrace({
      ...base,
      source: {
        ...source,
        software: { ...source.software, pytorchVersion: "2.9.0" },
      },
    }),
  );
  assert.notEqual(
    fingerprintRouterTrace(base),
    fingerprintRouterTrace({ ...base, scenario: "domain-shift" }),
  );
  assert.notEqual(
    fingerprintRouterTrace(base),
    fingerprintRouterTrace({
      ...base,
      source: {
        ...source,
        model: { ...source.model, revision: "c".repeat(40) },
      },
    }),
  );
  assert.notEqual(
    fingerprintRouterTrace(base),
    fingerprintRouterTrace({
      ...base,
      source: {
        ...source,
        workload: {
          kind: "dataset",
          datasetId: "openai/gsm8k",
          split: "test",
          exampleIds: ["18"],
        },
      },
    }),
  );
});

test("v1 imports migrate conservatively to canonical v2 provenance", () => {
  const legacy: RouterTrace = {
    version: 1,
    scenario: "steady",
    seed: 23,
    tokens: 2,
    layers: 1,
    expertsPerLayer: 2,
    topK: 1,
    selections: [[[0]], [[1]]],
  };

  const migrated = importRouterTrace(JSON.stringify(legacy));
  assert.equal(migrated.version, 2);
  assert.deepEqual(migrated.source, {
    kind: "synthetic",
    generator: "stratamoe-lab/legacy-v1-import",
  });
  assert.deepEqual(migrated.selections, legacy.selections);
  assert.equal(JSON.parse(exportRouterTrace(legacy)).version, 2);
  assert.deepEqual(validateRouterTrace(legacy), migrated);
});

test("captured trace provenance supports dataset IDs or a prompt-manifest digest", () => {
  const generated = generateRouterTrace(
    traceConfig(comparisonConfig({ tokens: 2, layers: 1, seed: 31 })),
  );
  const source: CapturedRouterTraceSource = {
    kind: "captured",
    model: {
      id: "allenai/OLMoE-1B-7B-0924-Instruct",
      revision: "1".repeat(40),
    },
    tokenizer: { revision: "2".repeat(40) },
    software: {
      transformersVersion: "4.55.2",
      pytorchVersion: "2.8.0+cu128",
    },
    workload: {
      kind: "dataset",
      datasetId: "Salesforce/wikitext",
      split: "test",
      exampleIds: ["0", "1"],
    },
    capture: {
      seed: generated.seed,
      device: "cuda:0",
      dtype: "torch.bfloat16",
    },
  };
  const captured: RouterTraceV2 = { ...generated, source };

  assert.deepEqual(importRouterTrace(exportRouterTrace(captured)), captured);

  const manifestTrace: RouterTraceV2 = {
    ...captured,
    source: {
      ...source,
      workload: {
        kind: "prompt-manifest",
        sha256: "3".repeat(64),
      },
    },
  };
  assert.deepEqual(validateRouterTrace(manifestTrace), manifestTrace);
  assert.notEqual(
    fingerprintRouterTrace(captured),
    fingerprintRouterTrace(manifestTrace),
  );
});

test("captured trace validation rejects mutable or incomplete provenance", () => {
  const generated = generateRouterTrace(
    traceConfig(comparisonConfig({ tokens: 2, layers: 1, seed: 41 })),
  );
  const source = {
    kind: "captured",
    model: { id: "Qwen/Qwen1.5-MoE-A2.7B-Chat", revision: "a".repeat(40) },
    tokenizer: { revision: "b".repeat(40) },
    software: { transformersVersion: "4.55.2", pytorchVersion: "2.8.0" },
    workload: {
      kind: "prompt-manifest",
      sha256: "c".repeat(64),
    },
    capture: { seed: generated.seed, device: "cpu", dtype: "torch.float32" },
  } as const;
  const captured = { ...generated, source };

  assert.throws(
    () => validateRouterTrace({ ...generated, source: undefined }),
    /source must be an object/,
  );
  assert.throws(
    () => validateRouterTrace({ ...captured, source: { ...source, model: { ...source.model, revision: "main" } } }),
    /immutable lowercase/,
  );
  assert.throws(
    () => validateRouterTrace({ ...captured, source: { ...source, capture: { ...source.capture, seed: 42 } } }),
    /must match trace seed/,
  );
  assert.throws(
    () => validateRouterTrace({ ...captured, source: { ...source, workload: { kind: "prompt-manifest", sha256: "D".repeat(64) } } }),
    /lowercase 64-character/,
  );
  assert.throws(
    () => validateRouterTrace({ ...captured, source: { ...source, undocumented: true } }),
    /unsupported field/,
  );
  assert.throws(
    () => validateRouterTrace({ ...captured, source: { ...source, software: { ...source.software, pytorchVersion: "" } } }),
    /non-empty string/,
  );
  assert.throws(
    () => validateRouterTrace({
      ...captured,
      source: {
        ...source,
        workload: {
          kind: "dataset",
          datasetId: "openai/gsm8k",
          split: "test",
          exampleIds: [],
        },
      },
    }),
    /exampleIds must be a non-empty ordered array/,
  );
  assert.throws(
    () => validateRouterTrace({ ...generated, version: 3 }),
    /version must be 1 or 2/,
  );
});

test("config validation rejects unsafe or nonsensical settings", () => {
  assert.deepEqual(
    validateSimulationControls({
      shiftCachePrefetch: false,
      shiftCacheJsdReweighting: true,
      shiftCacheTransitionRetention: false,
    }),
    {
      shiftCachePrefetch: false,
      shiftCacheJsdReweighting: true,
      shiftCacheTransitionRetention: false,
    },
  );
  assert.throws(
    () =>
      validateSimulationControls({
        shiftCachePrefetch: "no",
        shiftCacheJsdReweighting: true,
        shiftCacheTransitionRetention: true,
      } as never),
    /shiftCachePrefetch must be a boolean/,
  );
  assert.throws(
    () =>
      validateSimulationControls({
        shiftCachePrefetch: true,
        shiftCacheJsdReweighting: true,
        shiftCacheTransitionRetention: true,
        unsupported: true,
      } as never),
    /unsupported field.*unsupported/,
  );
  assert.throws(
    () => validateSimulationConfig(simulationConfig({ topK: 17 })),
    /topK cannot exceed expertsPerLayer/,
  );
  assert.throws(
    () => validateSimulationConfig(simulationConfig({ gpuSlots: 0 })),
    /gpuSlots/,
  );
  assert.throws(
    () => validateSimulationConfig(simulationConfig({ ramSlots: -1 })),
    /ramSlots/,
  );
  assert.throws(
    () => validateSimulationConfig(simulationConfig({ pcieGBps: Number.NaN })),
    /pcieGBps/,
  );
  assert.throws(
    () =>
      validateSimulationConfig(
        simulationConfig({ scenario: "unknown" as SimulationConfig["scenario"] }),
      ),
    /scenario must be one of/,
  );
  assert.throws(
    () => generateRouterTrace({ scenario: "steady" } as never),
    /seed/,
  );
  assert.throws(
    () =>
      validateSimulationConfig(
        simulationConfig({ tokens: SIMULATION_LIMITS.maxTokens + 1 }),
      ),
    /tokens must be between/,
  );
  assert.throws(
    () =>
      validateSimulationConfig(
        simulationConfig({ tokens: 4096, layers: 64, topK: 16 }),
      ),
    /tokens \* layers \* topK/,
  );
  assert.throws(
    () => validateSimulationConfig(simulationConfig({ expertSizeMB: Number.MAX_VALUE })),
    /expertSizeMB must be between/,
  );
  assert.throws(
    () => validateSimulationConfig(simulationConfig({ nvmeGBps: Number.MIN_VALUE })),
    /nvmeGBps must be between/,
  );
  assert.throws(
    () =>
      validateSimulationConfig(
        simulationConfig({ gpuSlots: SIMULATION_LIMITS.maxGpuSlots + 1 }),
      ),
    /gpuSlots must be between/,
  );
});

test("trace validation rejects malformed dimensions, duplicates, and config mismatches", () => {
  const config = comparisonConfig({ tokens: 12, layers: 2, expertsPerLayer: 8 });
  const trace = generateRouterTrace(traceConfig(config));
  const malformed = structuredClone(trace) as RouterTrace;
  malformed.selections[0][0] = [0, 0];
  assert.throws(() => validateRouterTrace(malformed), /more than once/);

  const missingToken = structuredClone(trace) as RouterTrace;
  missingToken.selections.pop();
  assert.throws(() => validateRouterTrace(missingToken), /one entry per token/);
  assert.throws(() => importRouterTrace("{not-json"), /Invalid router trace JSON/);
  assert.throws(
    () => validateRouterTrace(trace, { seed: trace.seed + 1 }),
    /does not match config/,
  );
  assert.throws(
    () =>
      validateRouterTrace({
        ...trace,
        tokens: SIMULATION_LIMITS.maxTokens + 1,
        selections: [],
      }),
    /tokens must be between/,
  );
});

test("MB and GB/s use decimal units on every transfer link", () => {
  const result = runSimulation(
    simulationConfig({
      scenario: "steady",
      policy: "lru",
      tokens: 1,
      layers: 1,
      expertsPerLayer: 2,
      topK: 1,
      gpuSlots: 1,
      ramSlots: 0,
      expertSizeMB: 1,
      pcieGBps: 1,
      nvmeGBps: 1,
      computeMsPerToken: 0,
    }),
  );
  assert.equal(result.metrics.demandBytesTransferred, 1_000_000);
  assert.equal(result.metrics.nvmeBytesRead, 1_000_000);
  assert.equal(result.metrics.totalBytesTransferred, 2_000_000);
  assert.equal(result.metrics.transferStallMsPerToken, 2);
  assert.equal(result.metrics.tokensPerSecond, 500);
});

test("Jensen-Shannon divergence is symmetric and bounded", () => {
  const left = new Map([
    ["a", 4],
    ["b", 2],
  ]);
  const same = new Map(left);
  const disjoint = new Map([["z", 6]]);
  assert.equal(jensenShannonDivergence(left, same), 0);
  assert.equal(jensenShannonDivergence(left, disjoint), 1);
  assert.equal(
    jensenShannonDivergence(left, disjoint),
    jensenShannonDivergence(disjoint, left),
  );
});

test("each policy obeys hierarchy capacity and accounting invariants", () => {
  const config = comparisonConfig({ tokens: 72, layers: 5, gpuSlots: 11, ramSlots: 13 });
  const results = runComparison(config);
  assert.deepEqual(
    results.map((result) => result.policy),
    POLICY_IDS,
  );
  for (const result of results) assertResultInvariants(result);
});

test("zero RAM capacity remains valid and sends GPU evictions to NVMe", () => {
  const result = runSimulation(
    simulationConfig({
      scenario: "high-churn",
      policy: "lru",
      tokens: 40,
      layers: 3,
      gpuSlots: 4,
      ramSlots: 0,
    }),
  );
  assertResultInvariants(result);
  assert.equal(result.finalResidency.ram.length, 0);
  assert.equal(result.metrics.ramHits, 0);
  assert.ok(result.metrics.nvmeMisses > 0);
});

test("simulation output is exactly reproducible", () => {
  const config = simulationConfig({ tokens: 80, layers: 4, seed: 2026 });
  assert.deepEqual(runSimulation(config), runSimulation(config));
});

test("comparison policies consume the exact same router trace", () => {
  const results = runComparison(comparisonConfig({ tokens: 80, layers: 4 }));
  const [first, ...rest] = results;
  for (const result of rest) {
    assert.equal(result.traceFingerprint, first.traceFingerprint);
    assert.deepEqual(result.trace.selections, first.trace.selections);
    assert.equal(result.metrics.semanticRoutingChanges, 0);
  }
});

test("a supplied router trace is preserved verbatim", () => {
  const config = simulationConfig({
    scenario: "steady",
    policy: "shift-cache",
    tokens: 8,
    layers: 1,
    expertsPerLayer: 4,
    topK: 2,
    gpuSlots: 2,
    ramSlots: 1,
  });
  const trace: RouterTraceV2 = {
    version: 2,
    source: {
      kind: "synthetic",
      generator: "tests/hand-authored",
    },
    scenario: "steady",
    seed: config.seed,
    tokens: 8,
    layers: 1,
    expertsPerLayer: 4,
    topK: 2,
    selections: [
      [[0, 1]],
      [[1, 2]],
      [[2, 3]],
      [[3, 0]],
      [[0, 1]],
      [[1, 2]],
      [[2, 3]],
      [[3, 0]],
    ],
  };
  const result = runSimulation(config, trace);
  assert.deepEqual(result.trace, trace);
  assert.equal(result.metrics.semanticRoutingChanges, 0);
});

test("domain-shift benchmark detects the shift and improves over both baselines", () => {
  const [lru, lfu, shiftCache] = runComparison(comparisonConfig());
  assert.ok(shiftCache.metrics.detectedShifts >= 1);
  assert.ok(shiftCache.metrics.bytesPerToken < lru.metrics.bytesPerToken);
  assert.ok(shiftCache.metrics.bytesPerToken < lfu.metrics.bytesPerToken);
  assert.ok(
    shiftCache.metrics.transferStallMsPerToken < lru.metrics.transferStallMsPerToken,
  );
  assert.ok(
    shiftCache.metrics.transferStallMsPerToken < lfu.metrics.transferStallMsPerToken,
  );
  assert.ok(shiftCache.metrics.tokensPerSecond > lru.metrics.tokensPerSecond);
  assert.ok(shiftCache.metrics.prefetchesIssued > 0);
  assert.ok(shiftCache.metrics.prefetchesUseful > 0);
  assert.equal(shiftCache.metrics.semanticRoutingChanges, 0);
});

test("prefetch transfer cost is serialized and no prefetch is issued after the final token", () => {
  const result = runSimulation(
    simulationConfig({ policy: "shift-cache", tokens: 96, layers: 4, gpuSlots: 16 }),
  );
  const pcieStall =
    ((result.metrics.demandBytesTransferred + result.metrics.prefetchBytesTransferred) /
      (result.config.pcieGBps * 1_000_000_000)) *
    1000;
  const nvmeStall =
    (result.metrics.nvmeBytesRead / (result.config.nvmeGBps * 1_000_000_000)) * 1000;
  assert.ok(
    Math.abs(result.metrics.totalTransferStallMs - (pcieStall + nvmeStall)) < 1e-6,
  );
  assert.equal(result.timeline.at(-1)?.prefetchesIssued, 0);
});

test("immediate reuse rate excludes the first token from its denominator", () => {
  const trace: RouterTrace = {
    version: 1,
    scenario: "steady",
    seed: 0,
    tokens: 3,
    layers: 1,
    expertsPerLayer: 2,
    topK: 1,
    selections: [[[0]], [[0]], [[1]]],
  };
  assert.equal(analyzeRouterTrace(trace, 1).immediateReuseRate, 0.5);
});

test("segment oracle distinguishes locality from high churn", () => {
  const base = comparisonConfig({ tokens: 96, layers: 4, gpuSlots: 12 });
  const steady = generateRouterTrace(traceConfig({ ...base, scenario: "steady" }));
  const churn = generateRouterTrace(traceConfig({ ...base, scenario: "high-churn" }));
  const steadyDiagnostic = analyzeRouterTrace(steady, base.gpuSlots, 16);
  const churnDiagnostic = analyzeRouterTrace(churn, base.gpuSlots, 16);
  assert.ok(steadyDiagnostic.segmentOracleHitRate > churnDiagnostic.segmentOracleHitRate);
  assert.ok(steadyDiagnostic.uniqueExperts < churnDiagnostic.uniqueExperts);
});
