import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  DEFAULT_SIMULATION_CONTROLS,
  POLICY_IDS,
  fingerprintRouterTrace,
  importRouterTrace,
  runSimulation,
  type ComparisonConfig,
  type SimulationMetrics,
} from "../lib/simulator";

const traceUrl = new URL(
  "../evidence/switch-base-8/router-trace.v2.json",
  import.meta.url,
);
const manifestUrl = new URL(
  "../evidence/switch-base-8/prompt-manifest.json",
  import.meta.url,
);
const metadataUrl = new URL(
  "../evidence/switch-base-8/capture-metadata.json",
  import.meta.url,
);
const comparisonUrl = new URL(
  "../evidence/switch-base-8/comparison.json",
  import.meta.url,
);
const prefetchAblationUrl = new URL(
  "../evidence/switch-base-8/prefetch-ablation.json",
  import.meta.url,
);
const retentionAblationUrl = new URL(
  "../evidence/switch-base-8/retention-ablation.json",
  import.meta.url,
);

test("checked-in Switch trace has pinned captured provenance", () => {
  const trace = importRouterTrace(readFileSync(traceUrl, "utf8"));
  assert.equal(trace.version, 2);
  assert.equal(trace.source.kind, "captured");
  assert.equal(trace.tokens, 215);
  assert.equal(trace.layers, 6);
  assert.equal(trace.expertsPerLayer, 8);
  assert.equal(trace.topK, 1);

  if (trace.source.kind !== "captured") {
    assert.fail("Expected captured source.");
  }
  assert.equal(trace.source.model.id, "google/switch-base-8");
  assert.equal(
    trace.source.model.revision,
    "92fe2d22b024d9937146fe097ba3d3a7ba146e1b",
  );
  assert.equal(trace.source.workload.kind, "prompt-manifest");
  if (trace.source.workload.kind !== "prompt-manifest") {
    assert.fail("Expected prompt-manifest workload.");
  }
  const manifestSha256 = createHash("sha256")
    .update(readFileSync(manifestUrl))
    .digest("hex");
  assert.equal(trace.source.workload.sha256, manifestSha256);

  for (let layer = 0; layer < trace.layers; layer += 1) {
    const observed = new Set(trace.selections.map((token) => token[layer][0]));
    assert.deepEqual([...observed].sort((a, b) => a - b), [0, 1, 2, 3, 4, 5, 6, 7]);
  }
  assert.equal(fingerprintRouterTrace(trace), "f3b18fe2");

  const metadata = JSON.parse(readFileSync(metadataUrl, "utf8")) as {
    trace_sha256: string;
    selections_sha256: string;
  };
  const traceSha256 = createHash("sha256")
    .update(readFileSync(traceUrl))
    .digest("hex");
  assert.equal(metadata.trace_sha256, traceSha256);
  assert.equal(
    metadata.selections_sha256,
    "4510d2bbb7dfeb14cdf38786a581fe5d94a373648072876a76d359b9e691aa8b",
  );
});

test("captured comparison preserves the negative result", () => {
  const comparison = JSON.parse(readFileSync(comparisonUrl, "utf8")) as {
    evidenceBoundary: string;
    traceFingerprint: string;
    winnerByModeledBytes: string;
    results: Array<{
      policy: string;
      metrics: { bytesPerToken: number; semanticRoutingChanges: number };
    }>;
  };

  assert.match(comparison.evidenceBoundary, /remain simulated/i);
  assert.equal(comparison.traceFingerprint, "f3b18fe2");
  assert.equal(comparison.winnerByModeledBytes, "lfu");
  const metrics = Object.fromEntries(
    comparison.results.map((result) => [result.policy, result.metrics]),
  );
  assert.ok(metrics["shift-cache"].bytesPerToken > metrics.lfu.bytesPerToken);
  for (const result of comparison.results) {
    assert.equal(result.metrics.semanticRoutingChanges, 0);
  }
});

test("every policy replays the captured selections verbatim", () => {
  const trace = importRouterTrace(readFileSync(traceUrl, "utf8"));
  const config = {
    scenario: trace.scenario,
    seed: trace.seed,
    tokens: trace.tokens,
    layers: trace.layers,
    expertsPerLayer: trace.expertsPerLayer,
    topK: trace.topK,
    gpuSlots: 12,
    ramSlots: 18,
    expertSizeMB: 64,
    pcieGBps: 24,
    nvmeGBps: 7,
    computeMsPerToken: 6,
  } satisfies ComparisonConfig;

  const results = POLICY_IDS.map((policy) =>
    runSimulation({ ...config, policy }, trace),
  );
  for (const result of results) {
    assert.deepEqual(result.trace.selections, trace.selections);
    assert.equal(result.metrics.semanticRoutingChanges, 0);
  }
});

test("prefetch-disabled ShiftCache isolates speculative transfers", () => {
  const trace = importRouterTrace(readFileSync(traceUrl, "utf8"));
  const config = {
    scenario: trace.scenario,
    seed: trace.seed,
    tokens: trace.tokens,
    layers: trace.layers,
    expertsPerLayer: trace.expertsPerLayer,
    topK: trace.topK,
    gpuSlots: 12,
    ramSlots: 18,
    expertSizeMB: 64,
    pcieGBps: 24,
    nvmeGBps: 7,
    computeMsPerToken: 6,
    policy: "shift-cache",
  } as const;

  const prefetchOn = runSimulation(config, trace);
  const prefetchOff = runSimulation(config, trace, {
    ...DEFAULT_SIMULATION_CONTROLS,
    shiftCachePrefetch: false,
  });

  assert.equal(prefetchOn.controls.shiftCachePrefetch, true);
  assert.equal(prefetchOff.controls.shiftCachePrefetch, false);
  assert.ok(prefetchOn.metrics.prefetchesIssued > 0);
  assert.equal(prefetchOff.metrics.prefetchesIssued, 0);
  assert.equal(prefetchOff.metrics.prefetchBytesTransferred, 0);
  assert.equal(
    prefetchOff.metrics.detectedShifts,
    prefetchOn.metrics.detectedShifts,
  );
  assert.deepEqual(prefetchOff.trace.selections, prefetchOn.trace.selections);
  assert.equal(prefetchOff.metrics.semanticRoutingChanges, 0);
});

test("checked-in prefetch ablation preserves the captured result", () => {
  const ablation = JSON.parse(readFileSync(prefetchAblationUrl, "utf8")) as {
    traceSha256: string;
    traceFingerprint: string;
    winnerByModeledBytes: string;
    config: ComparisonConfig;
    variants: Array<{
      id: string;
      controls: {
        shiftCachePrefetch: boolean;
        shiftCacheJsdReweighting: boolean;
        shiftCacheTransitionRetention: boolean;
      };
      metrics: SimulationMetrics;
    }>;
    effectOfDisablingPrefetch: {
      bytesPerToken: { percentChange: number };
    };
  };

  assert.equal(
    ablation.traceSha256,
    createHash("sha256").update(readFileSync(traceUrl)).digest("hex"),
  );
  assert.equal(ablation.traceFingerprint, "f3b18fe2");
  assert.equal(ablation.winnerByModeledBytes, "prefetch-off");

  const variants = Object.fromEntries(
    ablation.variants.map((variant) => [variant.id, variant]),
  );
  const trace = importRouterTrace(readFileSync(traceUrl, "utf8"));
  const replayConfig = {
    ...ablation.config,
    policy: "shift-cache",
  } as const;
  assert.deepEqual(
    variants["prefetch-on"].metrics,
    runSimulation(replayConfig, trace).metrics,
  );
  assert.deepEqual(
    variants["prefetch-off"].metrics,
    runSimulation(replayConfig, trace, {
      ...DEFAULT_SIMULATION_CONTROLS,
      shiftCachePrefetch: false,
    }).metrics,
  );
  assert.equal(variants["prefetch-on"].controls.shiftCachePrefetch, true);
  assert.equal(variants["prefetch-off"].controls.shiftCachePrefetch, false);
  assert.equal(variants["prefetch-on"].metrics.bytesPerToken, 516762790.7);
  assert.equal(variants["prefetch-off"].metrics.bytesPerToken, 411088372.09);
  assert.equal(variants["prefetch-off"].metrics.prefetchesIssued, 0);
  assert.equal(
    variants["prefetch-off"].metrics.detectedShifts,
    variants["prefetch-on"].metrics.detectedShifts,
  );
  assert.equal(
    variants["prefetch-off"].metrics.semanticRoutingChanges,
    0,
  );
  assert.equal(ablation.effectOfDisablingPrefetch.bytesPerToken.percentChange, -20.44930876);
});

test("checked-in retention ablation isolates the two scoring factors", () => {
  const ablation = JSON.parse(readFileSync(retentionAblationUrl, "utf8")) as {
    evidenceBoundary: string;
    terminology: string;
    traceSha256: string;
    traceFingerprint: string;
    config: ComparisonConfig;
    semanticGroupBoundaries: Array<{ token: number; group: string }>;
    variants: Array<{
      id: string;
      controls: {
        shiftCachePrefetch: boolean;
        shiftCacheJsdReweighting: boolean;
        shiftCacheTransitionRetention: boolean;
      };
      detectedShiftEvents: Array<{ token: number; scoreBits: number }>;
      metrics: SimulationMetrics;
      postBoundary: {
        aggregate: { modeledLinkBytesPerToken: number };
      };
      versusFixedFrequencyRecency: {
        wholeRunModeledLinkBytesPercentChange: number;
        postBoundaryModeledLinkBytesPercentChange: number;
      };
      versusLfu: {
        wholeRunModeledLinkBytesPercentChange: number;
        postBoundaryModeledLinkBytesPercentChange: number;
      };
    }>;
    lfuReference: { metrics: SimulationMetrics };
    bestVariantByWholeRunModeledLinkBytes: string;
    bestVariantByPostBoundaryModeledLinkBytes: string;
  };

  assert.match(ablation.evidenceBoundary, /remain simulated/i);
  assert.match(ablation.terminology, /continuously changes eviction weights/i);
  assert.equal(
    ablation.traceSha256,
    createHash("sha256").update(readFileSync(traceUrl)).digest("hex"),
  );
  assert.equal(ablation.traceFingerprint, "f3b18fe2");
  assert.deepEqual(ablation.semanticGroupBoundaries, [
    { token: 64, group: "electronics" },
    { token: 127, group: "science" },
    { token: 167, group: "software" },
  ]);

  const trace = importRouterTrace(readFileSync(traceUrl, "utf8"));
  const replayConfig = { ...ablation.config, policy: "shift-cache" } as const;
  const variants = Object.fromEntries(
    ablation.variants.map((variant) => [variant.id, variant]),
  );
  assert.deepEqual(Object.keys(variants), [
    "fixed-frequency-recency",
    "jsd-only",
    "transition-only",
    "jsd-and-transition",
  ]);

  for (const variant of ablation.variants) {
    const replay = runSimulation(replayConfig, trace, variant.controls);
    assert.deepEqual(variant.metrics, replay.metrics);
    assert.deepEqual(
      variant.detectedShiftEvents.map((event) => event.token),
      [23, 59, 140],
    );
    assert.equal(variant.controls.shiftCachePrefetch, false);
    assert.equal(variant.metrics.prefetchesIssued, 0);
    assert.equal(variant.metrics.prefetchBytesTransferred, 0);
    assert.equal(variant.metrics.semanticRoutingChanges, 0);
    assert.deepEqual(replay.trace.selections, trace.selections);
  }

  assert.equal(variants["fixed-frequency-recency"].metrics.bytesPerToken, 400372093.02);
  assert.equal(variants["jsd-only"].metrics.bytesPerToken, 417637209.3);
  assert.equal(variants["transition-only"].metrics.bytesPerToken, 398288372.09);
  assert.equal(variants["jsd-and-transition"].metrics.bytesPerToken, 411088372.09);
  assert.equal(
    variants["jsd-only"].versusFixedFrequencyRecency
      .wholeRunModeledLinkBytesPercentChange,
    4.31226766,
  );
  assert.equal(
    variants["transition-only"].versusFixedFrequencyRecency
      .postBoundaryModeledLinkBytesPercentChange,
    -1.33111481,
  );
  assert.equal(ablation.lfuReference.metrics.bytesPerToken, 357209302.33);
  assert.ok(
    variants["transition-only"].metrics.bytesPerToken >
      ablation.lfuReference.metrics.bytesPerToken,
  );
  assert.equal(ablation.bestVariantByWholeRunModeledLinkBytes, "transition-only");
  assert.equal(ablation.bestVariantByPostBoundaryModeledLinkBytes, "transition-only");
});
