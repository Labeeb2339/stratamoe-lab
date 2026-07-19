import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  POLICY_IDS,
  fingerprintRouterTrace,
  importRouterTrace,
  runSimulation,
  type ComparisonConfig,
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
