import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
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
const outputUrl = new URL(
  "../evidence/switch-base-8/comparison.json",
  import.meta.url,
);
const serializedTrace = readFileSync(traceUrl, "utf8");
const trace = importRouterTrace(serializedTrace);

if (trace.source.kind !== "captured") {
  throw new TypeError("Captured benchmark requires source.kind=captured.");
}

export const CAPTURED_BENCHMARK_CONFIG = Object.freeze({
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
}) satisfies ComparisonConfig;

const results = POLICY_IDS.map((policy) =>
  runSimulation({ ...CAPTURED_BENCHMARK_CONFIG, policy }, trace),
);
const byBytes = [...results].sort(
  (left, right) => left.metrics.bytesPerToken - right.metrics.bytesPerToken,
);

const output = {
  benchmark: "switch-base-8-captured-router-v1",
  evidenceBoundary:
    "Router selections came from a pinned model execution; memory traffic, stalls, and throughput remain simulated.",
  traceSha256: createHash("sha256").update(serializedTrace).digest("hex"),
  traceFingerprint: fingerprintRouterTrace(trace),
  source: trace.source,
  config: CAPTURED_BENCHMARK_CONFIG,
  traceDiagnostics: results[0].traceDiagnostics,
  winnerByModeledBytes: byBytes[0].policy,
  results: results.map(({ policy, metrics }) => ({ policy, metrics })),
};

const serializedOutput = `${JSON.stringify(output, null, 2)}\n`;
writeFileSync(outputUrl, serializedOutput, "utf8");
process.stdout.write(serializedOutput);
