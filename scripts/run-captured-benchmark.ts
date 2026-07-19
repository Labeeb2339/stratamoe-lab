import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import {
  POLICY_IDS,
  fingerprintRouterTrace,
  runSimulation,
} from "../lib/simulator";
import {
  CAPTURED_BENCHMARK_CONFIG,
  CAPTURED_TRACE,
  CAPTURED_TRACE_SERIALIZED,
} from "./captured-switch-fixture";

const outputUrl = new URL(
  "../evidence/switch-base-8/comparison.json",
  import.meta.url,
);

const results = POLICY_IDS.map((policy) =>
  runSimulation({ ...CAPTURED_BENCHMARK_CONFIG, policy }, CAPTURED_TRACE),
);
const byBytes = [...results].sort(
  (left, right) => left.metrics.bytesPerToken - right.metrics.bytesPerToken,
);

const output = {
  benchmark: "switch-base-8-captured-router-v1",
  evidenceBoundary:
    "Router selections came from a pinned model execution; memory traffic, stalls, and throughput remain simulated.",
  traceSha256: createHash("sha256")
    .update(CAPTURED_TRACE_SERIALIZED)
    .digest("hex"),
  traceFingerprint: fingerprintRouterTrace(CAPTURED_TRACE),
  source: CAPTURED_TRACE.source,
  config: CAPTURED_BENCHMARK_CONFIG,
  traceDiagnostics: results[0].traceDiagnostics,
  winnerByModeledBytes: byBytes[0].policy,
  results: results.map(({ policy, metrics }) => ({ policy, metrics })),
};

const serializedOutput = `${JSON.stringify(output, null, 2)}\n`;
writeFileSync(outputUrl, serializedOutput, "utf8");
process.stdout.write(serializedOutput);
