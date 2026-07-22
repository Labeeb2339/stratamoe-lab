import {
  runComparison,
  type ComparisonConfig,
  type PolicyId,
} from "../lib/simulator";

export const FIXED_BENCHMARK_CONFIG = Object.freeze({
  scenario: "domain-shift",
  seed: 2339,
  tokens: 240,
  layers: 8,
  expertsPerLayer: 16,
  topK: 2,
  gpuSlots: 32,
  ramSlots: 64,
  expertSizeMB: 64,
  pcieGBps: 24,
  nvmeGBps: 7,
  computeMsPerToken: 6,
}) satisfies ComparisonConfig;

export function buildFixedBenchmark() {
  const results = runComparison(FIXED_BENCHMARK_CONFIG);
  const byBytes = [...results].sort(
    (left, right) => left.metrics.bytesPerToken - right.metrics.bytesPerToken,
  );
  const byStall = [...results].sort(
    (left, right) =>
      left.metrics.transferStallMsPerToken -
      right.metrics.transferStallMsPerToken,
  );

  const output: {
    benchmark: string;
    config: ComparisonConfig;
    traceFingerprint: string;
    traceDiagnostics: (typeof results)[number]["traceDiagnostics"];
    winnerByBytes: PolicyId;
    winnerByTransferStall: PolicyId;
    results: Array<Pick<(typeof results)[number], "policy" | "metrics">>;
  } = {
    benchmark: "stratamoe-domain-shift-v1",
    config: FIXED_BENCHMARK_CONFIG,
    traceFingerprint: results[0].traceFingerprint,
    traceDiagnostics: results[0].traceDiagnostics,
    winnerByBytes: byBytes[0].policy,
    winnerByTransferStall: byStall[0].policy,
    results: results.map(({ policy, metrics }) => ({ policy, metrics })),
  };

  return output;
}
