import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
import {
  DEFAULT_SIMULATION_CONTROLS,
  SHIFT_CACHE_PARAMETERS,
  fingerprintRouterTrace,
  runSimulation,
} from "../lib/simulator";
import {
  CAPTURED_BENCHMARK_CONFIG,
  CAPTURED_TRACE,
  CAPTURED_TRACE_SERIALIZED,
} from "./captured-switch-fixture";

const outputUrl = new URL(
  "../evidence/switch-base-8/prefetch-ablation.json",
  import.meta.url,
);
const config = { ...CAPTURED_BENCHMARK_CONFIG, policy: "shift-cache" } as const;
const prefetchOn = runSimulation(
  config,
  CAPTURED_TRACE,
  DEFAULT_SIMULATION_CONTROLS,
);
const prefetchOff = runSimulation(config, CAPTURED_TRACE, {
  shiftCachePrefetch: false,
});

function difference(after: number, before: number, digits = 8): {
  absoluteChange: number;
  percentChange: number;
} {
  const absoluteChange = after - before;
  return {
    absoluteChange: Number(absoluteChange.toFixed(digits)),
    percentChange: Number(((absoluteChange / before) * 100).toFixed(8)),
  };
}

const output = {
  benchmark: "switch-base-8-shiftcache-prefetch-ablation-v1",
  question:
    "What changes when transition prefetching is disabled while ShiftCache's detector and retention scoring stay unchanged?",
  evidenceBoundary:
    "Router selections came from a pinned model execution; memory traffic, stalls, and throughput remain simulated.",
  traceSha256: createHash("sha256")
    .update(CAPTURED_TRACE_SERIALIZED)
    .digest("hex"),
  traceFingerprint: fingerprintRouterTrace(CAPTURED_TRACE),
  source: CAPTURED_TRACE.source,
  config: CAPTURED_BENCHMARK_CONFIG,
  shiftCacheParameters: SHIFT_CACHE_PARAMETERS,
  traceDiagnostics: prefetchOn.traceDiagnostics,
  variants: [
    {
      id: "prefetch-on",
      controls: prefetchOn.controls,
      metrics: prefetchOn.metrics,
    },
    {
      id: "prefetch-off",
      controls: prefetchOff.controls,
      metrics: prefetchOff.metrics,
    },
  ],
  effectOfDisablingPrefetch: {
    bytesPerToken: difference(
      prefetchOff.metrics.bytesPerToken,
      prefetchOn.metrics.bytesPerToken,
      2,
    ),
    transferStallMsPerToken: difference(
      prefetchOff.metrics.transferStallMsPerToken,
      prefetchOn.metrics.transferStallMsPerToken,
    ),
    gpuHitRate: difference(
      prefetchOff.metrics.gpuHitRate,
      prefetchOn.metrics.gpuHitRate,
    ),
    nvmeMissRate: difference(
      prefetchOff.metrics.nvmeMissRate,
      prefetchOn.metrics.nvmeMissRate,
    ),
  },
  winnerByModeledBytes:
    prefetchOff.metrics.bytesPerToken < prefetchOn.metrics.bytesPerToken
      ? "prefetch-off"
      : "prefetch-on",
};

const serializedOutput = `${JSON.stringify(output, null, 2)}\n`;
writeFileSync(outputUrl, serializedOutput, "utf8");
process.stdout.write(serializedOutput);
