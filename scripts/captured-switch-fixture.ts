import { readFileSync } from "node:fs";
import {
  importRouterTrace,
  type ComparisonConfig,
} from "../lib/simulator";

export const CAPTURED_TRACE_URL = new URL(
  "../evidence/switch-base-8/router-trace.v2.json",
  import.meta.url,
);

export const CAPTURED_TRACE_SERIALIZED = readFileSync(CAPTURED_TRACE_URL, "utf8");
export const CAPTURED_TRACE = importRouterTrace(CAPTURED_TRACE_SERIALIZED);

if (CAPTURED_TRACE.source.kind !== "captured") {
  throw new TypeError("Captured benchmark requires source.kind=captured.");
}

export const CAPTURED_BENCHMARK_CONFIG = Object.freeze({
  scenario: CAPTURED_TRACE.scenario,
  seed: CAPTURED_TRACE.seed,
  tokens: CAPTURED_TRACE.tokens,
  layers: CAPTURED_TRACE.layers,
  expertsPerLayer: CAPTURED_TRACE.expertsPerLayer,
  topK: CAPTURED_TRACE.topK,
  gpuSlots: 12,
  ramSlots: 18,
  expertSizeMB: 64,
  pcieGBps: 24,
  nvmeGBps: 7,
  computeMsPerToken: 6,
}) satisfies ComparisonConfig;
