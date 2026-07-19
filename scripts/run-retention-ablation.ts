import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import {
  DEFAULT_SIMULATION_CONTROLS,
  SHIFT_CACHE_PARAMETERS,
  fingerprintRouterTrace,
  runSimulation,
  type SimulationControls,
  type SimulationResult,
} from "../lib/simulator";
import {
  CAPTURED_BENCHMARK_CONFIG,
  CAPTURED_TRACE,
  CAPTURED_TRACE_SERIALIZED,
} from "./captured-switch-fixture";

const outputUrl = new URL(
  "../evidence/switch-base-8/retention-ablation.json",
  import.meta.url,
);
const metadataUrl = new URL(
  "../evidence/switch-base-8/capture-metadata.json",
  import.meta.url,
);

interface PromptSpan {
  id: string;
  tokenStart: number;
  tokenEndExclusive: number;
}

interface CaptureMetadata {
  prompt_spans: Array<{
    id: string;
    token_start: number;
    token_end_exclusive: number;
  }>;
}

interface RetentionVariant {
  id: string;
  description: string;
  controls: SimulationControls;
}

const baseControls = {
  ...DEFAULT_SIMULATION_CONTROLS,
  shiftCachePrefetch: false,
};

const variants: RetentionVariant[] = [
  {
    id: "fixed-frequency-recency",
    description:
      "Fixed long-window, short-window, and recency weights; no JSD reweighting or transition retention.",
    controls: {
      ...baseControls,
      shiftCacheJsdReweighting: false,
      shiftCacheTransitionRetention: false,
    },
  },
  {
    id: "jsd-only",
    description:
      "JSD continuously reweights long-window, short-window, and recency terms; no transition retention.",
    controls: {
      ...baseControls,
      shiftCacheJsdReweighting: true,
      shiftCacheTransitionRetention: false,
    },
  },
  {
    id: "transition-only",
    description:
      "Fixed frequency/recency weights plus transition retention; no JSD reweighting.",
    controls: {
      ...baseControls,
      shiftCacheJsdReweighting: false,
      shiftCacheTransitionRetention: true,
    },
  },
  {
    id: "jsd-and-transition",
    description:
      "Current combined retention score with JSD reweighting and transition retention.",
    controls: baseControls,
  },
];

const metadata = JSON.parse(
  readFileSync(metadataUrl, "utf8"),
) as CaptureMetadata;
const promptSpans: PromptSpan[] = metadata.prompt_spans.map((span) => ({
  id: span.id,
  tokenStart: span.token_start,
  tokenEndExclusive: span.token_end_exclusive,
}));

function groupOf(id: string): string {
  return id.split("-", 1)[0];
}

const semanticGroupBoundaries = promptSpans
  .filter(
    (span, index) =>
      index > 0 && groupOf(span.id) !== groupOf(promptSpans[index - 1].id),
  )
  .map((span) => ({ token: span.tokenStart, group: groupOf(span.id) }));

function round(value: number, digits = 8): number {
  return Number(value.toFixed(digits));
}

function percentChange(after: number, before: number): number {
  return before === 0 ? 0 : round(((after - before) / before) * 100);
}

function postBoundarySummary(result: SimulationResult) {
  const windowTokens = 32;
  const windows = semanticGroupBoundaries.map(({ token, group }) => {
    const endExclusive = Math.min(result.timeline.length, token + windowTokens);
    const points = result.timeline.slice(token, endExclusive);
    const totalModeledLinkBytes = points.reduce(
      (sum, point) => sum + point.bytesTransferred,
      0,
    );
    return {
      boundaryToken: token,
      group,
      endExclusive,
      observedTokens: points.length,
      totalModeledLinkBytes: round(totalModeledLinkBytes, 2),
      modeledLinkBytesPerToken: round(
        points.length === 0 ? 0 : totalModeledLinkBytes / points.length,
        2,
      ),
    };
  });
  const totalTokens = windows.reduce(
    (sum, window) => sum + window.observedTokens,
    0,
  );
  const totalModeledLinkBytes = windows.reduce(
    (sum, window) => sum + window.totalModeledLinkBytes,
    0,
  );
  return {
    windowTokens,
    windows,
    aggregate: {
      observedTokens: totalTokens,
      totalModeledLinkBytes: round(totalModeledLinkBytes, 2),
      modeledLinkBytesPerToken: round(
        totalTokens === 0 ? 0 : totalModeledLinkBytes / totalTokens,
        2,
      ),
    },
  };
}

const config = { ...CAPTURED_BENCHMARK_CONFIG, policy: "shift-cache" } as const;
const results = variants.map((variant) => ({
  variant,
  result: runSimulation(config, CAPTURED_TRACE, variant.controls),
}));
const lfuResult = runSimulation(
  { ...CAPTURED_BENCHMARK_CONFIG, policy: "lfu" },
  CAPTURED_TRACE,
  baseControls,
);
const fixed = results[0].result;
const fixedPostBoundary = postBoundarySummary(fixed);
const lfuPostBoundary = postBoundarySummary(lfuResult);

const variantResults = results.map(({ variant, result }) => {
  const postBoundary = postBoundarySummary(result);
  return {
    id: variant.id,
    description: variant.description,
    controls: result.controls,
    detectedShiftEvents: result.timeline
      .filter((point) => point.shiftDetected)
      .map((point) => ({ token: point.token, scoreBits: point.shiftScore })),
    metrics: result.metrics,
    postBoundary,
    versusFixedFrequencyRecency: {
      wholeRunModeledLinkBytesPercentChange: percentChange(
        result.metrics.totalBytesTransferred,
        fixed.metrics.totalBytesTransferred,
      ),
      postBoundaryModeledLinkBytesPercentChange: percentChange(
        postBoundary.aggregate.totalModeledLinkBytes,
        fixedPostBoundary.aggregate.totalModeledLinkBytes,
      ),
    },
    versusLfu: {
      wholeRunModeledLinkBytesPercentChange: percentChange(
        result.metrics.totalBytesTransferred,
        lfuResult.metrics.totalBytesTransferred,
      ),
      postBoundaryModeledLinkBytesPercentChange: percentChange(
        postBoundary.aggregate.totalModeledLinkBytes,
        lfuPostBoundary.aggregate.totalModeledLinkBytes,
      ),
    },
  };
});

const bestWholeRun = [...variantResults].sort(
  (left, right) =>
    left.metrics.totalBytesTransferred - right.metrics.totalBytesTransferred,
)[0];
const bestPostBoundary = [...variantResults].sort(
  (left, right) =>
    left.postBoundary.aggregate.totalModeledLinkBytes -
    right.postBoundary.aggregate.totalModeledLinkBytes,
)[0];

const output = {
  benchmark: "switch-base-8-retention-factorial-v1",
  question:
    "With prefetch disabled, how do JSD-score reweighting and transition retention affect modeled traffic on the pinned captured trace?",
  evidenceBoundary:
    "Router selections came from a pinned Switch-Base-8 encoder execution and semantic boundaries come from its public prompt manifest. Placement traffic and stalls remain simulated. This 215-token diagnostic trace cannot validate change detection, language-model quality, or a ShiftQ-MoE research claim.",
  terminology:
    "The current JSD score continuously changes eviction weights; threshold-crossing events are telemetry and do not trigger a separate controller action.",
  traceSha256: createHash("sha256")
    .update(CAPTURED_TRACE_SERIALIZED)
    .digest("hex"),
  traceFingerprint: fingerprintRouterTrace(CAPTURED_TRACE),
  source: CAPTURED_TRACE.source,
  config: CAPTURED_BENCHMARK_CONFIG,
  shiftCacheParameters: SHIFT_CACHE_PARAMETERS,
  semanticGroupBoundaries,
  traceDiagnostics: fixed.traceDiagnostics,
  variants: variantResults,
  lfuReference: {
    metrics: lfuResult.metrics,
    postBoundary: lfuPostBoundary,
  },
  bestVariantByWholeRunModeledLinkBytes: bestWholeRun.id,
  bestVariantByPostBoundaryModeledLinkBytes: bestPostBoundary.id,
};

const serializedOutput = `${JSON.stringify(output, null, 2)}\n`;
writeFileSync(outputUrl, serializedOutput, "utf8");
process.stdout.write(serializedOutput);
