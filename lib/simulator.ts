export const POLICY_IDS = ["lru", "lfu", "shift-cache"] as const;
export type PolicyId = (typeof POLICY_IDS)[number];

export const SCENARIO_IDS = ["steady", "domain-shift", "high-churn"] as const;
export type ScenarioId = (typeof SCENARIO_IDS)[number];

export interface SimulationConfig {
  scenario: ScenarioId;
  policy: PolicyId;
  seed: number;
  tokens: number;
  layers: number;
  expertsPerLayer: number;
  topK: number;
  gpuSlots: number;
  ramSlots: number;
  expertSizeMB: number;
  pcieGBps: number;
  nvmeGBps: number;
  computeMsPerToken: number;
}

/** Mechanism-level controls used for ablations without inventing new policies. */
export interface SimulationControls {
  /** Only affects ShiftCache; baseline behavior keeps transition prefetch enabled. */
  shiftCachePrefetch: boolean;
}

export type ComparisonConfig = Omit<SimulationConfig, "policy">;

export type TraceConfig = Pick<
  SimulationConfig,
  "scenario" | "seed" | "tokens" | "layers" | "expertsPerLayer" | "topK"
>;

interface RouterTraceFields {
  scenario: ScenarioId;
  seed: number;
  tokens: number;
  layers: number;
  expertsPerLayer: number;
  topK: number;
  /** selections[token][layer] contains exact expert indexes selected by the router. */
  selections: number[][][];
}

/** Serialized by StrataMoE Lab before trace provenance was recorded. */
export interface RouterTraceV1 extends RouterTraceFields {
  version: 1;
}

export interface SyntheticRouterTraceSource {
  kind: "synthetic";
  /** Identifies the generator without implying that model weights were executed. */
  generator: string;
}

export interface DatasetTraceWorkload {
  kind: "dataset";
  datasetId: string;
  split: string;
  /** Ordered identifiers; prompt text is intentionally excluded from the trace. */
  exampleIds: string[];
}

export interface PromptManifestTraceWorkload {
  kind: "prompt-manifest";
  /** SHA-256 of an external, ordered prompt manifest. */
  sha256: string;
}

export interface CapturedRouterTraceSource {
  kind: "captured";
  model: {
    id: string;
    /** Immutable 40- or 64-hex model commit/content revision. */
    revision: string;
  };
  tokenizer: {
    /** Immutable 40- or 64-hex tokenizer commit/content revision. */
    revision: string;
  };
  software: {
    transformersVersion: string;
    pytorchVersion: string;
  };
  workload: DatasetTraceWorkload | PromptManifestTraceWorkload;
  capture: {
    seed: number;
    device: string;
    dtype: string;
  };
}

export type RouterTraceSource =
  | SyntheticRouterTraceSource
  | CapturedRouterTraceSource;

/** Canonical trace schema. Importing a v1 trace migrates it to this shape. */
export interface RouterTraceV2 extends RouterTraceFields {
  version: 2;
  source: RouterTraceSource;
}

/** Public input type retains serialized v1 compatibility. Outputs are canonical v2. */
export type RouterTrace = RouterTraceV1 | RouterTraceV2;

export interface TraceDiagnostics {
  totalExpertSelections: number;
  uniqueExperts: number;
  immediateReuseRate: number;
  segmentTokens: number;
  /**
   * Offline upper bound for a free-to-reconfigure static GPU cache per segment.
   * It diagnoses trace locality; it is not a result achieved by any policy.
   */
  segmentOracleHitRate: number;
}

export interface SimulationMetrics {
  gpuHitRate: number;
  ramHitRate: number;
  nvmeMissRate: number;
  bytesPerToken: number;
  transferStallMsPerToken: number;
  tokensPerSecond: number;
  evictions: number;
  prefetchUsefulness: number;
  /** Shared JSD trace signal; identical across policies given the same trace. */
  detectedShifts: number;
  /** Cache policies consume the router trace verbatim, so this must remain zero. */
  semanticRoutingChanges: 0;
  totalAccesses: number;
  gpuHits: number;
  ramHits: number;
  nvmeMisses: number;
  /** Total link traffic: GPU-bound demand + GPU-bound prefetch + NVMe reads. */
  totalBytesTransferred: number;
  /** Demand weights crossing the RAM-to-GPU/PCIe link. */
  demandBytesTransferred: number;
  /** Prefetched weights crossing the RAM-to-GPU/PCIe link. */
  prefetchBytesTransferred: number;
  /** Weights read across the NVMe-to-RAM link; additive to GPU-bound bytes. */
  nvmeBytesRead: number;
  totalTransferStallMs: number;
  gpuEvictions: number;
  ramEvictions: number;
  prefetchesIssued: number;
  prefetchesUseful: number;
  prefetchesWasted: number;
  prefetchesPending: number;
  segmentOracleHitRate: number;
}

export interface TokenTimelinePoint {
  token: number;
  gpuHitRate: number;
  ramHitRate: number;
  nvmeMissRate: number;
  bytesTransferred: number;
  transferStallMs: number;
  shiftScore: number;
  shiftDetected: boolean;
  gpuResident: number;
  ramResident: number;
  prefetchesIssued: number;
  prefetchesUseful: number;
}

export interface TierResidency {
  gpu: string[];
  ram: string[];
  nvme: string[];
}

export interface SimulationResult {
  policy: PolicyId;
  config: SimulationConfig;
  controls: SimulationControls;
  trace: RouterTraceV2;
  traceFingerprint: string;
  traceDiagnostics: TraceDiagnostics;
  metrics: SimulationMetrics;
  timeline: TokenTimelinePoint[];
  finalResidency: TierResidency;
}

export interface PolicyMetadata {
  label: string;
  description: string;
}

export interface ScenarioMetadata {
  label: string;
  description: string;
}

export const POLICY_META: Record<PolicyId, PolicyMetadata> = {
  lru: {
    label: "LRU",
    description: "Evicts the least recently accessed expert.",
  },
  lfu: {
    label: "LFU",
    description: "Evicts the least frequently accessed expert over the full run.",
  },
  "shift-cache": {
    label: "ShiftCache",
    description:
      "Adapts short/long frequency weights using workload shift and exact one-step transitions.",
  },
};

export const SCENARIO_META: Record<ScenarioId, ScenarioMetadata> = {
  steady: {
    label: "Steady locality",
    description: "A stable, repeating expert working set with occasional local variation.",
  },
  "domain-shift": {
    label: "Mid-run domain shift",
    description: "The active expert working set changes near the midpoint of the trace.",
  },
  "high-churn": {
    label: "High churn",
    description: "Router selections range broadly across experts with weak temporal locality.",
  },
};

export const DEFAULT_CONFIG: SimulationConfig = Object.freeze({
  scenario: "domain-shift",
  policy: "shift-cache",
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
});

export const DEFAULT_SIMULATION_CONTROLS: SimulationControls = Object.freeze({
  shiftCachePrefetch: true,
});

/** Runtime limits protect browser/CLI callers before trace allocation begins. */
export const SIMULATION_LIMITS = Object.freeze({
  maxTokens: 4096,
  maxLayers: 64,
  maxExpertsPerLayer: 128,
  maxTopK: 16,
  maxGpuSlots: 256,
  maxRamSlots: 1024,
  minExpertSizeMB: 1,
  maxExpertSizeMB: 4096,
  minPcieGBps: 0.5,
  maxPcieGBps: 128,
  minNvmeGBps: 0.1,
  maxNvmeGBps: 32,
  maxComputeMsPerToken: 60_000,
  maxTotalExperts: 8192,
  maxTotalSelections: 1_000_000,
});

const POLICY_SET = new Set<string>(POLICY_IDS);
const SCENARIO_SET = new Set<string>(SCENARIO_IDS);
const TRACE_VERSION = 2 as const;
const LEGACY_TRACE_VERSION = 1 as const;
const SYNTHETIC_GENERATOR = "stratamoe-lab/router-trace-v2";
const LEGACY_SYNTHETIC_GENERATOR = "stratamoe-lab/legacy-v1-import";
const MEGABYTE = 1_000_000;
const IMMUTABLE_REVISION_PATTERN = /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;

function assertRecord(value: unknown, label: string): asserts value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object.`);
  }
}

function assertFinite(value: unknown, label: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RangeError(`${label} must be a finite number.`);
  }
}

function assertInteger(value: unknown, label: string, minimum: number): asserts value is number {
  assertFinite(value, label);
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new RangeError(`${label} must be an integer greater than or equal to ${minimum}.`);
  }
}

function validatedString(value: unknown, label: string, maximum = 512): string {
  if (typeof value !== "string" || value.length === 0 || value.trim() !== value) {
    throw new TypeError(`${label} must be a non-empty string without surrounding whitespace.`);
  }
  if (value.length > maximum) {
    throw new RangeError(`${label} must be no longer than ${maximum} characters.`);
  }
  return value;
}

function assertAllowedKeys(
  value: Record<string, unknown>,
  label: string,
  allowed: readonly string[],
): void {
  const allowedSet = new Set(allowed);
  const unknown = Object.keys(value).filter((key) => !allowedSet.has(key));
  if (unknown.length > 0) {
    throw new TypeError(`${label} contains unsupported field(s): ${unknown.join(", ")}.`);
  }
}

function validatedImmutableRevision(value: unknown, label: string): string {
  const revision = validatedString(value, label, 64);
  if (!IMMUTABLE_REVISION_PATTERN.test(revision)) {
    throw new RangeError(
      `${label} must be an immutable lowercase 40- or 64-character hexadecimal revision.`,
    );
  }
  return revision;
}

function assertPositive(value: unknown, label: string): asserts value is number {
  assertFinite(value, label);
  if (value <= 0) {
    throw new RangeError(`${label} must be greater than zero.`);
  }
}

function assertRange(value: number, label: string, minimum: number, maximum: number): void {
  if (value < minimum || value > maximum) {
    throw new RangeError(`${label} must be between ${minimum} and ${maximum}.`);
  }
}

function validateTraceScale(
  tokens: number,
  layers: number,
  expertsPerLayer: number,
  topK: number,
): void {
  assertRange(tokens, "tokens", 1, SIMULATION_LIMITS.maxTokens);
  assertRange(layers, "layers", 1, SIMULATION_LIMITS.maxLayers);
  assertRange(
    expertsPerLayer,
    "expertsPerLayer",
    1,
    SIMULATION_LIMITS.maxExpertsPerLayer,
  );
  assertRange(topK, "topK", 1, SIMULATION_LIMITS.maxTopK);
  const totalExperts = layers * expertsPerLayer;
  if (totalExperts > SIMULATION_LIMITS.maxTotalExperts) {
    throw new RangeError(
      `layers * expertsPerLayer cannot exceed ${SIMULATION_LIMITS.maxTotalExperts}.`,
    );
  }
  const totalSelections = tokens * layers * topK;
  if (totalSelections > SIMULATION_LIMITS.maxTotalSelections) {
    throw new RangeError(
      `tokens * layers * topK cannot exceed ${SIMULATION_LIMITS.maxTotalSelections}.`,
    );
  }
}

export function validateSimulationConfig(input: SimulationConfig): SimulationConfig {
  assertRecord(input, "Simulation config");
  if (typeof input.scenario !== "string" || !SCENARIO_SET.has(input.scenario)) {
    throw new RangeError(`scenario must be one of: ${SCENARIO_IDS.join(", ")}.`);
  }
  if (typeof input.policy !== "string" || !POLICY_SET.has(input.policy)) {
    throw new RangeError(`policy must be one of: ${POLICY_IDS.join(", ")}.`);
  }

  assertInteger(input.seed, "seed", 0);
  if (input.seed > 0xffff_ffff) {
    throw new RangeError("seed must be no greater than 4294967295.");
  }
  assertInteger(input.tokens, "tokens", 1);
  assertInteger(input.layers, "layers", 1);
  assertInteger(input.expertsPerLayer, "expertsPerLayer", 1);
  assertInteger(input.topK, "topK", 1);
  assertInteger(input.gpuSlots, "gpuSlots", 1);
  assertInteger(input.ramSlots, "ramSlots", 0);
  assertPositive(input.expertSizeMB, "expertSizeMB");
  assertPositive(input.pcieGBps, "pcieGBps");
  assertPositive(input.nvmeGBps, "nvmeGBps");
  assertFinite(input.computeMsPerToken, "computeMsPerToken");
  if (input.computeMsPerToken < 0) {
    throw new RangeError("computeMsPerToken must be greater than or equal to zero.");
  }
  if (input.topK > input.expertsPerLayer) {
    throw new RangeError("topK cannot exceed expertsPerLayer.");
  }
  validateTraceScale(input.tokens, input.layers, input.expertsPerLayer, input.topK);
  assertRange(input.gpuSlots, "gpuSlots", 1, SIMULATION_LIMITS.maxGpuSlots);
  assertRange(input.ramSlots, "ramSlots", 0, SIMULATION_LIMITS.maxRamSlots);
  assertRange(
    input.expertSizeMB,
    "expertSizeMB",
    SIMULATION_LIMITS.minExpertSizeMB,
    SIMULATION_LIMITS.maxExpertSizeMB,
  );
  assertRange(
    input.pcieGBps,
    "pcieGBps",
    SIMULATION_LIMITS.minPcieGBps,
    SIMULATION_LIMITS.maxPcieGBps,
  );
  assertRange(
    input.nvmeGBps,
    "nvmeGBps",
    SIMULATION_LIMITS.minNvmeGBps,
    SIMULATION_LIMITS.maxNvmeGBps,
  );
  assertRange(
    input.computeMsPerToken,
    "computeMsPerToken",
    0,
    SIMULATION_LIMITS.maxComputeMsPerToken,
  );
  const expertBytes = input.expertSizeMB * MEGABYTE;
  const slowestTransferMs =
    (expertBytes / (Math.min(input.pcieGBps, input.nvmeGBps) * 1_000_000_000)) *
    1000;
  if (!Number.isFinite(expertBytes) || !Number.isFinite(slowestTransferMs)) {
    throw new RangeError("Derived expert byte size and transfer time must remain finite and safe.");
  }

  return { ...input };
}

export function validateSimulationControls(
  input: SimulationControls,
): SimulationControls {
  assertRecord(input, "Simulation controls");
  assertAllowedKeys(input, "Simulation controls", ["shiftCachePrefetch"]);
  if (typeof input.shiftCachePrefetch !== "boolean") {
    throw new TypeError("shiftCachePrefetch must be a boolean.");
  }
  return { shiftCachePrefetch: input.shiftCachePrefetch };
}

function validateTraceDescriptor(input: TraceConfig): TraceConfig {
  assertRecord(input, "Trace config");
  const config = validateSimulationConfig({
    ...DEFAULT_CONFIG,
    scenario: input.scenario as ScenarioId,
    seed: input.seed as number,
    tokens: input.tokens as number,
    layers: input.layers as number,
    expertsPerLayer: input.expertsPerLayer as number,
    topK: input.topK as number,
    policy: "lru",
  });
  return {
    scenario: config.scenario,
    seed: config.seed,
    tokens: config.tokens,
    layers: config.layers,
    expertsPerLayer: config.expertsPerLayer,
    topK: config.topK,
  };
}

/** Small seeded generator with identical output across modern JavaScript runtimes. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleWithoutReplacement(
  pool: readonly number[],
  count: number,
  random: () => number,
): number[] {
  const available = [...pool];
  const selected: number[] = [];
  for (let index = 0; index < count; index += 1) {
    const chosen = Math.floor(random() * available.length);
    selected.push(available[chosen]);
    available.splice(chosen, 1);
  }
  return selected;
}

function range(start: number, length: number): number[] {
  return Array.from({ length }, (_, index) => start + index);
}

function patternedSelection(
  pool: readonly number[],
  topK: number,
  tokenInPhase: number,
  layer: number,
  random: () => number,
  noiseProbability: number,
): number[] {
  const selected: number[] = [];
  const center = (tokenInPhase + layer * 3) % pool.length;

  for (let rank = 0; rank < topK; rank += 1) {
    let candidate = pool[(center + rank * 2) % pool.length];
    if (random() < noiseProbability) {
      candidate = pool[Math.floor(random() * pool.length)];
    }
    if (selected.includes(candidate)) {
      candidate = pool.find((expert) => !selected.includes(expert)) ?? candidate;
    }
    selected.push(candidate);
  }

  return selected;
}

function domainShiftSelection(
  phasePool: readonly number[],
  topK: number,
  tokenInPhase: number,
  layer: number,
  random: () => number,
): number[] {
  const hotWidth = Math.max(topK, Math.ceil(phasePool.length / 2));
  const hotPool = phasePool.slice(0, hotWidth);
  const coldPool = phasePool.slice(hotWidth);
  const cycleLength = 12;
  const scanTokens = 2;
  const cyclePosition = tokenInPhase % cycleLength;
  const scanning =
    coldPool.length >= topK && cyclePosition >= cycleLength - scanTokens;
  const selectedPool = scanning ? coldPool : hotPool;
  const localToken = scanning
    ? (cyclePosition - (cycleLength - scanTokens)) * 2
    : tokenInPhase;
  return patternedSelection(
    selectedPool,
    topK,
    localToken,
    layer,
    random,
    0.02,
  );
}

export function generateRouterTrace(input: TraceConfig): RouterTraceV2 {
  const config = validateTraceDescriptor(input);
  const random = createRandom(config.seed);
  const allExperts = range(0, config.expertsPerLayer);
  const selections: number[][][] = [];
  const steadyWidth = Math.max(
    config.topK,
    Math.min(config.expertsPerLayer, Math.ceil(config.expertsPerLayer * 0.375)),
  );
  const phaseWidth = Math.max(
    config.topK,
    Math.min(config.expertsPerLayer, Math.ceil(config.expertsPerLayer / 2)),
  );
  const shiftToken = Math.floor(config.tokens / 2);

  for (let token = 0; token < config.tokens; token += 1) {
    const tokenSelections: number[][] = [];
    for (let layer = 0; layer < config.layers; layer += 1) {
      if (config.scenario === "high-churn") {
        tokenSelections.push(
          sampleWithoutReplacement(allExperts, config.topK, random),
        );
        continue;
      }

      if (config.scenario === "steady") {
        tokenSelections.push(
          patternedSelection(
            allExperts.slice(0, steadyWidth),
            config.topK,
            token,
            layer,
            random,
            0.08,
          ),
        );
        continue;
      }

      const afterShift = token >= shiftToken;
      const poolStart = afterShift ? config.expertsPerLayer - phaseWidth : 0;
      const phaseToken = afterShift ? token - shiftToken : token;
      tokenSelections.push(
        domainShiftSelection(
          allExperts.slice(poolStart, poolStart + phaseWidth),
          config.topK,
          phaseToken,
          layer,
          random,
        ),
      );
    }
    selections.push(tokenSelections);
  }

  return {
    version: TRACE_VERSION,
    source: {
      kind: "synthetic",
      generator: SYNTHETIC_GENERATOR,
    },
    scenario: config.scenario,
    seed: config.seed,
    tokens: config.tokens,
    layers: config.layers,
    expertsPerLayer: config.expertsPerLayer,
    topK: config.topK,
    selections,
  };
}

function validateRouterTraceSource(
  value: unknown,
  traceSeed: number,
): RouterTraceSource {
  assertRecord(value, "Router trace source");

  if (value.kind === "synthetic") {
    assertAllowedKeys(value, "Synthetic router trace source", ["kind", "generator"]);
    return {
      kind: "synthetic",
      generator: validatedString(
        value.generator,
        "Synthetic router trace source generator",
      ),
    };
  }

  if (value.kind !== "captured") {
    throw new RangeError("Router trace source kind must be synthetic or captured.");
  }

  assertAllowedKeys(value, "Captured router trace source", [
    "kind",
    "model",
    "tokenizer",
    "software",
    "workload",
    "capture",
  ]);

  assertRecord(value.model, "Captured router trace model");
  assertAllowedKeys(value.model, "Captured router trace model", ["id", "revision"]);
  const model = {
    id: validatedString(value.model.id, "Captured router trace model id"),
    revision: validatedImmutableRevision(
      value.model.revision,
      "Captured router trace model revision",
    ),
  };

  assertRecord(value.tokenizer, "Captured router trace tokenizer");
  assertAllowedKeys(value.tokenizer, "Captured router trace tokenizer", ["revision"]);
  const tokenizer = {
    revision: validatedImmutableRevision(
      value.tokenizer.revision,
      "Captured router trace tokenizer revision",
    ),
  };

  assertRecord(value.software, "Captured router trace software");
  assertAllowedKeys(value.software, "Captured router trace software", [
    "transformersVersion",
    "pytorchVersion",
  ]);
  const software = {
    transformersVersion: validatedString(
      value.software.transformersVersion,
      "Captured router trace Transformers version",
      128,
    ),
    pytorchVersion: validatedString(
      value.software.pytorchVersion,
      "Captured router trace PyTorch version",
      128,
    ),
  };

  assertRecord(value.workload, "Captured router trace workload");
  let workload: DatasetTraceWorkload | PromptManifestTraceWorkload;
  if (value.workload.kind === "dataset") {
    assertAllowedKeys(value.workload, "Captured dataset workload", [
      "kind",
      "datasetId",
      "split",
      "exampleIds",
    ]);
    if (!Array.isArray(value.workload.exampleIds) || value.workload.exampleIds.length === 0) {
      throw new RangeError(
        "Captured dataset workload exampleIds must be a non-empty ordered array.",
      );
    }
    if (value.workload.exampleIds.length > SIMULATION_LIMITS.maxTokens) {
      throw new RangeError(
        `Captured dataset workload exampleIds cannot exceed ${SIMULATION_LIMITS.maxTokens} entries.`,
      );
    }
    workload = {
      kind: "dataset",
      datasetId: validatedString(
        value.workload.datasetId,
        "Captured dataset workload datasetId",
      ),
      split: validatedString(value.workload.split, "Captured dataset workload split", 128),
      exampleIds: value.workload.exampleIds.map((exampleId, index) =>
        validatedString(
          exampleId,
          `Captured dataset workload exampleIds[${index}]`,
        ),
      ),
    };
  } else if (value.workload.kind === "prompt-manifest") {
    assertAllowedKeys(value.workload, "Captured prompt-manifest workload", [
      "kind",
      "sha256",
    ]);
    const sha256 = validatedString(
      value.workload.sha256,
      "Captured prompt-manifest workload sha256",
      64,
    );
    if (!SHA256_PATTERN.test(sha256)) {
      throw new RangeError(
        "Captured prompt-manifest workload sha256 must be a lowercase 64-character hexadecimal digest.",
      );
    }
    workload = { kind: "prompt-manifest", sha256 };
  } else {
    throw new RangeError(
      "Captured router trace workload kind must be dataset or prompt-manifest.",
    );
  }

  assertRecord(value.capture, "Captured router trace capture");
  assertAllowedKeys(value.capture, "Captured router trace capture", [
    "seed",
    "device",
    "dtype",
  ]);
  assertInteger(value.capture.seed, "Captured router trace capture seed", 0);
  if (value.capture.seed > 0xffff_ffff) {
    throw new RangeError(
      "Captured router trace capture seed must be no greater than 4294967295.",
    );
  }
  if (value.capture.seed !== traceSeed) {
    throw new RangeError(
      `Captured router trace capture seed (${value.capture.seed}) must match trace seed (${traceSeed}).`,
    );
  }
  const capture = {
    seed: value.capture.seed,
    device: validatedString(value.capture.device, "Captured router trace capture device", 128),
    dtype: validatedString(value.capture.dtype, "Captured router trace capture dtype", 128),
  };

  return {
    kind: "captured",
    model,
    tokenizer,
    software,
    workload,
    capture,
  };
}

export function validateRouterTrace(
  value: unknown,
  expected?: Partial<TraceConfig>,
): RouterTraceV2 {
  assertRecord(value, "Router trace");
  if (value.version !== TRACE_VERSION && value.version !== LEGACY_TRACE_VERSION) {
    throw new RangeError(
      `Router trace version must be ${LEGACY_TRACE_VERSION} or ${TRACE_VERSION}.`,
    );
  }
  if (typeof value.scenario !== "string" || !SCENARIO_SET.has(value.scenario)) {
    throw new RangeError(`Router trace scenario must be one of: ${SCENARIO_IDS.join(", ")}.`);
  }
  assertInteger(value.seed, "Router trace seed", 0);
  if (value.seed > 0xffff_ffff) {
    throw new RangeError("Router trace seed must be no greater than 4294967295.");
  }
  assertInteger(value.tokens, "Router trace tokens", 1);
  assertInteger(value.layers, "Router trace layers", 1);
  assertInteger(value.expertsPerLayer, "Router trace expertsPerLayer", 1);
  assertInteger(value.topK, "Router trace topK", 1);
  if (value.topK > value.expertsPerLayer) {
    throw new RangeError("Router trace topK cannot exceed expertsPerLayer.");
  }
  validateTraceScale(value.tokens, value.layers, value.expertsPerLayer, value.topK);
  if (!Array.isArray(value.selections) || value.selections.length !== value.tokens) {
    throw new RangeError("Router trace selections must contain exactly one entry per token.");
  }

  // Copy validated scalar fields before callbacks so TypeScript retains their narrowing.
  const tokens = value.tokens;
  const layers = value.layers;
  const expertsPerLayer = value.expertsPerLayer;
  const topK = value.topK;

  const selections = value.selections.map((tokenValue, token) => {
    if (!Array.isArray(tokenValue) || tokenValue.length !== layers) {
      throw new RangeError(`Router trace token ${token} must contain exactly ${layers} layers.`);
    }
    return tokenValue.map((layerValue, layer) => {
      if (!Array.isArray(layerValue) || layerValue.length !== topK) {
        throw new RangeError(
          `Router trace token ${token}, layer ${layer} must contain exactly ${topK} experts.`,
        );
      }
      const seen = new Set<number>();
      return layerValue.map((expertValue, rank) => {
        assertInteger(
          expertValue,
          `Router trace token ${token}, layer ${layer}, rank ${rank}`,
          0,
        );
        if (expertValue >= expertsPerLayer) {
          throw new RangeError(
            `Router trace token ${token}, layer ${layer}, rank ${rank} is outside the expert range.`,
          );
        }
        if (seen.has(expertValue)) {
          throw new RangeError(
            `Router trace token ${token}, layer ${layer} selects expert ${expertValue} more than once.`,
          );
        }
        seen.add(expertValue);
        return expertValue;
      });
    });
  });

  const source: RouterTraceSource =
    value.version === LEGACY_TRACE_VERSION
      ? {
          kind: "synthetic",
          generator: LEGACY_SYNTHETIC_GENERATOR,
        }
      : validateRouterTraceSource(value.source, value.seed);

  const trace: RouterTraceV2 = {
    version: TRACE_VERSION,
    source,
    scenario: value.scenario as ScenarioId,
    seed: value.seed,
    tokens,
    layers,
    expertsPerLayer,
    topK,
    selections,
  };

  if (expected) {
    for (const field of [
      "scenario",
      "seed",
      "tokens",
      "layers",
      "expertsPerLayer",
      "topK",
    ] as const) {
      const expectedValue = expected[field];
      if (expectedValue !== undefined && trace[field] !== expectedValue) {
        throw new RangeError(
          `Router trace ${field} (${trace[field]}) does not match config (${expectedValue}).`,
        );
      }
    }
  }

  return trace;
}

export function exportRouterTrace(trace: RouterTrace): string {
  return JSON.stringify(validateRouterTrace(trace), null, 2);
}

export function importRouterTrace(
  serialized: string,
  expected?: Partial<TraceConfig>,
): RouterTraceV2 {
  if (typeof serialized !== "string") {
    throw new TypeError("Serialized router trace must be a string.");
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized) as unknown;
  } catch (error) {
    const message = error instanceof Error ? error.message : "unknown JSON error";
    throw new SyntaxError(`Invalid router trace JSON: ${message}`);
  }
  return validateRouterTrace(parsed, expected);
}

function expertKey(layer: number, expert: number): string {
  return `L${layer}:E${expert}`;
}

function flattenToken(token: readonly (readonly number[])[]): string[] {
  const flattened: string[] = [];
  for (let layer = 0; layer < token.length; layer += 1) {
    for (const expert of token[layer]) {
      flattened.push(expertKey(layer, expert));
    }
  }
  return flattened;
}

export function fingerprintRouterTrace(traceInput: RouterTrace): string {
  const trace = validateRouterTrace(traceInput);
  let hash = 0x811c9dc5;
  const serialized = JSON.stringify(trace);
  for (let index = 0; index < serialized.length; index += 1) {
    const value = serialized.charCodeAt(index);
    hash ^= value & 0xff;
    hash = Math.imul(hash, 0x01000193) >>> 0;
    hash ^= (value >>> 8) & 0xff;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

function round(value: number, digits = 8): number {
  const scale = 10 ** digits;
  return Math.round((value + Number.EPSILON) * scale) / scale;
}

export function analyzeRouterTrace(
  traceInput: RouterTrace,
  gpuSlots: number,
  segmentTokens = 16,
): TraceDiagnostics {
  const trace = validateRouterTrace(traceInput);
  assertInteger(gpuSlots, "gpuSlots", 1);
  assertInteger(segmentTokens, "segmentTokens", 1);
  const flattened = trace.selections.map(flattenToken);
  const all = flattened.flat();
  const uniqueExperts = new Set(all).size;
  let immediateReuses = 0;
  for (let token = 1; token < flattened.length; token += 1) {
    const previous = new Set(flattened[token - 1]);
    for (const key of flattened[token]) {
      if (previous.has(key)) immediateReuses += 1;
    }
  }

  let oracleHits = 0;
  for (let start = 0; start < flattened.length; start += segmentTokens) {
    const frequencies = new Map<string, number>();
    for (const key of flattened.slice(start, start + segmentTokens).flat()) {
      frequencies.set(key, (frequencies.get(key) ?? 0) + 1);
    }
    const best = [...frequencies.values()].sort((left, right) => right - left);
    oracleHits += best.slice(0, gpuSlots).reduce((sum, count) => sum + count, 0);
  }

  return {
    totalExpertSelections: all.length,
    uniqueExperts,
    immediateReuseRate: round(
      trace.tokens <= 1
        ? 0
        : immediateReuses /
            ((trace.tokens - 1) * trace.layers * trace.topK),
    ),
    segmentTokens,
    segmentOracleHitRate: round(all.length === 0 ? 0 : oracleHits / all.length),
  };
}

export const SHIFT_CACHE_PARAMETERS = Object.freeze({
  minimumShortWindowTokens: 4,
  maximumShortWindowTokens: 12,
  longWindowMultiplier: 4,
  shiftThresholdBits: 0.28,
  rearmThresholdBits: 0.12,
  minimumTransitionObservations: 2,
  maximumPrefetchesPerToken: 2,
});

interface CacheCounters {
  totalAccesses: number;
  gpuHits: number;
  ramHits: number;
  nvmeMisses: number;
  demandBytesTransferred: number;
  prefetchBytesTransferred: number;
  nvmeBytesRead: number;
  totalTransferStallMs: number;
  gpuEvictions: number;
  ramEvictions: number;
  prefetchesIssued: number;
  prefetchesUseful: number;
  prefetchesWasted: number;
}

type AccessSource = "gpu" | "ram" | "nvme";

function incrementCount(map: Map<string, number>, key: string, amount = 1): void {
  const next = (map.get(key) ?? 0) + amount;
  if (next === 0) map.delete(key);
  else map.set(key, next);
}

function probabilityDistribution(
  counts: ReadonlyMap<string, number>,
): Map<string, number> {
  const total = [...counts.values()].reduce((sum, count) => sum + count, 0);
  const distribution = new Map<string, number>();
  if (total <= 0) return distribution;
  for (const [key, count] of counts) {
    if (count > 0) distribution.set(key, count / total);
  }
  return distribution;
}

/** Jensen-Shannon divergence in bits; bounded to [0, 1] for two distributions. */
export function jensenShannonDivergence(
  leftCounts: ReadonlyMap<string, number>,
  rightCounts: ReadonlyMap<string, number>,
): number {
  const left = probabilityDistribution(leftCounts);
  const right = probabilityDistribution(rightCounts);
  if (left.size === 0 || right.size === 0) return 0;
  const keys = new Set([...left.keys(), ...right.keys()]);
  let divergence = 0;
  for (const key of keys) {
    const leftProbability = left.get(key) ?? 0;
    const rightProbability = right.get(key) ?? 0;
    const midpoint = (leftProbability + rightProbability) / 2;
    if (leftProbability > 0) {
      divergence += 0.5 * leftProbability * Math.log2(leftProbability / midpoint);
    }
    if (rightProbability > 0) {
      divergence += 0.5 * rightProbability * Math.log2(rightProbability / midpoint);
    }
  }
  return round(Math.min(1, Math.max(0, divergence)), 12);
}

class ShiftTracker {
  readonly shortWindowTokens: number;
  readonly longWindowTokens: number;
  readonly shortCounts = new Map<string, number>();
  readonly longCounts = new Map<string, number>();
  currentScore = 0;
  detectedShifts = 0;

  private readonly shortQueue: string[][] = [];
  private readonly longQueue: string[][] = [];
  private armed = true;

  constructor(tokens: number) {
    this.shortWindowTokens = Math.max(
      SHIFT_CACHE_PARAMETERS.minimumShortWindowTokens,
      Math.min(
        SHIFT_CACHE_PARAMETERS.maximumShortWindowTokens,
        Math.round(tokens * 0.05),
      ),
    );
    this.longWindowTokens = this.shortWindowTokens * SHIFT_CACHE_PARAMETERS.longWindowMultiplier;
  }

  observe(experts: readonly string[]): { score: number; detected: boolean } {
    const tokenExperts = [...experts];
    this.shortQueue.push(tokenExperts);
    this.longQueue.push(tokenExperts);
    for (const key of tokenExperts) {
      incrementCount(this.shortCounts, key);
      incrementCount(this.longCounts, key);
    }

    while (this.shortQueue.length > this.shortWindowTokens) {
      for (const key of this.shortQueue.shift() ?? []) {
        incrementCount(this.shortCounts, key, -1);
      }
    }
    while (this.longQueue.length > this.longWindowTokens) {
      for (const key of this.longQueue.shift() ?? []) {
        incrementCount(this.longCounts, key, -1);
      }
    }

    if (this.shortQueue.length < this.shortWindowTokens || this.longQueue.length < this.shortWindowTokens * 2) {
      this.currentScore = 0;
      return { score: 0, detected: false };
    }

    const baselineCounts = new Map(this.longCounts);
    for (const [key, count] of this.shortCounts) {
      incrementCount(baselineCounts, key, -count);
    }
    this.currentScore = jensenShannonDivergence(this.shortCounts, baselineCounts);

    let detected = false;
    if (this.armed && this.currentScore >= SHIFT_CACHE_PARAMETERS.shiftThresholdBits) {
      this.detectedShifts += 1;
      this.armed = false;
      detected = true;
    } else if (this.currentScore <= SHIFT_CACHE_PARAMETERS.rearmThresholdBits) {
      this.armed = true;
    }
    return { score: this.currentScore, detected };
  }
}

function parseExpertKey(key: string): [number, number] {
  const match = /^L(\d+):E(\d+)$/.exec(key);
  if (!match) throw new Error(`Internal invalid expert key: ${key}`);
  return [Number(match[1]), Number(match[2])];
}

function compareExpertKeys(left: string, right: string): number {
  const [leftLayer, leftExpert] = parseExpertKey(left);
  const [rightLayer, rightExpert] = parseExpertKey(right);
  return leftLayer - rightLayer || leftExpert - rightExpert;
}

function transferMilliseconds(bytes: number, gigabytesPerSecond: number): number {
  return (bytes / (gigabytesPerSecond * 1_000_000_000)) * 1000;
}

class HierarchySimulator {
  readonly gpu = new Set<string>();
  readonly ram = new Set<string>();
  readonly counters: CacheCounters = {
    totalAccesses: 0,
    gpuHits: 0,
    ramHits: 0,
    nvmeMisses: 0,
    demandBytesTransferred: 0,
    prefetchBytesTransferred: 0,
    nvmeBytesRead: 0,
    totalTransferStallMs: 0,
    gpuEvictions: 0,
    ramEvictions: 0,
    prefetchesIssued: 0,
    prefetchesUseful: 0,
    prefetchesWasted: 0,
  };
  readonly shiftTracker: ShiftTracker;

  private readonly totalFrequency = new Map<string, number>();
  private readonly lastAccess = new Map<string, number>();
  private readonly transitions = new Map<string, Map<string, number>>();
  private readonly prefetched = new Set<string>();
  private predictionScores = new Map<string, number>();
  private previousLayers: string[][] | undefined;
  private clock = 0;
  private readonly config: SimulationConfig;
  private readonly controls: SimulationControls;

  constructor(config: SimulationConfig, controls: SimulationControls) {
    this.config = config;
    this.controls = controls;
    this.shiftTracker = new ShiftTracker(config.tokens);
  }

  demand(key: string): AccessSource {
    this.clock += 1;
    this.counters.totalAccesses += 1;
    incrementCount(this.totalFrequency, key);

    if (this.gpu.has(key)) {
      this.counters.gpuHits += 1;
      this.lastAccess.set(key, this.clock);
      if (this.prefetched.delete(key)) this.counters.prefetchesUseful += 1;
      return "gpu";
    }

    const source: AccessSource = this.ram.has(key) ? "ram" : "nvme";
    if (source === "ram") {
      this.counters.ramHits += 1;
      this.ram.delete(key);
    } else {
      this.counters.nvmeMisses += 1;
      this.counters.nvmeBytesRead += this.expertBytes;
    }
    this.counters.demandBytesTransferred += this.expertBytes;
    this.counters.totalTransferStallMs += transferMilliseconds(
      this.expertBytes,
      this.config.pcieGBps,
    );
    if (source === "nvme") {
      this.counters.totalTransferStallMs += transferMilliseconds(
        this.expertBytes,
        this.config.nvmeGBps,
      );
    }
    this.insertGpu(key);
    this.lastAccess.set(key, this.clock);
    return source;
  }

  finishToken(layerSelections: string[][], hasNextToken: boolean): {
    shiftScore: number;
    shiftDetected: boolean;
  } {
    const flattened = layerSelections.flat();
    const shift = this.shiftTracker.observe(flattened);
    if (this.previousLayers) this.learnTransitions(this.previousLayers, layerSelections);
    this.updatePredictions(layerSelections);
    if (
      this.config.policy === "shift-cache" &&
      this.controls.shiftCachePrefetch &&
      hasNextToken
    ) {
      this.prefetchPredictions();
    }
    this.previousLayers = layerSelections.map((layer) => [...layer]);
    return { shiftScore: shift.score, shiftDetected: shift.detected };
  }

  private get expertBytes(): number {
    return this.config.expertSizeMB * MEGABYTE;
  }

  get pendingPrefetches(): number {
    return this.prefetched.size;
  }

  private insertGpu(key: string): void {
    if (this.gpu.has(key)) return;
    if (this.gpu.size >= this.config.gpuSlots) {
      const victim = this.chooseVictim(this.gpu);
      this.gpu.delete(victim);
      this.counters.gpuEvictions += 1;
      if (this.prefetched.delete(victim)) this.counters.prefetchesWasted += 1;
      this.insertRam(victim);
    }
    this.gpu.add(key);
  }

  private insertRam(key: string): void {
    if (this.config.ramSlots === 0 || this.ram.has(key)) return;
    if (this.ram.size >= this.config.ramSlots) {
      this.ram.delete(this.chooseVictim(this.ram));
      this.counters.ramEvictions += 1;
    }
    this.ram.add(key);
  }

  private chooseVictim(candidates: ReadonlySet<string>): string {
    const ordered = [...candidates].sort(compareExpertKeys);
    if (ordered.length === 0) throw new Error("Cannot choose a victim from an empty tier.");

    const accessTimes = ordered.map((key) => this.lastAccess.get(key) ?? -1);
    const oldestAccess = Math.min(...accessTimes);
    const newestAccess = Math.max(...accessTimes);

    const retention = (key: string): number => {
      if (this.config.policy === "lru") return this.lastAccess.get(key) ?? -1;
      if (this.config.policy === "lfu") return this.totalFrequency.get(key) ?? 0;

      const longMaximum = Math.max(1, ...this.shiftTracker.longCounts.values());
      const shortMaximum = Math.max(1, ...this.shiftTracker.shortCounts.values());
      const predictionMaximum = Math.max(1, ...this.predictionScores.values());
      const adaptation = Math.min(
        1,
        this.shiftTracker.currentScore / SHIFT_CACHE_PARAMETERS.shiftThresholdBits,
      );
      const longWeight = 0.55 - 0.45 * adaptation;
      const shortWeight = 0.15 + 0.3 * adaptation;
      const transitionWeight = 0.05 + 0.1 * adaptation;
      const recencyWeight = 1 - longWeight - shortWeight - transitionWeight;
      const accessTime = this.lastAccess.get(key) ?? -1;
      const recency =
        newestAccess === oldestAccess
          ? 0
          : (accessTime - oldestAccess) / (newestAccess - oldestAccess);
      return (
        longWeight * ((this.shiftTracker.longCounts.get(key) ?? 0) / longMaximum) +
        shortWeight * ((this.shiftTracker.shortCounts.get(key) ?? 0) / shortMaximum) +
        transitionWeight * ((this.predictionScores.get(key) ?? 0) / predictionMaximum) +
        recencyWeight * recency
      );
    };

    let victim = ordered[0];
    let victimScore = retention(victim);
    for (const candidate of ordered.slice(1)) {
      const candidateScore = retention(candidate);
      const candidateLastAccess = this.lastAccess.get(candidate) ?? -1;
      const victimLastAccess = this.lastAccess.get(victim) ?? -1;
      if (
        candidateScore < victimScore - Number.EPSILON ||
        (Math.abs(candidateScore - victimScore) <= Number.EPSILON &&
          candidateLastAccess < victimLastAccess)
      ) {
        victim = candidate;
        victimScore = candidateScore;
      }
    }
    return victim;
  }

  private learnTransitions(previous: string[][], current: string[][]): void {
    for (let layer = 0; layer < current.length; layer += 1) {
      const source = [...previous[layer]].sort(compareExpertKeys).join("|");
      let targets = this.transitions.get(source);
      if (!targets) {
        targets = new Map<string, number>();
        this.transitions.set(source, targets);
      }
      for (const target of current[layer]) incrementCount(targets, target);
    }
  }

  private updatePredictions(current: string[][]): void {
    const scores = new Map<string, number>();
    for (const layer of current) {
      const source = [...layer].sort(compareExpertKeys).join("|");
      const targets = this.transitions.get(source);
      const strongest = Math.max(0, ...(targets?.values() ?? []));
      for (const [target, count] of targets ?? []) {
        // Ignore weak alternative edges caused by trace noise or rare scan exits.
        if (strongest > 0 && count < strongest * 0.8) continue;
        incrementCount(scores, target, count);
      }
    }
    this.predictionScores = scores;
  }

  private prefetchPredictions(): void {
    const budget = Math.min(
      SHIFT_CACHE_PARAMETERS.maximumPrefetchesPerToken,
      Math.floor(
        this.config.gpuSlots / Math.max(1, this.config.layers * this.config.topK),
      ),
    );
    if (budget <= 0) return;

    const candidates = [...this.predictionScores.entries()]
      .filter(
        ([key, score]) =>
          score >= SHIFT_CACHE_PARAMETERS.minimumTransitionObservations && !this.gpu.has(key),
      )
      .sort((left, right) => right[1] - left[1] || compareExpertKeys(left[0], right[0]))
      .slice(0, budget);

    for (const [key] of candidates) {
      this.clock += 1;
      const fromNvme = !this.ram.has(key);
      if (!fromNvme) this.ram.delete(key);
      else this.counters.nvmeBytesRead += this.expertBytes;
      this.counters.prefetchesIssued += 1;
      this.counters.prefetchBytesTransferred += this.expertBytes;
      // V1 uses conservative serialized transfer accounting; no free overlap is assumed.
      this.counters.totalTransferStallMs += transferMilliseconds(
        this.expertBytes,
        this.config.pcieGBps,
      );
      if (fromNvme) {
        this.counters.totalTransferStallMs += transferMilliseconds(
          this.expertBytes,
          this.config.nvmeGBps,
        );
      }
      this.insertGpu(key);
      this.prefetched.add(key);
      this.lastAccess.set(key, this.clock);
    }
  }
}

function rate(count: number, total: number): number {
  return round(total === 0 ? 0 : count / total);
}

function snapshotCounters(counters: CacheCounters): CacheCounters {
  return { ...counters };
}

function counterDelta(
  after: CacheCounters,
  before: CacheCounters,
  field: keyof CacheCounters,
): number {
  return after[field] - before[field];
}

function traceConfigFromSimulation(config: SimulationConfig): TraceConfig {
  return {
    scenario: config.scenario,
    seed: config.seed,
    tokens: config.tokens,
    layers: config.layers,
    expertsPerLayer: config.expertsPerLayer,
    topK: config.topK,
  };
}

function allExpertKeys(config: SimulationConfig): string[] {
  const keys: string[] = [];
  for (let layer = 0; layer < config.layers; layer += 1) {
    for (let expert = 0; expert < config.expertsPerLayer; expert += 1) {
      keys.push(expertKey(layer, expert));
    }
  }
  return keys;
}

export function runSimulation(
  input: SimulationConfig,
  traceInput?: RouterTrace,
  controlsInput: SimulationControls = DEFAULT_SIMULATION_CONTROLS,
): SimulationResult {
  const config = validateSimulationConfig(input);
  const controls = validateSimulationControls(controlsInput);
  const expectedTrace = traceConfigFromSimulation(config);
  const trace = traceInput
    ? validateRouterTrace(traceInput, expectedTrace)
    : generateRouterTrace(expectedTrace);
  const traceDiagnostics = analyzeRouterTrace(trace, config.gpuSlots);
  const simulator = new HierarchySimulator(config, controls);
  const timeline: TokenTimelinePoint[] = [];

  for (let token = 0; token < trace.tokens; token += 1) {
    const before = snapshotCounters(simulator.counters);
    const layerSelections = trace.selections[token].map((experts, layer) =>
      experts.map((expert) => expertKey(layer, expert)),
    );

    for (const key of layerSelections.flat()) simulator.demand(key);
    const shift = simulator.finishToken(layerSelections, token < trace.tokens - 1);
    const after = simulator.counters;
    const tokenAccesses = counterDelta(after, before, "totalAccesses");
    const gpuHits = counterDelta(after, before, "gpuHits");
    const ramHits = counterDelta(after, before, "ramHits");
    const nvmeMisses = counterDelta(after, before, "nvmeMisses");
    const bytesTransferred =
      counterDelta(after, before, "demandBytesTransferred") +
      counterDelta(after, before, "prefetchBytesTransferred") +
      counterDelta(after, before, "nvmeBytesRead");

    timeline.push({
      token,
      gpuHitRate: rate(gpuHits, tokenAccesses),
      ramHitRate: rate(ramHits, tokenAccesses),
      nvmeMissRate: rate(nvmeMisses, tokenAccesses),
      bytesTransferred: round(bytesTransferred, 2),
      transferStallMs: round(counterDelta(after, before, "totalTransferStallMs")),
      shiftScore: round(shift.shiftScore),
      shiftDetected: shift.shiftDetected,
      gpuResident: simulator.gpu.size,
      ramResident: simulator.ram.size,
      prefetchesIssued: counterDelta(after, before, "prefetchesIssued"),
      prefetchesUseful: counterDelta(after, before, "prefetchesUseful"),
    });
  }

  const counters = simulator.counters;
  const totalBytesTransferred =
    counters.demandBytesTransferred +
    counters.prefetchBytesTransferred +
    counters.nvmeBytesRead;
  const averageStall = counters.totalTransferStallMs / config.tokens;
  const millisecondsPerToken = config.computeMsPerToken + averageStall;
  const allKeys = allExpertKeys(config);
  const finalResidency: TierResidency = {
    gpu: [...simulator.gpu].sort(compareExpertKeys),
    ram: [...simulator.ram].sort(compareExpertKeys),
    nvme: allKeys
      .filter((key) => !simulator.gpu.has(key) && !simulator.ram.has(key))
      .sort(compareExpertKeys),
  };

  return {
    policy: config.policy,
    config,
    controls,
    trace,
    traceFingerprint: fingerprintRouterTrace(trace),
    traceDiagnostics,
    metrics: {
      gpuHitRate: rate(counters.gpuHits, counters.totalAccesses),
      ramHitRate: rate(counters.ramHits, counters.totalAccesses),
      nvmeMissRate: rate(counters.nvmeMisses, counters.totalAccesses),
      bytesPerToken: round(totalBytesTransferred / config.tokens, 2),
      transferStallMsPerToken: round(averageStall),
      tokensPerSecond: round(millisecondsPerToken === 0 ? 0 : 1000 / millisecondsPerToken),
      evictions: counters.gpuEvictions + counters.ramEvictions,
      prefetchUsefulness: rate(counters.prefetchesUseful, counters.prefetchesIssued),
      detectedShifts: simulator.shiftTracker.detectedShifts,
      semanticRoutingChanges: 0,
      totalAccesses: counters.totalAccesses,
      gpuHits: counters.gpuHits,
      ramHits: counters.ramHits,
      nvmeMisses: counters.nvmeMisses,
      totalBytesTransferred: round(totalBytesTransferred, 2),
      demandBytesTransferred: round(counters.demandBytesTransferred, 2),
      prefetchBytesTransferred: round(counters.prefetchBytesTransferred, 2),
      nvmeBytesRead: round(counters.nvmeBytesRead, 2),
      totalTransferStallMs: round(counters.totalTransferStallMs),
      gpuEvictions: counters.gpuEvictions,
      ramEvictions: counters.ramEvictions,
      prefetchesIssued: counters.prefetchesIssued,
      prefetchesUseful: counters.prefetchesUseful,
      prefetchesWasted: counters.prefetchesWasted,
      prefetchesPending: simulator.pendingPrefetches,
      segmentOracleHitRate: traceDiagnostics.segmentOracleHitRate,
    },
    timeline,
    finalResidency,
  };
}

export function runComparison(input: ComparisonConfig): SimulationResult[] {
  const validated = validateSimulationConfig({ ...input, policy: "lru" });
  const comparisonConfig: ComparisonConfig = {
    scenario: validated.scenario,
    seed: validated.seed,
    tokens: validated.tokens,
    layers: validated.layers,
    expertsPerLayer: validated.expertsPerLayer,
    topK: validated.topK,
    gpuSlots: validated.gpuSlots,
    ramSlots: validated.ramSlots,
    expertSizeMB: validated.expertSizeMB,
    pcieGBps: validated.pcieGBps,
    nvmeGBps: validated.nvmeGBps,
    computeMsPerToken: validated.computeMsPerToken,
  };
  const trace = generateRouterTrace(traceConfigFromSimulation(validated));
  return POLICY_IDS.map((policy) =>
    runSimulation({ ...comparisonConfig, policy }, trace),
  );
}
