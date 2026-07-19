# StrataMoE Lab — implementation contract

StrataMoE Lab is a deterministic, trace-driven research harness for studying how Mixture-of-Experts inference behaves when expert weights move through a constrained GPU → RAM → NVMe memory hierarchy.

It is inspired by systems such as Colibrì, but it is not a fork, a model runner, or a claim that a huge model has been executed. The first release is a simulator, benchmark, and inspection dashboard. It must never change the selected experts or silently lower precision; every policy result therefore reports zero semantic routing changes.

## Measured question

Can a shift-aware cache policy reduce expert bytes streamed per token and estimated transfer stalls under domain changes, compared with LRU and LFU, without changing router decisions?

## Required experiment surface

- Scenarios: steady locality, mid-run domain shift, and high churn.
- Policies: LRU, LFU, and `ShiftCache`.
- Configurable token count, layer count, experts per layer, top-k, GPU slots, RAM slots, PCIe bandwidth, NVMe bandwidth, expert size, and seed.
- Metrics: GPU hit rate, RAM hit rate, NVMe miss rate, bytes per token, estimated transfer stall per token, estimated tokens per second, evictions, prefetch usefulness, detected shifts, and semantic routing changes.
- Inspection data: per-token time series and final tier residency.
- Deterministic JSON export/import of router traces.
- Fixed benchmark command and automated tests.

## ShiftCache policy

Maintain short- and long-window expert frequency distributions. Detect a workload shift using Jensen–Shannon divergence. As divergence rises, reduce long-term LFU weight and favor recent frequency plus learned one-step expert transitions. Prefetching may move exact expert weights earlier, but it may not change the trace or selected experts.

## UI direction

Minimal dark research instrument, not a generic admin dashboard. Use graphite, warm white, muted cyan, and restrained amber. No gradients, generated SVG illustrations, glassmorphism, or decorative charts. The first viewport must communicate the measured question, the simulation-only boundary, core controls, and policy comparison.

## Public TypeScript contract

The simulator should export:

```ts
type PolicyId = "lru" | "lfu" | "shift-cache";
type ScenarioId = "steady" | "domain-shift" | "high-churn";

interface SimulationConfig {
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

function runSimulation(config: SimulationConfig): SimulationResult;
function runComparison(config: Omit<SimulationConfig, "policy">): SimulationResult[];
```

Implementations may add fields while preserving these names.
