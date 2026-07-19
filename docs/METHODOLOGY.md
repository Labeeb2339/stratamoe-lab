# Methodology

## 1. Research question

StrataMoE Lab evaluates a cache-policy hypothesis, not a language-model hypothesis:

> Under a fixed expert-routing trace, can a policy that uses changes in the activation distribution reduce simulated lower-tier traffic and bandwidth-derived stalls after a shift, relative to LRU and LFU?

The independent variable is the placement policy. The router trace is controlled and identical across policies. The principal outcomes are bytes moved per token and estimated transfer stall per token. GPU, RAM, and NVMe hit rates are supporting diagnostics.

## 2. Experimental object

A trace is an ordered sequence of tokens. Each token contains the expert IDs selected at each MoE layer. An expert ID is scoped to a layer; for example, layer 3 expert 7 is distinct from layer 4 expert 7.

The built-in generators represent three controlled regimes:

- **steady locality**: a stable, reused working set;
- **domain shift**: an abrupt change from one synthetic activation distribution to another partway through the trace; and
- **high churn**: weak locality with frequent working-set turnover.

These names describe generator behavior. In particular, `domain shift` is not evidence that a real model learned semantic domains. It is a synthetic distribution change designed to stress cache adaptation.

Trace generation must be deterministic for a fixed configuration and seed. Exported JSON should contain enough configuration and routing data to replay the same selected experts under every policy.

[RouterTrace v2](ROUTER_TRACE_SCHEMA.md) makes the source claim explicit. Synthetic traces name their generator. Captured traces pin immutable model and tokenizer revisions, Transformers and PyTorch versions, capture seed/device/dtype, and either ordered dataset example IDs or a prompt-manifest SHA-256. All provenance participates in the trace fingerprint. A v1 import is replayable but is marked as synthetic because its original source cannot be established retrospectively.

The public experiment surface controls `tokens`, `layers`, `expertsPerLayer`, `topK`, `gpuSlots`, `ramSlots`, `expertSizeMB`, `pcieGBps`, `nvmeGBps`, `computeMsPerToken`, and `seed`. Capacity is expressed in expert slots in the first simulator; transfer cost is derived from the fixed expert size. Invalid combinations such as `topK > expertsPerLayer` or capacities that cannot represent the documented tier behavior should fail validation rather than being silently repaired.

An imported trace should be rejected if expert IDs, layer indexes, token count, or model-shape metadata are inconsistent. Import must not regenerate or reorder expert selections.

## 3. Memory model

The simulator models three ordered tiers:

1. **GPU** — smallest and fastest expert store;
2. **RAM** — larger intermediate store; and
3. **NVMe** — backing store for every expert not resident above it.

On each requested expert:

- a GPU hit moves no expert bytes;
- a RAM hit promotes the exact requested expert to GPU and charges a RAM-to-GPU transfer;
- an NVMe miss moves the exact requested expert through the hierarchy and charges the modeled lower-tier transfers required by the implementation; and
- evictions preserve capacity constraints but never modify the routing trace.

`expertSizeMB`, `pcieGBps`, and `nvmeGBps` use decimal units: one MB is `1,000,000` bytes and one GB/s is `1,000,000,000` bytes per second. They are analytical parameters, not hardware probes.

The current timing model serializes charged links with no compute/I/O overlap. A RAM hit charges one expert across PCIe. An NVMe miss charges one expert across NVMe and then PCIe. A prefetch is charged when issued using the same source-tier rule, even if the prediction is never used; no new prefetch is issued after the final token. `bytes per token` is total link traffic, so an NVMe-to-RAM-to-GPU demand contributes one expert size on each link.

## 4. Baselines

### LRU

Least Recently Used evicts the resident expert whose last access is oldest. It is a standard locality baseline and is used by real offloading projects including Mixtral-Offloading and Colibrì's documented per-layer cache.

### LFU

Least Frequently Used retains experts with the largest accumulated access counts. LFU tests whether stable popularity is enough; it is intentionally vulnerable when historical frequency becomes stale after an abrupt shift.

### ShiftCache

ShiftCache maintains short- and long-window expert-frequency distributions. Let `P` be the normalized recent distribution, `Q` the normalized longer-horizon distribution, and `M = (P + Q) / 2`. It computes Jensen-Shannon divergence:

```text
JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
```

Jensen-Shannon divergence is symmetric and finite for discrete distributions with non-overlapping support; the measure is traced to Jianhua Lin's [1991 paper](https://doi.org/10.1109/18.61115). In this simulator it is a heuristic change signal, not a calibrated statistical test.

As divergence rises, ShiftCache continuously reduces the weight assigned to long-term frequency and increases the influence of recent frequency, recency, and optionally learned one-step expert transitions. A transition score estimates which expert is likely after recently observed experts. Prefetching may promote an exact expert earlier; a wrong prefetch wastes bandwidth or cache capacity but must not change later router selections.

In the original/default mode, a threshold-crossing event is telemetry only and
the JSD score changes retention continuously. The preregistered sanity sweep
adds an opt-in persistent detector and a fixed event-gated intervention without
changing the historical continuous-scoring result.

This combination is a **project hypothesis**, not a claim that distribution-aware caching, recency/frequency hybrids, transition prediction, or workload adaptation are new. MoE-Infinity studies request-level activation patterns and workload changes; DALI studies workload-aware cache replacement; HybriMoE uses score-based caching for unstable activations; and Colibrì documents live LFRU placement.

## 5. Metrics and units

The implementation reports:

| Metric | Interpretation | Claim class |
| --- | --- | --- |
| GPU hit rate | requested experts already in GPU | exact simulator counter |
| RAM hit rate | requested experts found in RAM but not GPU | exact simulator counter |
| NVMe miss rate | requested experts absent from both faster tiers | exact simulator counter |
| bytes per token | modeled expert-weight traffic divided by tokens | exact within the stated transfer model |
| transfer stall per token | transfer bytes divided by configured bandwidths | analytical estimate |
| tokens per second | reciprocal of configured compute time plus modeled transfer stall | analytical estimate |
| evictions | placement removals caused by capacity pressure | exact simulator counter |
| prefetch usefulness | prefetched experts subsequently requested, under the implementation's accounting window | exact simulator counter with policy-specific semantics |
| detected shifts | shared JSD threshold crossings computed from the identical trace for every policy | exact trace diagnostic, not a policy outcome |
| semantic routing changes | selected expert IDs changed by the policy | invariant; expected to be zero |

The estimated throughput follows a simplified structure such as:

```text
transfer_ms = RAM_to_GPU_bytes / PCIe_bytes_per_ms
            + NVMe_bytes / NVMe_bytes_per_ms

estimated_tokens_per_second = 1000 / (compute_ms_per_token + transfer_ms_per_token)
```

Transfers are serialized and additive in this release, including prefetch traffic. Reports must link the commit containing the implementation because later runtime models may add overlap or concurrency. No estimate should be presented as wall-clock latency.

## 6. Fair comparison protocol

For one comparison:

1. Generate or import one trace.
2. Freeze the complete `SimulationConfig` except `policy`.
3. Reset tier state and all policy state before each replay.
4. Replay the identical ordered expert selections under LRU, LFU, and ShiftCache.
5. Confirm equal token, layer, and expert-request counts.
6. Confirm `semanticRoutingChanges === 0` for every policy.
7. Report absolute outcomes and relative differences.

Recommended relative reporting is:

```text
reduction_vs_baseline = (baseline - candidate) / baseline * 100%
```

Report the denominator and guard zero-valued baselines. Do not select only seeds or configurations where ShiftCache wins.

## 7. Fixed benchmark and exploratory sweeps

The repository's fixed benchmark should be treated as a regression fixture: it detects behavioral changes in code, but one seed and one synthetic trace cannot establish generality.

A research-grade sweep should vary at least:

- 30 or more preregistered seeds per synthetic scenario;
- GPU and RAM capacity ratios;
- shift location, abruptness, and overlap between old and new working sets;
- expert size and bandwidth ratios;
- layer count, experts per layer, and top-k;
- short and long windows, divergence threshold, and transition weight; and
- prefetch on/off plus an oracle future-aware upper bound.

For each paired trace, compute policy differences before aggregation. Report medians and uncertainty intervals in addition to means; show failure cases and high-churn regressions.

## 8. Ablations needed for the hypothesis

ShiftCache combines several mechanisms. Its effect cannot be attributed to shift detection without ablation:

- recency plus frequency, no JSD;
- JSD-driven reweighting, no transition score;
- transition score, fixed weights;
- no prefetch;
- fixed threshold versus adaptive or quantile threshold;
- per-layer versus global distributions; and
- true shift location versus detected shift timing.

Detection quality should include delay, false positives, and missed shifts—not only downstream hit rate.

The pinned Switch-Base-8 replay now completes the first four retention cells
with prefetch disabled. JSD-only scoring increased modeled link bytes by 4.31%
relative to fixed frequency/recency scoring, while transition-only scoring
reduced them by 0.52%. The combined score was 2.68% worse than the fixed score,
and every ShiftCache variant remained worse than LFU. These are descriptive
results from one 215-token encoder trace, not estimates of a population effect.

The preregistered detector sweep is now complete. Three-token persistence detected all 30 known
midpoint shifts with a six-token median delay and no stationary false triggers.
However, the registered 64-token intervention increased paired post-shift
modeled link bytes by a median 11.52%, with a paired-bootstrap 95% interval of
[+10.78%, +12.64%]. The traffic gate failed, so the current JSD-gated retention
mechanism is stopped rather than retuned on the evaluated seeds. See
[Preregistered detector sanity sweep](DETECTOR_SANITY.md).

## 9. Path to real-trace validation

The next validation stage is trace replay before full runtime integration:

1. Instrument an open MoE implementation to record router-selected expert IDs without modifying routing.
2. Collect traces from multiple models, prompt domains, decoding lengths, and request-order schedules.
3. Write RouterTrace v2 captured provenance and publish only redistributable trace metadata or anonymized expert IDs; do not publish private prompts.
4. Replay the real traces through every baseline using identical tier configurations.
5. Add DALI-style window scoring, MoE-Infinity-style activation-pattern matching, and Colibrì-style LFRU as stronger baselines where faithfully reproducible.
6. Only then integrate candidate placement logic into a real runtime and measure bytes, cache residency, PCIe/NVMe traffic, TTFT, TPOT, and energy on named hardware.

Real runtime tests must separately report cold-cache and warm-cache behavior, prefill and decode, batch size, quantization format, OS page-cache state, and whether transfers overlap computation.
