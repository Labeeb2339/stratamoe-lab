# Limitations and responsible claims

## Simulation boundary

StrataMoE Lab is a discrete-event-style trace simulator. It does not instantiate a transformer, calculate router logits, execute experts, generate tokens, allocate CUDA memory, or read expert tensors from storage. Its outputs describe the implemented model, not a physical deployment.

Consequently:

- cache hits, misses, evictions, and simulated byte counters are exact only with respect to the replay logic;
- transfer stalls are estimates derived from configured bandwidths;
- tokens per second is a projection derived from configured compute time and modeled stalls; and
- zero semantic routing changes means the simulator did not edit expert IDs in the trace. It does not prove byte-identical logits or generated text.

## Missing hardware effects

The current abstraction omits or simplifies:

- PCIe protocol overhead, DMA setup, pinned versus pageable memory, and topology;
- NVMe queue depth, random versus sequential access, filesystem behavior, OS page cache, and read amplification;
- overlapping I/O, CPU work, GPU kernels, and multiple transfers;
- kernel launch, dequantization, reconstruction, and synchronization costs;
- NUMA placement, memory bandwidth contention, thermal throttling, and other processes;
- batching, continuous scheduling, request queues, prefill/decode differences, and KV-cache pressure; and
- variable expert sizes, shared experts, dense layers, and model-specific routing rules.

Because some effects overlap and others add contention, multiplying configured bandwidth by bytes is not a conservative upper or lower bound in general.

## Synthetic-trace limitations

The built-in scenarios encode assumptions chosen by the generator. A policy can perform well because it matches those assumptions.

- `steady` tests a stable working set, not every stationary workload.
- `domain-shift` inserts a synthetic distribution change; it does not demonstrate a semantic domain boundary in a trained model.
- `high-churn` is one kind of weak locality, not a complete adversarial workload.
- pseudorandom seeds provide repeatability, not representativeness.

The fixed benchmark is a regression test. It is not a dataset, leader board, or statistical study.

## Policy limitations

ShiftCache has hyperparameters and multiple coupled mechanisms. Without preregistered sweeps and ablations, a favorable result may be threshold tuning, window tuning, or transition-prefetch behavior rather than useful shift detection.

Jensen-Shannon divergence on short empirical windows is noisy and sample-size dependent. A fixed threshold has no universal false-positive guarantee. The detector compares activation histograms; it can miss changes in ordering that preserve marginal frequencies, and it can signal harmless marginal changes.

Transition counts can overfit recent paths. Prefetching can consume bandwidth, evict useful experts, or arrive too late. The simulator's usefulness metric should not be equated with latency hidden on real hardware.

LRU and LFU are necessary but weak baselines. Close prior work includes DALI's workload window, MoE-Infinity's activation-pattern matching, HybriMoE's score-based cache, Colibrì's LFRU-style live placement, and learned predictors such as ProMoE and ExpertFlow. A paper-quality claim requires stronger baselines.

## Precision and quality

The first release assumes a fixed expert size and does not silently lower precision. This avoids conflating placement with quantization but also means the simulator cannot answer quantization questions.

Future mixed-precision work must measure at least:

- bytes and latency by tier and bit width;
- dequantization and kernel costs;
- router decision stability if router computation changes;
- logit or hidden-state error;
- task accuracy and generation quality; and
- reproducibility differences across kernel shapes and devices.

Using the same expert ID at a lower precision is not automatically semantically equivalent.

[MoEQuant](https://proceedings.mlr.press/v267/chen25aa.html) also shows that MoE post-training quantization can suffer from uneven calibration coverage across experts and unequal token-expert affinity. A future precision policy must not infer “safe to quantize” from cache popularity alone. It should use expert-balanced calibration, report per-expert error or sensitivity, and validate both common and rarely activated experts. Mixed-precision systems such as [HOBBIT](https://arxiv.org/abs/2411.01433) and [MxMoE](https://proceedings.mlr.press/v267/duanmu25a.html) are additional real baselines; neither is implemented in this release.

## Security and privacy

Imported traces should contain expert IDs and minimal metadata, not prompts, generated private text, credentials, or proprietary model artifacts. Router traces may still reveal workload structure, so real-trace releases need a privacy review and clear dataset/model licenses.

The simulator is not a security, compliance, or capacity-planning tool.

## Claims that are currently supportable

With a linked commit and complete configuration, the repository may support statements such as:

> On this deterministic simulated trace, ShiftCache produced fewer modeled lower-tier bytes per token than LRU and LFU while replaying identical expert IDs.

> The fixed benchmark and automated tests reproduced the checked-in policy behavior.

> Under the simulator's bandwidth model, the candidate policy had a lower estimated transfer stall.

## Claims that are not currently supportable

Do not claim that StrataMoE:

- runs or accelerates any named model;
- achieves a real hardware tokens-per-second figure;
- is faster than Colibrì, MoE-Infinity, DALI, Fiddler, KTransformers, or another runtime;
- invented workload-aware, domain-aware, recency/frequency, predictive, or multi-tier expert caching;
- preserves model accuracy, output quality, or exact logits;
- generalizes across MoE architectures or real prompt domains; or
- is a thesis-level novel algorithm solely because the simulator produces a favorable result.

## Evidence needed for stronger claims

1. **Real router traces:** several open MoE model families, datasets, domains, seeds, and request orders.
2. **Faithful baselines:** close cache and predictor methods, not only LRU/LFU.
3. **Ablation:** detector, weighting, transitions, and prefetch measured separately.
4. **Statistics:** paired per-trace comparisons, uncertainty intervals, sensitivity analysis, and disclosed failures.
5. **Runtime integration:** a named open engine and pinned commit with no routing edits.
6. **Hardware measurements:** named CPU/GPU/storage, cache state, batch size, TTFT, TPOT, traffic, and energy where possible.
7. **Quality checks:** output or task metrics whenever precision, execution order, or routing changes.
8. **Reproduction:** scripts, environment, raw aggregate results, and a machine-readable experiment manifest.

Until those steps are completed, StrataMoE Lab should be presented as an honest, inspectable research harness and an invitation to test the hypothesis.
