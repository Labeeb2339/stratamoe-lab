# Captured Switch-Base-8 router trace

This evidence step records top-1 expert selections produced by an actual,
pinned Mixture-of-Experts checkpoint. It validates StrataMoE's captured-trace
pipeline; it does not turn the simulator into a model runtime or hardware
benchmark.

## What was executed

- Model and tokenizer: [`google/switch-base-8`](https://huggingface.co/google/switch-base-8)
- Immutable revision: `92fe2d22b024d9937146fe097ba3d3a7ba146e1b`
- License reported by the model repository: Apache 2.0
- Software: PyTorch `2.7.1+cpu`, Transformers `4.57.1`
- Device and dtype: CPU, `torch.float32`
- Router selection: `argmax` of raw encoder router logits
- Scope: six sparse encoder feed-forward blocks, numbered `1, 3, 5, 7, 9, 11`
- Workload: ten non-private prompts grouped as security, electronics, science,
  and software; padding tokens excluded and model special tokens retained

The result contains `215` token positions, `6` sparse layers, top-`1` routing,
and `8` experts per layer. All eight expert IDs appear in every captured layer.
The prompt manifest SHA-256, per-prompt token spans, software versions, and
selection hash are recorded beside the canonical RouterTrace v2 file.

## Reproduce the capture

The checkpoint is about 1.24 GB and remains in the Hugging Face cache; model
weights are never copied into this repository.

```powershell
uv venv .capture-venv --python 3.11
uv pip install --python .capture-venv\Scripts\python.exe `
  -r tools\requirements-switch-capture.txt
$env:HF_HUB_DISABLE_XET = "1"
.\.capture-venv\Scripts\python.exe tools\capture_switch_router_trace.py
```

After the first download, an offline repeat is possible:

```powershell
.\.capture-venv\Scripts\python.exe tools\capture_switch_router_trace.py `
  --local-files-only
```

The checked-in capture was repeated twice in one process and then regenerated
in a separate offline process. The separate process produced byte-identical
trace evidence.

## Replay through the simulator

```bash
npm ci
npm run benchmark:captured
```

The fixed replay uses `12` modeled GPU slots, `18` RAM slots, `64` MB experts,
`24` GB/s PCIe, `7` GB/s NVMe, and `6` ms modeled compute per token. Every
policy consumed fingerprint `f3b18fe2` and reported zero semantic routing
changes.

| Policy | Modeled link bytes / token | Modeled transfer stall / token | Prefetches useful / issued |
| --- | ---: | ---: | ---: |
| LFU | 357,209,302.33 B | 26.0288 ms | 0 / 0 |
| LRU | 419,125,581.40 B | 31.6811 ms | 0 / 0 |
| ShiftCache | 516,762,790.70 B | 36.5023 ms | 108 / 366 |

This is a useful negative result. On this captured trace and configuration,
ShiftCache moved **44.67% more modeled bytes than LFU** and **23.30% more than
LRU**. Its transition prefetcher issued 366 prefetches, of which 258 were not
used before eviction or the end of replay. All policies saw the same three JSD
threshold crossings, while the current ShiftCache score continuously reweighted
retention and still did not make the policy competitive here.

The result blocks any broad claim that the existing ShiftCache policy
generalizes beyond its synthetic regression fixture. The following controls
therefore separate prefetch, JSD reweighting, and transition retention before
any threshold is changed.

## Prefetch-disabled control

The first ablation changes exactly one simulator control: ShiftCache transition
prefetch is disabled while its JSD score, frequency/recency reweighting,
transition retention score, trace, and memory configuration remain fixed.

```bash
npm run benchmark:captured:prefetch
```

| Variant | Modeled link bytes / token | Modeled transfer stall / token | GPU hit rate | NVMe miss rate |
| --- | ---: | ---: | ---: | ---: |
| Prefetch on | 516,762,790.70 B | 36.5023 ms | 32.33% | 32.40% |
| Prefetch off | 411,088,372.09 B | 30.7739 ms | 28.06% | 35.12% |

Disabling prefetch reduced modeled link bytes by **20.45%** and modeled
transfer stall by **15.69%**, despite lowering GPU hit rate by 4.26 percentage
points and increasing NVMe miss rate by 2.71 points. It removed 23.424 GB of
prefetch transfers and reduced total modeled traffic by 22.720 GB over the 215
token positions. Both variants detected the same three JSD threshold crossings
and made zero semantic routing changes.

This supports a narrow diagnosis: speculative prefetch traffic polluted this
particular replay. It does not show that prefetch is generally harmful. The
no-prefetch control still moved **15.08% more modeled bytes than LFU**, although
it moved **1.92% fewer than LRU**. Retention reweighting and transition-score
ablations were therefore required. The machine-readable result is
[`prefetch-ablation.json`](../evidence/switch-base-8/prefetch-ablation.json).

## Retention-factor ablation

The second ablation keeps prefetch disabled and crosses two independent
retention controls: continuous JSD-score reweighting and one-step transition
retention.

```bash
npm run benchmark:captured:retention
```

| Variant | Modeled link bytes / token | First 32 post-boundary tokens | Whole run vs fixed |
| --- | ---: | ---: | ---: |
| Fixed frequency/recency | 400,372,093.02 B | 400,666,666.67 B/token | baseline |
| JSD only | 417,637,209.30 B | 422,666,666.67 B/token | +4.31% |
| Transition only | 398,288,372.09 B | 395,333,333.33 B/token | -0.52% |
| JSD + transition | 411,088,372.09 B | 411,333,333.33 B/token | +2.68% |
| LFU reference | 357,209,302.33 B | 349,333,333.33 B/token | — |

The post-boundary aggregate covers the first 32 token positions after the
public prompt groups change at positions `64`, `127`, and `167`. The shared JSD
threshold crossings occurred at `23`, `59`, and `140`; more importantly, the
current threshold event is telemetry only while the JSD score continuously
changes eviction weights.

On this replay, JSD-only scoring was **5.49% worse** than fixed scoring in the
post-boundary windows. Transition-only was the best ShiftCache variant, but its
whole-run traffic remained **11.50% worse than LFU**. This rejects a positive
JSD story for this trace. It does not prove that JSD is generally harmful or
that transition retention generalizes. The machine-readable result is
[`retention-ablation.json`](../evidence/switch-base-8/retention-ablation.json).

The resulting `comparison.json` must be described as:

> Router decisions captured from a real model; memory traffic, transfer stalls,
> and throughput remain simulated.

This small prompt sequence is not enough to establish semantic domain routing,
general cache-policy superiority, model-quality preservation, or hardware
speed. It is the first pipeline-validating trace; stronger work requires more
models, public datasets, preregistered schedules, close baselines, and named
hardware measurements.
