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
threshold crossings, so detecting change did not make the current policy
competitive here.

The result blocks any broad claim that the existing ShiftCache policy
generalizes beyond its synthetic regression fixture. The next experiment must
separate detector, reweighting, and prefetch effects and include a no-prefetch
ablation before changing thresholds.

The resulting `comparison.json` must be described as:

> Router decisions captured from a real model; memory traffic, transfer stalls,
> and throughput remain simulated.

This small prompt sequence is not enough to establish semantic domain routing,
general cache-policy superiority, model-quality preservation, or hardware
speed. It is the first pipeline-validating trace; stronger work requires more
models, public datasets, preregistered schedules, close baselines, and named
hardware measurements.
