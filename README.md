# StrataMoE Lab

**A deterministic, trace-driven simulator for studying expert placement across GPU, RAM, and NVMe under changing Mixture-of-Experts workloads.**

StrataMoE Lab asks a deliberately narrow question:

> Can an explicitly shift-aware cache policy reduce expert bytes streamed per token and modeled transfer stalls after routing-distribution changes, compared with LRU and LFU, without changing the router's selected experts?

The repository implements an inspectable experiment, not a production inference engine. It replays synthetic or imported router traces through a modeled memory hierarchy and compares three policies on the exact same expert selections:

- **LRU** keeps the most recently used experts.
- **LFU** keeps the most frequently used experts over the run.
- **ShiftCache** compares recent and longer-horizon activation distributions with Jensen-Shannon divergence, then shifts weight from historical frequency toward recency and observed one-step transitions when the distributions separate.

Every policy moves the requested expert weights; none may replace, skip, reroute, or silently lower the precision of a selected expert. `semanticRoutingChanges` should therefore remain zero.

## Status and claim boundary

| This repository does | This repository does not do |
| --- | --- |
| Generate deterministic steady-locality, abrupt-shift, and high-churn traces | Load or execute an MoE checkpoint |
| Import and export provenance-bearing router traces as JSON | Measure a physical GPU, PCIe bus, RAM subsystem, or SSD |
| Model GPU/RAM/NVMe residency and transfers | Run CUDA kernels, DMA, decompression, or quantized matrix multiplication |
| Compare policies on identical routing decisions | Establish model-quality preservation on generated text |
| Report exact simulator counters and bandwidth-derived estimates | Claim a real tokens-per-second speedup or a new state of the art |

The honest result format is: **“On trace T under configuration C, policy A reduced simulated bytes per token by X% relative to policy B.”** It is not: “A accelerates MoE inference by X%.” See [Methodology](docs/METHODOLOGY.md) and [Limitations](docs/LIMITATIONS.md).

## Fixed regression benchmark

`npm run benchmark` currently reproduces the synthetic `domain-shift` RouterTrace v2 with seed `2339` and provenance-bearing fingerprint `d860285d` (`240` tokens, `8` layers, `16` experts per layer, top-`2`, `32` GPU slots, `64` RAM slots, `64` MB experts, `24` GB/s PCIe, and `7` GB/s NVMe).

| Policy | Total modeled link bytes / token | Modeled transfer stall / token | Estimated throughput |
| --- | ---: | ---: | ---: |
| ShiftCache | 264,000,000 B | 13.7254 ms | 50.6961 tok/s |
| LRU | 282,666,666.67 B | 14.5032 ms | 48.7729 tok/s |
| LFU | 633,866,666.67 B | 29.4063 ms | 28.2435 tok/s |

On this one checked-in synthetic fixture, ShiftCache moves **6.60% fewer modeled link bytes** and has **5.36% less modeled transfer stall** than LRU. Prefetch traffic is included, and all policies report `semanticRoutingChanges = 0`. This is a deterministic regression result, not evidence of general superiority or measured hardware speed.

## Why this experiment exists

[Colibrì](https://github.com/JustVugg/colibri) demonstrates a real VRAM/RAM/storage hierarchy, per-layer caching, pinned hot experts, and live placement policies while preserving router semantics by default. [MoE-Infinity](https://arxiv.org/abs/2401.14361) studies request-level activation traces, predictive caching, and recovery after task or dataset shifts. [DALI](https://arxiv.org/abs/2602.03495) already proposes workload-aware cache replacement, and Colibrì now documents an LFRU-style live placement policy. Those systems make a broad “first workload-aware MoE cache” claim indefensible here.

StrataMoE Lab instead contributes a small, reproducible surface for isolating one hypothesis: whether **explicit distribution-change detection** helps a non-semantic placement policy recover from abrupt shifts in a controlled trace. The current simulator is an experiment scaffold. Scientific novelty would require real router traces, stronger baselines, hardware measurements, ablations, and statistical replication.

## Quick start

Requires Node.js `>=22.13.0`.

```bash
npm ci
npm run dev
```

Quality and experiment commands are declared in `package.json`:

```bash
npm run check
npm run benchmark
```

The dashboard lets you choose a scenario, seed, token count, model shape, cache capacity, expert size, and modeled bandwidths; run all policies; inspect per-token behavior and final tier residency; and export or import deterministic router traces.

RouterTrace v2 records `source.kind` as either `synthetic` or `captured`. Captured traces must pin immutable model and tokenizer revisions, software versions, capture settings, and either ordered dataset example IDs or the SHA-256 of an external prompt manifest. Provenance is included in the trace fingerprint. Legacy v1 files still import, but are conservatively marked as synthetic because they contain no evidence of how their selections were obtained. See the [RouterTrace v2 schema](docs/ROUTER_TRACE_SCHEMA.md).

## Reproducible comparison checklist

When reporting a result, include:

1. repository commit;
2. trace source, scenario, seed, and token count;
3. all model-shape and memory-tier parameters;
4. short/long window and shift-threshold settings, if configurable;
5. absolute metrics for every policy, not only the best relative percentage;
6. confirmation that every policy replayed the same trace and reported zero semantic routing changes; and
7. an explicit `simulated` or `modeled` label on bytes, stalls, and throughput.

## Research notes

- [Methodology](docs/METHODOLOGY.md) — experiment design, metrics, equations, and validation plan
- [Related work](docs/RELATED_WORK.md) — primary papers and official repositories, including work from Chinese universities and labs
- [Limitations](docs/LIMITATIONS.md) — what the simulator cannot support as a claim
- [Two-minute outreach demo](docs/OUTREACH_DEMO.md) — an honest walkthrough and ask for the 2026-07-21 call

[RouterTrace v2 schema](docs/ROUTER_TRACE_SCHEMA.md) documents synthetic/captured provenance, validation, privacy, and v1 migration.

## Research roadmap

1. **Trace harness:** deterministic synthetic traces, inspection UI, fixed regression benchmark, and honest modeled metrics.
2. **Real-trace replay:** collect router IDs from multiple open MoE families and compare stronger activation-aware baselines.
3. **Runtime validation:** integrate a frozen policy into one open runtime and measure actual traffic, TTFT, TPOT, and energy on named hardware.
4. **Precision study:** only after the placement result is understood, test variable precision with expert-balanced calibration and per-expert quality evidence informed by [MoEQuant](https://proceedings.mlr.press/v267/chen25aa.html), [MxMoE](https://proceedings.mlr.press/v267/duanmu25a.html), and [HOBBIT](https://arxiv.org/abs/2411.01433).

The source code is licensed under the [Apache License 2.0](LICENSE).

StrataMoE Lab is independent educational research. It is not affiliated with Colibrì, AirLLM, the cited authors, their universities, or their organisations.
