# Related work and novelty boundary

This review uses primary papers and official project repositories. It records what informed StrataMoE Lab and, equally importantly, what prevents an inflated novelty claim. Sources were checked on 2026-07-19; arXiv preprints and active repositories may change, so any submission should pin the reviewed versions and rerun the search.

## Closest systems

### Colibrì

[JustVugg/colibri](https://github.com/JustVugg/colibri) is a real, pure-C MoE runtime. Its documentation describes VRAM, RAM, and storage as one managed hierarchy; routed experts streamed from disk; per-layer LRU caching; an optional pinned hot store; OS page cache; asynchronous expert readahead; and experimental router-lookahead prefetch. Its current documentation also describes a lossless live placement policy with an LFRU score combining decaying session frequency and recent access.

**Relation to StrataMoE:** Colibrì is the direct systems inspiration for visualizing exact expert weights moving through multiple tiers while retaining router semantics. StrataMoE is not a fork and does not run Colibrì or GLM-5.2. LRU, frequency/recency mixtures, multi-tier placement, and semantics-preserving movement are not novel claims here.

### MoE-Infinity

[MoE-Infinity](https://arxiv.org/abs/2401.14361) and its [official repository](https://github.com/EfficientMoE/MoE-Infinity) use request-level Expert Activation Matrices to inform expert caching and prefetching. The paper reports that activation reuse can be sparse within a request yet become much more uniform across requests. It explicitly evaluates task and dataset workload changes and measures requests needed to recover low latency. The repository warns that its redesigned open-source version differs from the performance-first paper implementation, so paper results should not be attributed automatically to the current code.

**Relation to StrataMoE:** historical activation traces, transition/pattern prediction, and adaptation after workload changes all have clear prior art. StrataMoE's narrower hypothesis is whether a transparent short-versus-long distribution divergence signal is useful in a controlled trace harness.

### DALI

[DALI](https://arxiv.org/abs/2602.03495), from authors at the Chinese Academy of Sciences, University of Chinese Academy of Sciences, Shanghai Jiao Tong University, and collaborating organisations listed in the paper, is a workload-aware offloading framework for local PCs. It combines dynamic CPU/GPU assignment, residual-based prefetching, and a windowed Workload-Aware Cache Replacement strategy based on recent expert workloads. The paper compares its replacement policy with LRU and describes adaptation to the current sequence.

**Relation to StrataMoE:** DALI makes any broad claim of being the first workload-aware or domain-adaptive expert cache untenable. A future evaluation must compare ShiftCache against a faithful DALI-style window score, not only LRU and LFU.

### HybriMoE

[HybriMoE](https://arxiv.org/abs/2504.05897), from Peking University and Beihang University authors, combines dynamic intra-layer CPU/GPU scheduling, impact-driven inter-layer prefetching, and score-based cache management for unstable expert activations. Its [official repository](https://github.com/PKU-SEC-Lab/HybriMoE) is built on KTransformers.

**Relation to StrataMoE:** dynamic activation-aware scheduling and score-based caching are prior art. HybriMoE also shows that placement, scheduling, and compute assignment interact in a real system—interactions the current simulator does not model.

## Offloading and heterogeneous execution

### Fiddler

[Fiddler](https://arxiv.org/abs/2402.07033) ([official code](https://github.com/efeslab/fiddler)) avoids some CPU-to-GPU weight movement by executing cache-miss experts on the CPU, while placing attention and offline-profiled popular experts on the GPU. The paper evaluates real Mixtral inference and includes dataset-sensitivity analysis.

**Relation to StrataMoE:** Fiddler demonstrates that moving weights is not the only response to a miss. Its offline popularity placement is also an important static baseline for real traces. StrataMoE's current transfer-only model cannot compare CPU execution with promotion to GPU, and its synthetic shifts do not resolve how stable expert popularity is across real domains.

### KTransformers

[KTransformers](https://doi.org/10.1145/3731569.3764843) and its [official repository](https://github.com/kvcache-ai/ktransformers) come from Tsinghua University's MADSys Lab, Approaching.AI, and the other institutions listed in the paper. The SOSP 2025 system uses optimized CPU kernels and asynchronous CPU/GPU scheduling for heterogeneous MoE inference; it also studies Expert Deferral.

**Relation to StrataMoE:** KTransformers is a real heterogeneous execution system, not merely a cache. Its kernel, synchronization, NUMA, and CPU-compute effects sit outside StrataMoE's bandwidth-derived model.

### AirLLM

[AirLLM](https://github.com/lyogavin/airllm) decomposes models into layer-wise shards and loads one layer at a time, with optional prefetching and weight-only blockwise compression. It targets minimal VRAM rather than MoE-expert cache adaptation specifically.

**Relation to StrataMoE:** AirLLM motivates disk-backed inference but uses a different unit of movement and optimization target. StrataMoE should not borrow AirLLM's model-size or VRAM claims.

### MoE-Lightning

[MoE-Lightning](https://arxiv.org/abs/2411.11217) uses CPU-GPU-I/O pipelining, paged weights, and a hierarchical roofline model for throughput-oriented batch inference on memory-constrained GPUs.

**Relation to StrataMoE:** serialized bandwidth arithmetic is an intentionally simpler abstraction. A real system must model overlap and resource contention rather than treating all transfer time as additive.

## Caching and prefetching

### ProMoE

[ProMoE](https://arxiv.org/abs/2410.22134) and its [official repository](https://github.com/promoe-opensource/promoe) are from Shanghai Jiao Tong University's Institute of Parallel and Distributed Systems, the CAS Key Laboratory of System Software, and Zhejiang University authors. ProMoE learns correlations between intermediate states and future expert selections, then coordinates chunked prefetching, early preemption, and reordered inference to move misses off the critical path.

**Relation to StrataMoE:** one-step transition counts are a much simpler predictor and should not be represented as competitive with a learned cross-layer predictor. Prefetch usefulness alone also cannot estimate hidden latency without a concurrency model.

### ExpertFlow

[ExpertFlow](https://arxiv.org/abs/2410.17954) uses a learned routing-path predictor, token scheduling, and a predictive expert cache with runtime correction.

**Relation to StrataMoE:** route prediction and predictive caching predate this project. Unlike ExpertFlow's token scheduling, StrataMoE keeps the trace order fixed so that placement policies see the same request sequence.

### LFU and speculative prefetch analysis

[In-depth Analysis on Caching and Pre-fetching in Mixture of Experts Offloading](https://arxiv.org/abs/2511.05814) analyzes LRU behavior, proposes LFU as an improvement in its experiments, and studies speculative expert prefetching.

**Relation to StrataMoE:** LFU is a literature baseline, not an invention. The paper also motivates reporting traces and failure modes rather than only aggregate hit rates.

### Proactive caching and buffering

[Toward Efficient Inference for Mixture of Experts](https://papers.nips.cc/paper_files/paper/2024/hash/98bf3b8505c611ac21055dd9d355c66e-Abstract-Conference.html) characterizes language-modeling and machine-translation MoE workloads and proposes dynamic gating, Expert Buffering, and load balancing.

**Relation to StrataMoE:** keeping hot experts in GPU memory while buffering others in CPU memory is established. That work may change gating, whereas StrataMoE's placement policies are constrained not to change selected experts.

## Quantization and semantic boundaries

### HOBBIT

[HOBBIT](https://arxiv.org/abs/2411.01433), from Shanghai Jiao Tong University and the Chinese University of Hong Kong authors, combines token-level mixed-precision loading, adaptive prefetching, and multidimensional expert caching. It can substitute lower-precision forms for selected cache-miss experts based on estimated importance.

**Relation to StrataMoE:** mixed-precision transfer is meaningful prior art, but it crosses the current project's semantic boundary. StrataMoE moves the exact represented expert and models a fixed expert size; it does not evaluate quantization error or model quality. Adaptive precision belongs in a future experiment with output-quality and router-stability measurements.

### MoEQuant

[MoEQuant](https://proceedings.mlr.press/v267/chen25aa.html), an ICML 2025 paper from authors affiliated with Houmo AI and Southeast University, shows why ordinary post-training quantization calibration can be biased for MoE models. It identifies imbalance in how calibration samples reach different experts and differences in token-expert affinity, then proposes expert-balanced self-sampling and affinity-guided quantization.

**Relation to StrataMoE:** a future precision-aware trace study cannot assign bit widths from access frequency alone. Calibration coverage and token-expert affinity affect quality, so any bytes-versus-quality policy needs expert-balanced calibration and per-expert error evidence. MoEQuant is prior art for that future direction; it is not implemented here.

### MxMoE

[MxMoE](https://proceedings.mlr.press/v267/duanmu25a.html), an ICML 2025 paper with authors from Shanghai Jiao Tong University, Shanghai AI Laboratory, Peking University, ByteDance Seed, and the Chinese University of Hong Kong, co-designs mixed-precision allocation and Group-GEMM execution. It accounts for linear-block sensitivity, expert activation frequency, and hardware characteristics rather than assuming one bit width is best everywhere.

**Relation to StrataMoE:** if precision becomes another placement dimension, the simulator will need variable expert/block sizes and measured kernel profiles, while the evaluation must preserve quality. A simple “colder expert gets fewer bits” rule would ignore MxMoE's sensitivity and hardware findings and would not be a defensible new algorithm.

Colibrì also implements int8, packed int4, and packed int2 kernels and documents real numerical caveats. Those details reinforce why “same expert ID” alone is insufficient to prove identical generated output when precision or kernels change.

## Distribution changes

Jianhua Lin's [“Divergence measures based on the Shannon entropy”](https://doi.org/10.1109/18.61115) provides the information-theoretic basis for Jensen-Shannon divergence. Using divergence between two empirical windows as a change signal is a heuristic application, not a new divergence measure. Work outside MoE has also used double-window JSD for drift detection; for example, [Yang et al.](https://doi.org/10.1145/3573942.3573979) apply it to network-traffic concept drift.

Within MoE inference, MoE-Infinity already evaluates task/dataset shifts, DALI explicitly targets workload dynamics, and HybriMoE responds to activation instability. Therefore the possible research contribution is not “MoE workloads change.” It is a reproducible evaluation of one explicit detector-policy coupling under controlled and, later, real routing traces.

## What may still be worth testing

The following are hypotheses, not established contributions:

1. Short-versus-long JSD may identify an abrupt working-set change early enough to prevent stale LFU counts from dominating.
2. A detector may reduce unnecessary adaptation during steady locality compared with always-recency-heavy policies.
3. One-step transitions may help only in traces with stable sequential structure and may harm high-churn traces through pollution.
4. Detection delay and false positives may explain bytes-per-token outcomes better than aggregate JSD alone.

## Novelty statement suitable for this repository

> StrataMoE Lab implements and evaluates a transparent JSD-triggered cache-policy variant in a deterministic three-tier trace simulator. It is a reproducible hypothesis test, not a claim of first workload-aware MoE caching or a validated inference-speed improvement.

Any stronger novelty statement requires a systematic search current to the submission date, faithful implementation of close baselines, real traces from multiple MoE families, and hardware validation.

## Affiliation note

Chinese institutions are named here because their directly relevant primary work was reviewed, not as a quality ranking. StrataMoE Lab is not affiliated with any cited author, laboratory, university, company, or open-source project.

## Primary source index

| Work | Primary paper or official repository | Status used here |
| --- | --- | --- |
| Colibrì | [official repository](https://github.com/JustVugg/colibri) | active repository, checked 2026-07-19 |
| MoE-Infinity | [arXiv:2401.14361](https://arxiv.org/abs/2401.14361), [official repository](https://github.com/EfficientMoE/MoE-Infinity) | paper and current code |
| DALI | [arXiv:2602.03495](https://arxiv.org/abs/2602.03495) | v1 preprint, 2026-02-03 |
| HybriMoE | [arXiv:2504.05897](https://arxiv.org/abs/2504.05897), [official repository](https://github.com/PKU-SEC-Lab/HybriMoE) | v1 preprint and DAC'25 code |
| Fiddler | [arXiv:2402.07033](https://arxiv.org/abs/2402.07033), [official repository](https://github.com/efeslab/fiddler) | ICLR 2025 paper and code |
| KTransformers | [SOSP 2025 paper](https://doi.org/10.1145/3731569.3764843), [official repository](https://github.com/kvcache-ai/ktransformers) | peer-reviewed paper and active code |
| AirLLM | [official repository](https://github.com/lyogavin/airllm) | active software project |
| MoE-Lightning | [arXiv:2411.11217](https://arxiv.org/abs/2411.11217) | preprint |
| ProMoE | [arXiv:2410.22134](https://arxiv.org/abs/2410.22134), [official repository](https://github.com/promoe-opensource/promoe) | preprint and code |
| ExpertFlow | [arXiv:2410.17954](https://arxiv.org/abs/2410.17954) | v2, accepted at DAC 2026 |
| Caching and prefetching analysis | [arXiv:2511.05814](https://arxiv.org/abs/2511.05814) | v1 preprint |
| Toward Efficient Inference for MoE | [NeurIPS 2024 proceedings](https://papers.nips.cc/paper_files/paper/2024/hash/98bf3b8505c611ac21055dd9d355c66e-Abstract-Conference.html) | peer-reviewed paper |
| HOBBIT | [arXiv:2411.01433](https://arxiv.org/abs/2411.01433) | v2 preprint |
| MoEQuant | [ICML 2025 proceedings](https://proceedings.mlr.press/v267/chen25aa.html) | peer-reviewed paper |
| MxMoE | [ICML 2025 proceedings](https://proceedings.mlr.press/v267/duanmu25a.html), [official repository](https://github.com/cat538/MxMoE) | peer-reviewed paper and code |
| Jensen-Shannon divergence | [Lin, IEEE Transactions on Information Theory, 1991](https://doi.org/10.1109/18.61115) | peer-reviewed paper |
| JSD drift detector example | [Yang et al., AIPR 2022](https://doi.org/10.1145/3573942.3573979) | peer-reviewed proceedings paper |
