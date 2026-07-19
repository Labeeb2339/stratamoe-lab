# ShiftQ-MoE research plan

**Working question:** Does router-distribution change detection improve
mixed-precision expert residency under sequential workload shifts?

**Status on 2026-07-19:** candidate research direction, not a claimed new
algorithm, quantizer, thesis result, or performance improvement.

## Recommendation

Do not begin by inventing another 2-bit number format. Numerical quantization
is already crowded and a credible contribution would require model-quality
evaluation plus serious kernel work.

The student-scale, falsifiable direction is instead:

> **ShiftQ-MoE:** change-triggered, risk- and migration-aware mixed-precision
> expert residency using existing documented quantizers.

The narrow hypothesis is:

> Under sequential workload shifts, explicit change detection followed by
> precision allocation using measured expert damage per migrated byte reduces
> post-shift quality loss relative to hotness-only mixed-precision policies at
> the same memory and migration budgets.

This is a quantization-control method. It does not claim a new integer or
vector representation.

## Why the current captured result matters

StrataMoE's first pinned Switch-Base-8 trace is a failure case for the current
combined ShiftCache policy. ShiftCache moved 44.67% more modeled bytes than LFU
because 258 of 366 prefetches were unused. That result prevents a weak story in
which JSD detection is assumed to help merely because it wins on one synthetic
fixture.

Before any precision work, the captured-trace harness must separate:

1. change detection;
2. recency/frequency reweighting; and
3. transition prefetching.

ShiftQ-MoE should initially disable transition prefetching. Precision migration
must have its own explicit byte budget and hysteresis.

## Closest prior work and the remaining candidate gap

| Work | What it already covers | What may remain to test |
| --- | --- | --- |
| [DynaExq](https://arxiv.org/abs/2511.15015) | Runtime high/low-precision expert versions, EMA hotness, memory budgets, hysteresis, and asynchronous movement | Controlled sequential change-point experiments combining explicit JSD detection with calibrated expert damage and migration cost are not a reported focus |
| [DyMoE](https://arxiv.org/abs/2603.19172) (Sun Yat-sen University and Tencent authors) | Dynamic importance, mixed precision, look-ahead prefetching, and edge inference | Cross-request change detection plus an expert-specific damage table may be separable |
| [HOBBIT](https://arxiv.org/abs/2411.01433) (Shanghai Jiao Tong University and CUHK authors) | Token-level mixed-precision loading, adaptive prefetching, and multidimensional caching | No explicit sequential workload change-point experiment is claimed here |
| [MxMoE](https://arxiv.org/abs/2505.05799) | Calibration perturbation, activation frequency, hardware cost, and mixed-precision allocation | Strong static baseline rather than an online shift-triggered policy |
| [FRI-MxMoE](https://aclanthology.org/2026.acl-long.982/) (Xiamen University authors) | Fast quantization-error and runtime prediction followed by mixed-precision allocation | Static calibration rather than online sequential adaptation |
| [MoPEQ](https://arxiv.org/abs/2509.02512) | Per-expert Hessian sensitivity combined with activation frequency | Already rules out novelty based only on sensitivity times frequency; remains a static baseline |
| [HCRMap](https://arxiv.org/abs/2607.11586) (Nanjing University authors) | Hotness drift, multi-tier residency, migration cost, and hysteresis | A serious novelty threat to generic migration-aware scheduling; the possible distinction is measured bit-level quality damage |
| [KBVQ-MoE](https://arxiv.org/abs/2602.11184) | A genuine shared/expert vector quantizer with bias correction | Shows why creating a competitive new numerical quantizer is outside the first experiment's scope |

The possible gap is only the intersection of all four items:

1. explicit sequential router-distribution change detection;
2. measured per-expert quantization damage rather than frequency alone;
3. migration-aware precision reallocation; and
4. controlled evaluation on real router traces.

This gap is provisional. Literature must be searched again immediately before
any public novelty statement, especially because HCRMap appeared in July 2026.

## Proposed method

### Offline calibration

For layer `l`, expert `e`, and bit width `b`:

1. collect balanced routed activations from a held-out calibration set;
2. quantize a copy of the expert to 2, 4, or 8 bits using a documented baseline;
3. compare FP16 and quantized expert outputs; and
4. store an uncertainty-adjusted damage estimate.

```text
R_plus[l,e,b] = mean_damage[l,e,b] + kappa * standard_error[l,e,b]
```

The upper bound prevents a rarely sampled expert from looking safe merely
because it has too little evidence.

The first implementation should record normalized expert-output error and
teacher-logit KL. It must not call either one downstream accuracy.

### Online detection

For each layer, maintain short- and long-window router distributions. Compute
Jensen-Shannon divergence and aggregate layer scores using a preregistered
median or 90th percentile.

A change fires only when:

- the score exceeds a stationary-calibrated threshold;
- the exceedance persists for several windows; and
- a cooldown has expired.

### Precision allocation

At a trigger, estimate the new expert probabilities and solve a multiple-choice
knapsack:

```text
minimize sum(p_hat[l,e] * R_plus[l,e,b] * x[l,e,b])
       + lambda_m * migration_bytes[l,e,old_b->b] * x[l,e,b]
       + lambda_c * p_hat[l,e] * measured_cost[l,e,b] * x[l,e,b]
```

Subject to:

- exactly one precision per expert;
- total resident bytes within the memory budget;
- migrated bytes per interval within a migration budget; and
- insufficiently calibrated experts staying at 8 bits or above.

The first prototype should omit the measured-cost term and make only
quality-proxy, memory, and migration claims. Latency enters only after real
hardware measurements exist.

```text
OFFLINE
for each layer, expert and bit width:
    collect balanced routed activations
    quantize an expert copy
    measure FP16-versus-quantized damage
    store damage upper bound and represented bytes

ONLINE, each routing window
    update short and long expert distributions
    drift = aggregate_JSD(short, long)

    if threshold, persistence or cooldown conditions fail:
        keep the current precision plan
        continue

    estimate new expert probabilities with short/long shrinkage
    solve damage plus migration cost under the memory budget

    if improvement exceeds hysteresis:
        apply a bounded number of promotions and demotions
        log trigger, old plan, new plan, migrated bytes and quality proxy
```

## Models and real traces

The current Switch-Base-8 capture validates the provenance and replay pipeline,
but it is not a causal language-model quality experiment.

The first causal-MoE target should be
[`allenai/OLMoE-1B-7B-0924-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct),
supported by the [OLMoE paper](https://arxiv.org/abs/2409.02060). Use
[`Qwen/Qwen1.5-MoE-A2.7B-Chat`](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)
as a second architecture only if suitable GPU or cloud access exists.

Preregister sequential schedules such as:

- general text -> mathematics -> code;
- code -> text -> mathematics;
- A -> B -> A recurrence;
- gradual mixtures rather than only abrupt boundaries;
- stationary controls; and
- deliberately high-churn controls.

Record request/token index, prefill/decode phase, layer, selected experts,
router scores, top-k margins, model revisions, and a prompt-manifest hash. Do
not publish private prompts.

Fixed-trace replay tests the scheduler under controlled routing. It cannot
prove end-to-end quality because quantization can change later hidden states and
future router decisions. Any model-quality claim needs a closed-loop quantized
model run.

## Baselines

1. uniform static 8-bit, 4-bit, and 2-bit;
2. static activation-frequency allocation;
3. static sensitivity-aware allocation resembling MxMoE or MoPEQ;
4. DynaExq-style EMA hotness, top-n, and hysteresis;
5. HOBBIT-style importance thresholding; and
6. an offline oracle plan for each known segment.

## Required ablations

- no JSD: optimize every window;
- JSD plus frequency only;
- risk-aware allocation without migration cost;
- no uncertainty protection;
- no persistence or cooldown;
- fixed versus stationary-calibrated thresholds; and
- transition prefetch disabled versus enabled.

## Metrics

- detection delay, missed shifts, and false triggers per 10,000 tokens;
- teacher-logit KL, normalized expert/block error, perplexity, and task accuracy;
- top-k routing Jaccard and boundary-flip rate;
- resident bytes, average bits, migration bytes/events, and bytes per token;
- post-shift quality-regret area over the first 256 tokens; and
- TTFT, token latency, and tail latency only on named hardware.

## Strict go/no-go gates

### Detector gate

Proceed only if real OLMoE traces satisfy every item:

- at least two of three preregistered domain transitions exceed the stationary
  99th-percentile JSD threshold;
- at most one false trigger per 10,000 tokens; and
- median detection delay of at most 64 tokens.

If this fails, stop the shift-triggered thesis rather than tuning until the
plots look favorable.

### Research-claim gate

At the same memory budget, across two causal-MoE families and at least three
preregistered schedules, ShiftQ-MoE must:

- reduce post-shift cumulative teacher-KL over 256 tokens by at least 10%
  against the best dynamic baseline;
- have a paired bootstrap 95% confidence interval excluding zero;
- use no more migration bytes than that baseline; and
- keep downstream accuracy within 0.5 percentage points, or perplexity within
  1% relative, of the strongest budget-matched alternative.

Failure means it remains an engineering experiment, not a new research method.
A synthetic StrataMoE result alone can never pass this gate.

## Immediate next experiment

Do not implement mixed precision yet. The first captured-trace control now
disables transition prefetch while keeping the rest of ShiftCache fixed. It
reduced modeled link bytes by 20.45% on the pinned Switch-Base-8 trace, but the
control remained 15.08% worse than LFU. This diagnoses prefetch pollution on
one replay; it does not validate the detector or establish generality.

The evidence bundle now contains LRU, LFU, the current combined ShiftCache
policy, and the prefetch-disabled ShiftCache control. Continue the matrix with:

1. recency/frequency reweighting without JSD;
2. JSD reweighting without the transition retention score; and
3. transition retention scoring without JSD-driven weights.

The purpose is to identify whether the first captured failure came from the
detector, the reweighting rule, or prefetch pollution. Only a stable detector
and allocation signal should become the foundation for ShiftQ-MoE.

The strongest honest working title is:

> **ShiftQ-MoE: Does Router-Distribution Change Detection Improve
> Mixed-Precision Expert Residency?**
