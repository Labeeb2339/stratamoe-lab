# ShiftQ-MoE research plan

**Working question:** Does router-distribution change detection improve
mixed-precision expert residency under sequential workload shifts?

**Status on 2026-07-19:** current JSD-gated design stopped after a
preregistered traffic gate failed; not a claimed new algorithm, quantizer,
thesis result, or performance improvement.

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
combined ShiftCache policy. ShiftCache moved 44.67% more modeled bytes than LFU,
and 258 of 366 issued prefetches were unused. The no-prefetch ablation identified
substantial prefetch pollution, although the remaining policy still lost to
LFU. That result prevents a weak story in which JSD scoring is assumed to help
merely because it wins on one synthetic fixture.

Before any precision work, the captured-trace harness needed to separate:

1. JSD-score reweighting;
2. recency/frequency reweighting; and
3. transition prefetching.

That first mechanism matrix is now complete. It found prefetch pollution and a
negative JSD-score result on this trace; it did not validate change detection.

Any separate future precision study should initially disable transition
prefetching. Precision migration must have its own explicit byte budget and
hysteresis.

## Closest prior work and the remaining candidate gap

| Work | What it already covers | What may remain to test |
| --- | --- | --- |
| [DynaExq](https://arxiv.org/abs/2511.15015) | Runtime high/low-precision expert versions, EMA hotness, memory budgets, hysteresis, and asynchronous movement | Controlled sequential change-point experiments combining explicit JSD detection with calibrated expert damage and migration cost are not a reported focus |
| [DyMoE](https://arxiv.org/abs/2603.19172) (Sun Yat-sen University and Tencent authors) | Dynamic importance, mixed precision, look-ahead prefetching, and edge inference | Cross-request change detection plus an expert-specific damage table may be separable |
| [HOBBIT](https://arxiv.org/abs/2411.01433) (Shanghai Jiao Tong University and CUHK authors) | Token-level mixed-precision loading, adaptive prefetching, and multidimensional caching | No explicit sequential workload change-point experiment is claimed here |
| [MxMoE](https://arxiv.org/abs/2505.05799) | Calibration perturbation, activation frequency, hardware cost, and mixed-precision allocation | Strong static baseline rather than an online shift-triggered policy |
| [FRI-MxMoE](https://aclanthology.org/2026.acl-long.982/) (ACL 2026) | Fast quantization-error and runtime prediction followed by mixed-precision allocation | Static calibration rather than online sequential adaptation |
| [MoPEQ](https://arxiv.org/abs/2509.02512) | Per-expert Hessian sensitivity combined with activation frequency | Already rules out novelty based only on sensitivity times frequency; remains a static baseline |
| [HCRMap](https://arxiv.org/abs/2607.11586) (Yongqin Zhang, Nanjing Vocational College of Information Technology; arXiv v1 preprint) | Pressure-aware hot-expert replica residency across stacked SRAM, local HBM, and shared DRAM, accounting for migration overhead, runtime pressure, minimum residency, and hysteresis | A serious novelty threat to generic multi-tier migration-aware scheduling; it does not evaluate bit-level quantization damage |
| [KBVQ-MoE](https://arxiv.org/abs/2602.11184) | A genuine shared/expert vector quantizer with bias correction | Shows why creating a competitive new numerical quantizer is outside the first experiment's scope |

The possible gap is only the intersection of all four items:

1. explicit sequential router-distribution change detection;
2. measured per-expert quantization damage rather than frequency alone;
3. migration-aware precision reallocation; and
4. controlled evaluation on real router traces.

This gap is provisional. Literature must be searched again immediately before
any public novelty statement, especially because HCRMap v1 was submitted on 13
July 2026.

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

### Detector gate for any materially different future design

Proceed only if a materially different detector is preregistered on fresh data
and real OLMoE traces satisfy every item:

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

## Completed go/no-go experiment and decision

Do not implement mixed precision yet. The pinned Switch-Base-8 mechanism matrix
is now complete with prefetch disabled:

| Retention score | Whole-run modeled bytes vs fixed | First 32 post-boundary tokens vs fixed |
| --- | ---: | ---: |
| Fixed frequency/recency | baseline | baseline |
| JSD only | +4.31% | +5.49% |
| Transition only | -0.52% | -1.33% |
| JSD plus transition | +2.68% | +2.66% |

Transition-only was the best ShiftCache variant but remained 11.50% worse than
LFU over the full replay. The JSD threshold crossings were at token positions
23, 59, and 140, while the public semantic-group boundaries were at 64, 127,
and 167. This is negative evidence for the current continuous JSD reweighting
rule on one short encoder trace.

That checkpoint also exposed a terminology boundary: its threshold crossing
was telemetry only while the eviction weights changed continuously with the JSD
score. The following preregistered control added a separate event-gated mode.

The next preregistered control was therefore a detector sanity experiment, not
quantization. It used seeds `2300` through `2329`, `1024` tokens, and otherwise
the fixed default synthetic configuration for both stationary and abrupt-shift
traces. The public plan required persistence and cooldown before observing the
sweep; the implementation then froze three-token persistence, a 64-token
cooldown, and a fixed 64-token intervention before its first run. Those exact
numeric implementation values were not in the earlier public commit. JSD gating
could continue only if all of these provisional engineering gates passed:

1. at least 27 of 30 abrupt shifts trigger within 64 tokens of the known change;
2. stationary traces produce at most one false trigger per 10,000 tokens;
3. the median paired first-64-token post-shift modeled-byte reduction versus
   fixed frequency/recency is at least 5%, with a paired-bootstrap 95% interval
   below zero; and
4. stationary whole-run modeled bytes regress by no more than 2%.

The detector passed the timing controls: 30 of 30 shifts triggered after six
tokens, with zero stationary false triggers. The intervention failed the
traffic gate decisively. It increased first-64-token post-shift modeled link
bytes by a median **11.52%**, with a paired-bootstrap 95% interval of
**[+10.78%, +12.64%]**. Stationary traffic was unchanged.

The registered outcome is `carryForward = false`. The current JSD-gated
retention design stops here and must not be tuned on the same seeds. These
synthetic gates can reject a broken mechanism but could never pass the real
OLMoE detector gate or support a novelty claim.

Do not implement the proposed ShiftQ-MoE mixed-precision layer on top of this
failed signal. A future quantization study should start separately with a static
expert-sensitivity baseline, or preregister a materially different detector on
new out-of-sample traces. It must not reuse the ShiftQ-MoE name as if this gate
had passed.

The strongest honest title for the completed placement experiment is:

> **When Change Detection Is Not Enough: A Negative Control for JSD-Gated MoE
> Expert Placement**

## Candidate continuation: shift actionability study

This is a proposal, not a result and not yet a new benchmark. The next question
is narrower than designing another detector:

> When does a correctly detected routing shift create enough placement headroom
> to repay cache churn and migration cost?

That question must be tested against the legitimate **do nothing** decision.
Generic caching with switching costs, workload-aware MoE caching, and dynamic
expert promotion already exist, so change detection or migration awareness alone
cannot support a novelty claim.

### Preregistered comparison matrix

Use fresh held-out causal-MoE router traces with stationary, abrupt `A -> B`,
reversal `A -> B -> A`, gradual, and high-churn schedules. Cross each detector
with each action so detector quality is not confused with policy quality:

- no detector and no action;
- a perfect boundary oracle used only as an upper-bound diagnostic;
- a persistent distribution-shift detector; and
- the frozen current detector, without retuning its published seeds;

against:

- no action;
- the frozen failed ShiftCache action;
- a simple LFU/LRU refresh baseline;
- a shadow-gated action that acts only when estimated reuse savings exceed its
  movement cost; and
- a clairvoyant action oracle used only to measure available headroom.

Report detector delay and false-trigger rate separately from false-action rate,
modeled demand-transfer bytes, migration bytes, time-to-benefit, and regret
against the action oracle. Any later latency claim requires a real implementation
and named hardware.

### Provisional evidence gates

Before describing the work as a benchmark contribution, require at least two
causal-MoE model families, three preregistered non-stationary schedules,
immutable trace and configuration hashes, faithful strong baselines, and a
clean-environment reproduction. A candidate action should also:

- regress by no more than 2% on the worst held-out trace versus the strongest
  non-adaptive baseline;
- improve median modeled traffic by at least 5% only where the action oracle
  shows at least 10% available headroom;
- have a paired 95% confidence interval excluding zero; and
- preserve semantic routing exactly.

Until those gates pass, the defensible description is **a preregistered
actionability experiment**, not a new method or a breakthrough.
