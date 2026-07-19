# Two-minute outreach demo

## Opening — about 30 seconds

> Hi, I'm Muhammad Labeeb Aryan, a Form 3 student in Sarawak. I build
> evidence-first AI and computer-science projects, especially RAG evaluation and
> efficient model inference. My current project, StrataMoE Lab, studies how
> Mixture-of-Experts models move expert weights through limited memory. I built a
> reproducible trace-and-simulation harness with a preregistered pass/fail gate,
> and the first idea failed in a useful way: it detected every controlled
> workload shift, but the intervention increased modeled transfer traffic. I
> published the negative result instead of tuning it away.

## Show StrataMoE Lab — about 70 seconds

1. Point to the **simulation-only** boundary at the top of the dashboard.
2. Select **Mid-run domain shift**, seed `2339`, and run all three policies.
3. Explain the controlled comparison:

   > Every policy receives the exact same router-selected experts. LRU uses recency, LFU uses accumulated frequency, and ShiftCache uses a short-versus-long Jensen-Shannon divergence signal plus recent transitions to adapt after the workload changes. The policy is only allowed to move expert weights; it cannot change routing.

4. Point to the provenance-bearing RouterTrace v2 fingerprint `d860285d`, then
   open the captured-trace and detector reports:

   > My first synthetic trace was 6.60% better than LRU, so I tested the
   > mechanism harder instead of treating that as proof. A pinned real-router
   > trace exposed prefetch pollution. Then, in a preregistered 30-seed control,
   > the detector found all 30 known shifts after a median six tokens with zero
   > stationary false triggers, but the action it triggered increased modeled
   > post-shift traffic by 11.52%. The counterintuitive finding is that detecting
   > a workload change does not automatically make an intervention useful.

5. Show **Export router trace** and the methodology/limitations links:

   > I then preregistered 30 new seeds across five cache capacities and tested a
   > causal shadow gate that could choose not to act. It found a useful regime
   > at one capacity, but 21% of its executed actions were harmful and its worst
   > regression was 21%. The safety and coverage gates failed, so I froze that
   > candidate too. This remains a simulator result, not a real GPU speed claim.

## The ask — about 30 seconds

For Sarawak AI Centre:

> Could I show you a five-minute reproducible demo and ask what evidence or
> small supervised contribution would make this useful to your team? My next
> evidence step is to take the frozen actionability harness to a pinned public
> OLMoE routing artifact, then ask whether a new cost model is worth testing on
> fresh data rather than tuning the failed synthetic candidate.

If speaking with SMD Semiconductor (Sarawak Microelectronics Design):

> This project is currently software simulation, not chip design. I am also interested in a later hardware-aware side project around memory traffic, quantization, or edge inference. What beginner-accessible measurement or FPGA-oriented problem would be genuinely useful for me to study?

## Direct answers if challenged

**Does it run a real model?**

> The dashboard does not execute a model. One checked-in trace was captured
> from pinned `google/switch-base-8` execution, but its memory traffic and timing
> are still simulated. I keep that boundary explicit so I do not present modeled
> throughput as hardware evidence.

**Is ShiftCache completely new?**

> I cannot claim that. Work such as MoE-Infinity, DALI, HybriMoE, and Colibrì
> already covers activation-aware or workload-aware caching. My current
> contribution is a transparent deterministic harness, a captured router-trace
> pipeline, mechanism ablations, and a documented negative control for the
> tested JSD-gated action.

**Did you make a breakthrough?**

> Not yet. I have a reproducible counterintuitive finding and a preregistered
> negative control: accurate shift detection still produced harmful cache
> actions, and the benefit changed sharply with cache capacity. A breakthrough
> claim would require a new method to beat strong baselines on held-out real
> models and named hardware, with quality checks and statistical replication.

**Why show a failed intervention?**

> Because the harness is useful only if it can falsify my idea. I kept the
> negative result, isolated prefetch pollution, and stopped the failed action
> instead of selecting only the synthetic trace where it won.

**Why is this useful if it is only a simulator?**

> It made the hypothesis falsifiable before a runtime integration. The harness
> has already exposed a failure case, replayed a captured real-model router
> trace, and shown that a causal action gate can help in one capacity regime
> while still failing its preregistered safety gates—all without changing
> routing.

**What would make it research-grade?**

> Multiple real MoE model traces, preregistered seeds, faithful close baselines, detector and prefetch ablations, statistical uncertainty, then measurements on named hardware.

**Did AI tools write the project?**

> I use AI coding tools, but I treat their output as untrusted. I require tests,
> pinned provenance, explicit evidence boundaries, and preregistered failure
> gates, and I can explain and reproduce every public claim. Keeping the negative
> result is evidence that I do not accept generated ideas just because they sound
> convincing.

## Do not say

- “I invented workload-aware MoE caching.”
- “It makes AI inference 6.6% faster.”
- “It preserves model quality” or “runs huge models on my laptop.”
- “It is an IC-design project.”
- “I made a breakthrough” or “ShiftQ-MoE works.”

Say **modeled**, **simulated**, **on this trace**, and **hypothesis**. That precision makes the work more credible, not less impressive.
