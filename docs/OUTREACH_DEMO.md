# Two-minute outreach demo

## Opening — about 20 seconds

> Hi, I'm Labeeb, a Form 3 student in Sarawak. I build applied AI systems, especially retrieval-augmented generation, agent harnesses, and experiments around efficient local inference. I wanted to show one research-style project and ask how I could develop it with stronger technical guidance.

## Show StrataMoE Lab — about 70 seconds

1. Point to the **simulation-only** boundary at the top of the dashboard.
2. Select **Mid-run domain shift**, seed `2339`, and run all three policies.
3. Explain the controlled comparison:

   > Every policy receives the exact same router-selected experts. LRU uses recency, LFU uses accumulated frequency, and ShiftCache uses a short-versus-long Jensen-Shannon divergence signal plus recent transitions to adapt after the workload changes. The policy is only allowed to move expert weights; it cannot change routing.

4. Point to the trace fingerprint `e5f913fa` and the fixed result:

   > On this one synthetic regression trace, ShiftCache moves 264 million modeled link bytes per token versus 282.67 million for LRU, which is 6.60% lower. Its modeled transfer stall is 5.36% lower. All three policies report zero semantic routing changes.

5. Show **Export router trace** and the methodology/limitations links:

   > The trace can be exported and replayed, so the result is inspectable. This is not a real GPU speed benchmark yet; the next stage is real router traces, stronger workload-aware baselines, ablations, and hardware measurements.

## The ask — about 30 seconds

For Sarawak AI Centre:

> I would value feedback from someone working on inference systems or model serving. My next evidence step is to collect open MoE router traces and test whether the detector still helps against stronger baselines. Is there a researcher or engineer I could speak with for 20 to 30 minutes, or a small scoped experiment I could contribute to?

If speaking with Sarawak Microelectronics Design:

> This project is currently software simulation, not chip design. I am also interested in a later hardware-aware side project around memory traffic, quantization, or edge inference. What beginner-accessible measurement or FPGA-oriented problem would be genuinely useful for me to study?

## Direct answers if challenged

**Does it run a real model?**

> No. It replays router traces through a modeled hierarchy. I separated that clearly so I do not present simulated throughput as hardware evidence.

**Is ShiftCache completely new?**

> I cannot claim that. Work such as MoE-Infinity, DALI, HybriMoE, and Colibrì already covers activation-aware or workload-aware caching. My current contribution is a transparent, deterministic harness for testing this specific JSD-triggered policy coupling.

**Why is this useful if it is only a simulator?**

> It makes the hypothesis falsifiable before spending time on a runtime integration. The same harness can expose failure cases, replay real traces, and compare policies without changing routing.

**What would make it research-grade?**

> Multiple real MoE model traces, preregistered seeds, faithful close baselines, detector and prefetch ablations, statistical uncertainty, then measurements on named hardware.

## Do not say

- “I invented workload-aware MoE caching.”
- “It makes AI inference 6.6% faster.”
- “It preserves model quality” or “runs huge models on my laptop.”
- “It is an IC-design project.”

Say **modeled**, **simulated**, **on this trace**, and **hypothesis**. That precision makes the work more credible, not less impressive.
