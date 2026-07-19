# Preregistered detector sanity sweep

## Question

Can a persistent JSD event safely gate temporary ShiftCache reweighting on
deterministic synthetic traces with a known midpoint change?

This is an engineering sanity check. It is not evidence about semantic domains,
language-model quality, a physical memory hierarchy, or ShiftQ-MoE novelty.

## Preregistration

The seed range, token count, requirement to add persistence/cooldown before the
run, and four gates were merged in commit
[`fe6042b`](https://github.com/Labeeb2339/stratamoe-lab/commit/fe6042b6cd628d4976c2069a94d2739614ce9ce9)
before the sweep ran:

- seeds `2300` through `2329`;
- `1024` tokens per stationary and abrupt-shift trace;
- known synthetic change at token `512`;
- detection, stationary false-trigger, post-shift traffic, and stationary
  traffic gates with fixed thresholds.

The implementation was then written and frozen before its first sweep with:

- three consecutive above-threshold JSD scores;
- a 64-token cooldown and 64-token triggered reweighting interval;
- prefetch and transition retention disabled in both arms; and
- fixed frequency/recency scoring as the paired baseline.

The triggered arm uses maximum short-window/recency reweighting for 64 tokens
after an accepted event. Parameters were not changed after the results were
observed. The exact `3/64/64` implementation values were not independently
timestamped in the earlier public commit, so they are a frozen reproducibility
record rather than a fully public preregistration claim.

## Gates and result

| Gate | Requirement | Result | Pass |
| --- | --- | ---: | :---: |
| Detection | At least 27/30 shifts within 64 tokens | 30/30; median delay 6 tokens | Yes |
| Stationary false triggers | At most 1 per 10,000 tokens | 0 | Yes |
| Post-shift traffic | Median change at most -5%; bootstrap interval below 0 | **+11.52%**, 95% interval **[+10.78%, +12.64%]** | **No** |
| Stationary traffic | Median regression no greater than 2% | 0% | Yes |

All 30 triggered runs moved more modeled link bytes than their paired fixed
baseline over tokens `512` through `575`. The detector recognized this strong,
synthetic boundary, but its prescribed cache intervention consistently made the
placement outcome worse.

The registered decision is therefore:

> **Do not carry this JSD-gated retention intervention into ShiftQ-MoE.**

This result rejects one mechanism, not every possible change detector. Retuning
the threshold, hold interval, or weights on these same 30 seeds would invalidate
the preregistered test and is intentionally not done.

## Reproduce

```bash
npm ci
npm run benchmark:detector
```

The run takes roughly two minutes on the development machine. The complete
per-seed record, controls, fingerprints, bootstrap settings, gate decisions,
and evidence boundary are checked in as
[`detector-sanity.json`](../evidence/synthetic/detector-sanity.json).

The result may be described as:

> On 30 deterministic synthetic traces with a known midpoint change, the
> persistent detector fired after six tokens with no stationary false triggers,
> but its fixed 64-token intervention increased simulated post-shift link traffic
> by a median 11.52%; the preregistered carry-forward gate failed.

It must not be described as a real-model inference slowdown, a quantization
result, or proof that all distribution-change detection is ineffective.
