# Shift actionability pilot v1: result

**Decision:** `carryForward = false`.

This is a preregistered synthetic falsification result. It is not a benchmark
contribution, a new cache method, a quantization result, a hardware
measurement, or a breakthrough claim.

## What was tested

The protocol was committed and pushed at
`daf79356365b21249fde42012f1b446069809548` before any confirmatory seed was
run. The implementation was then committed and pushed at
`090481f4b710996a343e3b07332d69ff16c03258` before execution.

The runner evaluated 30 untouched seeds (`4100` through `4129`) on one abrupt
midpoint-shift generator at five GPU capacities (`8`, `16`, `32`, `64`, and
`96` slots), producing 150 paired cells. It also ran 30 stationary controls at
32 slots. Every arm replayed identical expert selections.

The causal decision rule observed 12 tokens after the first detector event. It
applied the remaining 52 tokens of the frozen retention action only when a
counterfactual shadow state had used at least 5% fewer modeled link bytes over
that observed prefix. The rule could not read later tokens.

## Headline result

The detector again worked on the controlled generator: all 30 shifts were
detected after seven tokens, with no pre-boundary events and no stationary
events. Detection correctness did not make the action safe.

- The gate acted in 76 of 150 abrupt cells.
- 16 of those 76 actions were harmful: **21.05%**, above the frozen 10% limit.
- The worst cell regressed by **21.34%**, above the frozen 2% limit.
- Thirty cells had at least 10% finite-action-oracle headroom, but all 30 were
  at one capacity (`64` slots); the protocol required at least three
  capacities.
- Inside that narrow oracle-actionable subset, the gated action did improve
  primary traffic by a median **9.75%**, with a seed-cluster bootstrap 95%
  interval of **[9.45%, 10.32%] savings**.
- The overall all-capacity median change was `0%` because the gate abstained at
  two capacities and had opposing effects at the others.

Three mandatory gates failed, so the candidate is frozen without retuning on
these seeds.

## Capacity breakdown

All entries are per-capacity medians for first-64-token post-boundary modeled
link bytes. Negative change is less traffic. LRU is an external reference, not
part of the finite three-choice action oracle.

| GPU slots | Gated actions | Harmful | Gated vs no action | Frozen action vs no action | Boundary timing diagnostic vs no action | LRU vs no action | Finite-oracle headroom |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 30 / 30 | 0 | -1.43% | -3.50% | -3.59% | -3.59% | 3.59% |
| 16 | 0 / 30 | 0 | 0.00% | -0.05% | -0.19% | -7.38% | 0.19% |
| 32 | 16 / 30 | 16 | +14.15% | +14.94% | -4.98% | -19.95% | 4.98% |
| 64 | 30 / 30 | 0 | -9.75% | -30.00% | -36.11% | -36.11% | 36.11% |
| 96 | 0 / 30 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

The important pattern is not simply that one capacity won. At 32 slots the
perfect-boundary diagnostic showed that useful action headroom existed, yet
the delayed shadow-gated action harmed every cell in which it acted. At 64
slots the gated action helped, but the existing LRU reference helped much more.
That prevents presenting the narrow positive subset as a generally competitive
new policy.

## Frozen gates

| Gate | Result |
| --- | --- |
| 150 complete cells, identical routing | Pass |
| At least 27/30 timely detections; no pre-boundary event | Pass (30/30) |
| Stationary events/actions at most 1 per 10,000 tokens | Pass (0 / 0) |
| At least 30 oracle-actionable cells across at least 3 capacities | **Fail** (30 cells, 1 capacity) |
| At least 5% median improvement on oracle-actionable cells; bootstrap interval below zero | Pass (-9.75%, interval [-10.32%, -9.45%]) |
| No cell regresses more than 2% | **Fail** (worst +21.34%) |
| Harmful executed-action rate at most 10% | **Fail** (21.05%) |
| Gated median no worse than frozen action | Pass (tie) |
| Exact routing and zero semantic routing changes | Pass |
| Fresh-process byte-identical core rerun | Pass |

Passing one subset-effect gate cannot override three mandatory failures.

## Evidence and reproduction

The canonical result is
[`evidence/actionability-v1/results.json`](../evidence/actionability-v1/results.json).

- Raw file SHA-256:
  `be80f2238f8dd33587f0bf1f00656d3a963138368c82ca7f8840e9863cb4f681`
- Canonical payload SHA-256:
  `7b0b123d81344eb5b12e22f6539a266e00706b324e45160ff15854c3eca4f6c6`
- Protocol commit: `daf79356365b21249fde42012f1b446069809548`
- Execution and implementation commit:
  `090481f4b710996a343e3b07332d69ff16c03258`

From the implementation commit on a clean tree:

```bash
npm ci
npm run benchmark:actionability
npm run benchmark:actionability:verify
```

The write command refuses a dirty tree or an existing result. Before writing,
it recomputes the full core in a fresh Node.js process and compares canonical
bytes. The verify command independently rebuilt the checked-in result and
matched it byte-for-byte.

## Claim boundary and next step

The traffic counter is the repository's legacy model: RAM-to-GPU demand bytes,
prefetch bytes, and NVMe-to-RAM reads. Prefetch was disabled. GPU-to-RAM
demotions and proactive reconciliation migration are not charged, so this is
not a migration-aware or hardware-performance result.

The strongest truthful conclusion is:

> On a preregistered synthetic capacity sweep, a causal 12-token shadow gate
> found a useful action regime at 64 GPU slots but failed its safety and
> coverage gates. Correct shift detection was not enough to choose a generally
> safe intervention.

Do not tune the detector, shadow length, threshold, capacity grid, or action on
seeds `4100` through `4129`. Any next candidate needs a new protocol and fresh
data. The planned third-party OLMoE finance/general trace can be an explicitly
exploratory external check, but a benchmark or method claim still requires
multiple revision-pinned causal-MoE families, stronger baselines, and
independent reproduction.
