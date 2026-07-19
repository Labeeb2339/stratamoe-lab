# Shift actionability pilot v1

**Protocol status:** frozen before any confirmatory seed in this document is
executed.

**Evidence boundary:** this is a deterministic, CPU-only synthetic
falsification study. It is not a benchmark contribution, a new cache method, a
quantization result, a hardware measurement, or a breakthrough claim.

## Question

When a frozen distribution-shift detector fires, can a causal shadow replay
decide whether the already-published 64-token retention intervention is worth
applying, including the legitimate decision to do nothing?

This separates three questions that the previous detector sanity check could
not answer on its own:

1. Was a shift detected on time?
2. Was there any benefit available inside the finite action family?
3. Could a decision rule identify that benefit without reading future tokens?

## Prior observations and exclusions

Seeds `2300` through `2329` produced the published detector sanity result and
remain frozen development evidence. Seeds `0` through `9` and `2339` may be
used only in unit tests. A local feasibility sweep used seeds `3100` and `3101`
before this protocol was written; it showed that the sign of the frozen action's
traffic change varied with GPU capacity. Those two seeds are exploratory and
excluded from every v1 result and gate.

No seed from `4100` through `4129` may be inspected until this protocol is
committed and pushed.

## Confirmatory matrix

The paired statistical unit is the seed. The confirmatory seeds are `4100`
through `4129`, inclusive.

Every abrupt-shift trace uses:

| Parameter | Frozen value |
| --- | ---: |
| Tokens | 512 |
| Known boundary | 256 |
| Layers | 8 |
| Experts per layer | 16 |
| Router top-k | 2 |
| RAM slots | 64 |
| Expert size | 64 MB (decimal) |
| PCIe bandwidth | 24 GB/s (decimal) |
| NVMe bandwidth | 7 GB/s (decimal) |
| Modeled compute | 6 ms/token |

GPU capacity is evaluated on the complete systematic grid `8, 16, 32, 64,
96`. With 128 layer-expert slots, these are 6.25%, 12.5%, 25%, 50%, and 75%
resident capacity. Capacity is an input known before routing, not a learned
feature.

Each seed also has one 512-token stationary trace at 32 GPU slots. The detector
signal depends only on the trace, so duplicating that stationary trace across
all capacities would not create additional independent false-trigger evidence.

The existing generator's midpoint `domain-shift` and `steady` schedules are
used unchanged. Different seeds are new stochastic fixtures, not independent
semantic workload families.

## Frozen detector and action

Both causal ShiftCache arms disable transition prefetch, continuous JSD
reweighting, and transition retention. The detector keeps the published
parameters:

- short window: 12 tokens;
- long window: 48 tokens;
- JSD threshold: 0.28 bits;
- rearm threshold: 0.12 bits;
- persistence: 3 tokens;
- cooldown: 64 tokens; and
- action horizon: 64 tokens.

If the detector reports an event after token `d`, the published frozen action
can affect eviction decisions beginning at token `d + 1`.

## Arms

All arms replay identical ordered expert selections.

1. **LRU reference:** the existing LRU policy.
2. **LFU reference:** the existing lifetime-LFU policy.
3. **No action:** the persistent detector records events, but retention weights
   remain fixed.
4. **Frozen detected action:** the already-published intervention applies
   maximum short-window/recency reweighting for 64 tokens after a detector
   event.
5. **Perfect-boundary timing diagnostic:** the same action is forced for tokens
   `256` through `319`. It receives the known generator boundary and is not a
   deployable competitor.
6. **Traffic-shadow-gated action:** after the first detector event, maintain
   no-action and frozen-action shadow states for exactly 12 causal tokens,
   `d + 1` through `d + 12`. Act only if the frozen shadow used at least 5%
   fewer modeled link bytes over that prefix. If it acts, reweighting begins at
   `d + 13` and ends at `d + 65` exclusive, preserving only the remainder of
   the original 64-token horizon. Ties and incomplete decision windows abstain.
7. **Finite-action oracle diagnostic:** select the lowest-traffic result among
   no action, frozen detected action, and perfect-boundary action for each
   trace. This is an ex-post upper bound inside three frozen choices, not a
   global cache oracle.

The shadow decision may read only timeline entries before its decision token.
Changing any later suffix must not change the decision.

## Traffic model and metrics

The primary traffic quantity is the repository's existing modeled link-byte
counter: demand RAM-to-GPU bytes, prefetch RAM-to-GPU bytes, and NVMe-to-RAM
reads. Prefetch is disabled in these arms. The legacy model does not charge
GPU-to-RAM demotions or proactive state-reconciliation migration, so this study
must not use the phrase "migration-aware" for the shadow gate.

For every seed-capacity cell, record:

- trace fingerprint plus a SHA-256 digest of the canonical trace;
- detector events, first-event delay, and pre-boundary events;
- post-boundary bytes over tokens `256` through `319`;
- whole post-boundary and whole-trace bytes;
- the shadow observation range, decision, estimated prefix saving, and applied
  action range;
- finite-action-oracle choice and headroom;
- false action, defined as acting when the finite oracle chooses no action;
- harmful action, defined as gated traffic exceeding no-action traffic;
- first permanent break-even token within the 64-token primary window;
- regret in bytes against the finite-action oracle;
- LRU and LFU reference traffic; and
- semantic routing changes, which must remain zero.

Oracle headroom is:

```text
(no_action_bytes - finite_action_oracle_bytes) / no_action_bytes
```

The primary paired effect is gated-action versus no-action percent change in
the first 64 post-boundary tokens. LRU and LFU remain external references so a
weak ShiftCache-relative result cannot be presented as general cache
superiority.

## Statistics

Use a paired cluster percentile bootstrap with:

- seed as the resampled cluster;
- 10,000 resamples;
- xorshift32;
- bootstrap seed `2339`; and
- a two-sided 95% percentile interval.

For the all-cell endpoint, each resampled seed contributes all five capacity
cells and the statistic is the median paired percent change. For the
oracle-actionable endpoint, retain cells whose finite-action oracle headroom is
at least 10%, but still resample complete seed clusters. Gate calculations use
unrounded values; rounding is presentation-only.

## Carry-forward gates

Every item must pass:

1. all 150 abrupt trace-capacity cells replay completely, share exact routing
   within a cell, and report zero semantic routing changes;
2. at least 27 of 30 abrupt traces detect the known change within 64 tokens,
   with no pre-boundary event;
3. stationary traces produce at most one detector event and at most one gated
   action per 10,000 tokens;
4. at least 30 cells spanning at least three capacities have finite-action
   oracle headroom of 10% or more;
5. on those oracle-actionable cells, the gated action improves median primary
   traffic by at least 5% versus no action and the paired-bootstrap interval is
   entirely below zero;
6. no confirmatory cell regresses by more than 2% versus no action;
7. at most 10% of executed gated actions are harmful;
8. the gated arm's median primary traffic is no worse than the frozen detected
   action; and
9. an immediate clean rerun produces byte-identical canonical evidence.

Failure freezes this candidate. The confirmatory seeds must not be used to tune
the shadow length, threshold, capacity grid, detector, or action.

Even a pass permits only this sentence:

> The causal shadow gate passed a preregistered synthetic carry-forward pilot
> for the frozen retention action.

## External real-trace phase (outside v1 gates)

A later external pilot may use
`miguelbetances/moe-finance-specialization` at dataset revision
`9bbdd12571d410e7af45f9f77231d4282486fdef`, routing artifact
`routing_traces/routing_v1.parquet`, SHA-256
`c63d56354c8c710f497155578ef1f91a6829f0e8f4dda215b80ff6614ed48614`.
It must be labelled third-party OLMoE prefill routing under CC BY-SA 4.0. The
upstream capture did not report an immutable model revision, so it cannot be
confirmatory evidence for a model-version-specific claim.

A benchmark or method claim still requires fresh preregistered traces from at
least two causal-MoE model families and independent reproduction.
