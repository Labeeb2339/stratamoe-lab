import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";

import { runActionabilityCell } from "../lib/actionability";
import { generateRouterTrace } from "../lib/simulator";

const evidenceUrl = new URL(
  "../evidence/actionability-v1/results.json",
  import.meta.url,
);

type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

interface CompactArm {
  primaryWindowBytes: number;
  postBoundaryBytes: number;
  wholeTraceBytes: number;
  resultSha256: string;
  semanticRoutingChanges: number;
}

interface EvidenceRecord {
  scenario: "domain-shift" | "steady" | "high-churn";
  seed: number;
  gpuSlots: number;
  traceFingerprint: string;
  traceSha256: string;
  actionabilityResultSha256: string;
  detector: ReturnType<typeof runActionabilityCell>["detector"];
  shadowDecision: ReturnType<typeof runActionabilityCell>["shadowDecision"];
  references: { lru: CompactArm; lfu: CompactArm };
  arms: {
    noAction: CompactArm;
    frozenDetectedAction: CompactArm;
    perfectBoundaryAction: CompactArm;
    trafficShadowGatedAction: CompactArm;
  };
  finiteActionOracle: ReturnType<
    typeof runActionabilityCell
  >["finiteActionOracle"];
  invariants: ReturnType<typeof runActionabilityCell>["invariants"];
}

interface ActionabilityEvidence {
  canonicalPayloadSha256: string;
  payload: {
    schemaVersion: number;
    provenance: {
      cleanTreeRequiredByRunner: boolean;
      protocolCommit: string;
      executionCommit: string;
      implementationCommit: string;
      implementationFileCommits: Record<string, string>;
    };
    configuration: {
      tokens: number;
      knownBoundaryToken: number;
      layers: number;
      expertsPerLayer: number;
      topK: number;
      gpuSlots: number[];
      ramSlots: number;
      expertSizeMB: number;
      pcieGBps: number;
      nvmeGBps: number;
      computeMsPerToken: number;
    };
    records: {
      abrupt: EvidenceRecord[];
      stationary: EvidenceRecord[];
    };
    summary: {
      abruptCells: number;
      detectedWithin64Count: number;
      executedActions: number;
      harmfulExecutedActions: number;
      harmfulExecutedActionRate: number;
      maximumRegressionPercent: number;
      oracleActionableCells: number;
      oracleActionableCapacities: number[];
      stationaryDetectorEvents: number;
      stationaryGatedActions: number;
    };
    bootstrap: {
      oracleActionableCells: {
        estimate: number;
        lower: number;
        upper: number;
        validResamples: number;
      };
    };
    gates: Record<string, { passed: boolean }>;
    carryForward: boolean;
  };
}

function canonicalize(value: unknown): JsonValue {
  if (value === null) return null;
  if (typeof value === "boolean" || typeof value === "string") return value;
  if (typeof value === "number") {
    assert.ok(Number.isFinite(value));
    return Object.is(value, -0) ? 0 : value;
  }
  if (Array.isArray(value)) return value.map(canonicalize);
  assert.equal(typeof value, "object");
  const record = value as Record<string, unknown>;
  return Object.fromEntries(
    Object.keys(record)
      .sort()
      .map((key) => [key, canonicalize(record[key])]),
  );
}

function sha256Canonical(value: unknown): string {
  return createHash("sha256")
    .update(JSON.stringify(canonicalize(value)), "utf8")
    .digest("hex");
}

test("checked-in actionability evidence freezes the failed carry-forward decision", () => {
  const raw = readFileSync(evidenceUrl, "utf8");
  const evidence = JSON.parse(raw) as ActionabilityEvidence;

  assert.equal(
    createHash("sha256").update(raw, "utf8").digest("hex"),
    "be80f2238f8dd33587f0bf1f00656d3a963138368c82ca7f8840e9863cb4f681",
  );
  assert.equal(
    evidence.canonicalPayloadSha256,
    "7b0b123d81344eb5b12e22f6539a266e00706b324e45160ff15854c3eca4f6c6",
  );
  assert.equal(evidence.payload.schemaVersion, 1);
  assert.equal(evidence.payload.provenance.cleanTreeRequiredByRunner, true);
  assert.equal(
    evidence.payload.provenance.executionCommit,
    "090481f4b710996a343e3b07332d69ff16c03258",
  );
  assert.equal(
    evidence.payload.provenance.implementationCommit,
    "090481f4b710996a343e3b07332d69ff16c03258",
  );
  assert.equal(
    evidence.payload.provenance.protocolCommit,
    "daf79356365b21249fde42012f1b446069809548",
  );
  assert.equal(
    Object.keys(evidence.payload.provenance.implementationFileCommits).length,
    7,
  );
  assert.ok(
    Object.values(
      evidence.payload.provenance.implementationFileCommits,
    ).every(
      (commit) => commit === "090481f4b710996a343e3b07332d69ff16c03258",
    ),
  );
  assert.equal(evidence.payload.records.abrupt.length, 150);
  assert.equal(evidence.payload.records.stationary.length, 30);
  assert.deepEqual(evidence.payload.summary, {
    abruptCells: 150,
    detectedWithin64Count: 30,
    executedActions: 76,
    falseActions: 0,
    harmfulExecutedActionRate: 0.21052631578947367,
    harmfulExecutedActions: 16,
    maximumRegressionPercent: 21.339950372208435,
    medianFrozenPrimaryBytes: 29696000000,
    medianGatedPrimaryBytes: 29696000000,
    oracleActionableCapacities: [64],
    oracleActionableCells: 30,
    preBoundaryEvents: 0,
    stationaryActionsPer10000Tokens: 0,
    stationaryCells: 30,
    stationaryDetectorEvents: 0,
    stationaryEventsPer10000Tokens: 0,
    stationaryGatedActions: 0,
    stationaryTokens: 15360,
  });
  assert.deepEqual(evidence.payload.bootstrap.oracleActionableCells, {
    emptyResamples: 0,
    estimate: -9.748178650617675,
    lower: -10.322794066439167,
    lowerPercentile: 2.5,
    prng: "xorshift32",
    resamplesRequested: 10000,
    seed: 2339,
    statistic: "median-paired-percent-change",
    unit: "seed-cluster",
    upper: -9.449404761904763,
    upperPercentile: 97.5,
    validResamples: 10000,
  });
  assert.deepEqual(
    Object.fromEntries(
      Object.entries(evidence.payload.gates).map(([name, gate]) => [
        name,
        gate.passed,
      ]),
    ),
    {
      byteIdenticalRerun: true,
      completeCellsAndRouting: true,
      detector: true,
      gatedMedianNoWorseThanFrozen: true,
      harmfulExecutedActions: false,
      maximumRegression: false,
      oracleActionableEffect: true,
      oracleCoverage: false,
      semanticRoutingChangesZero: true,
      stationaryControls: true,
    },
  );
  assert.equal(evidence.payload.carryForward, false);
});

test("one checked-in actionability cell replays from the recorded configuration", () => {
  const evidence = JSON.parse(
    readFileSync(evidenceUrl, "utf8"),
  ) as ActionabilityEvidence;
  const record = evidence.payload.records.abrupt[0];
  const base = evidence.payload.configuration;
  const config = {
    scenario: record.scenario,
    seed: record.seed,
    tokens: base.tokens,
    layers: base.layers,
    expertsPerLayer: base.expertsPerLayer,
    topK: base.topK,
    gpuSlots: record.gpuSlots,
    ramSlots: base.ramSlots,
    expertSizeMB: base.expertSizeMB,
    pcieGBps: base.pcieGBps,
    nvmeGBps: base.nvmeGBps,
    computeMsPerToken: base.computeMsPerToken,
  } as const;
  const trace = generateRouterTrace({
    scenario: config.scenario,
    seed: config.seed,
    tokens: config.tokens,
    layers: config.layers,
    expertsPerLayer: config.expertsPerLayer,
    topK: config.topK,
  });
  const replay = runActionabilityCell({
    config,
    trace,
    knownBoundaryToken: base.knownBoundaryToken,
  });

  assert.equal(replay.arms.noAction.result.traceFingerprint, record.traceFingerprint);
  assert.equal(sha256Canonical(trace), record.traceSha256);
  assert.equal(sha256Canonical(replay), record.actionabilityResultSha256);
  assert.deepEqual(replay.detector, record.detector);
  assert.deepEqual(replay.shadowDecision, record.shadowDecision);
  assert.deepEqual(replay.finiteActionOracle, record.finiteActionOracle);
  assert.deepEqual(replay.invariants, record.invariants);
  assert.equal(
    sha256Canonical(replay.references.lru.result),
    record.references.lru.resultSha256,
  );
  assert.equal(
    sha256Canonical(replay.references.lfu.result),
    record.references.lfu.resultSha256,
  );
  for (const key of Object.keys(record.arms) as Array<keyof typeof record.arms>) {
    assert.equal(
      sha256Canonical(replay.arms[key].result),
      record.arms[key].resultSha256,
    );
    assert.equal(
      replay.arms[key].primaryWindowBytes,
      record.arms[key].primaryWindowBytes,
    );
  }
});
