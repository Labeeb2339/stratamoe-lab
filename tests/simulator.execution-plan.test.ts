import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_CONFIG,
  DEFAULT_SIMULATION_CONTROLS,
  runPlannedSimulation,
  runSimulation,
  validateSimulationExecutionPlan,
  type RouterTraceV2,
  type SimulationConfig,
  type SimulationControls,
} from "../lib/simulator";

const FIXED_CONTROLS = {
  ...DEFAULT_SIMULATION_CONTROLS,
  shiftCachePrefetch: false,
  shiftCacheJsdReweighting: false,
  shiftCacheTransitionRetention: false,
  shiftCachePersistentDetector: false,
  shiftCacheTriggeredReweighting: false,
} satisfies SimulationControls;

const CONFIG = {
  ...DEFAULT_CONFIG,
  scenario: "steady",
  policy: "shift-cache",
  seed: 7,
  tokens: 16,
  layers: 1,
  expertsPerLayer: 4,
  topK: 1,
  gpuSlots: 2,
  ramSlots: 0,
} satisfies SimulationConfig;

const EXPERT_SEQUENCE = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 3, 1,
] as const;

const TRACE: RouterTraceV2 = {
  version: 2,
  source: {
    kind: "synthetic",
    generator: "tests/execution-plan",
  },
  scenario: CONFIG.scenario,
  seed: CONFIG.seed,
  tokens: CONFIG.tokens,
  layers: CONFIG.layers,
  expertsPerLayer: CONFIG.expertsPerLayer,
  topK: CONFIG.topK,
  selections: EXPERT_SEQUENCE.map((expert) => [[expert]]),
};

test("execution-plan validation accepts sorted adjacent ShiftCache windows", () => {
  const plan = {
    forcedReweightingWindows: [
      { startToken: 2, endTokenExclusive: 5 },
      { startToken: 5, endTokenExclusive: 8 },
    ],
  } as const;

  assert.deepEqual(
    validateSimulationExecutionPlan(plan, {
      policy: "shift-cache",
      tokens: CONFIG.tokens,
    }),
    plan,
  );
  assert.deepEqual(
    validateSimulationExecutionPlan(
      { forcedReweightingWindows: [] },
      { policy: "lru", tokens: CONFIG.tokens },
    ),
    { forcedReweightingWindows: [] },
  );
});

test("execution-plan validation rejects invalid windows and non-ShiftCache actions", () => {
  const context = { policy: "shift-cache", tokens: CONFIG.tokens } as const;

  assert.throws(
    () =>
      validateSimulationExecutionPlan(
        {
          forcedReweightingWindows: [
            { startToken: 8, endTokenExclusive: 10 },
            { startToken: 3, endTokenExclusive: 5 },
          ],
        },
        context,
      ),
    /sorted by startToken/,
  );
  assert.throws(
    () =>
      validateSimulationExecutionPlan(
        {
          forcedReweightingWindows: [
            { startToken: 2, endTokenExclusive: 6 },
            { startToken: 5, endTokenExclusive: 8 },
          ],
        },
        context,
      ),
    /must not overlap/,
  );
  assert.throws(
    () =>
      validateSimulationExecutionPlan(
        {
          forcedReweightingWindows: [
            { startToken: 4, endTokenExclusive: 4 },
          ],
        },
        context,
      ),
    /less than endTokenExclusive/,
  );
  assert.throws(
    () =>
      validateSimulationExecutionPlan(
        {
          forcedReweightingWindows: [
            { startToken: 15, endTokenExclusive: 17 },
          ],
        },
        context,
      ),
    /cannot exceed the simulation token count/,
  );
  assert.throws(
    () =>
      validateSimulationExecutionPlan(
        {
          forcedReweightingWindows: [
            { startToken: 2, endTokenExclusive: 3 },
          ],
        },
        { policy: "lfu", tokens: CONFIG.tokens },
      ),
    /only supported by the shift-cache policy/,
  );
});

test("an empty execution plan is deeply compatible with runSimulation", () => {
  const historical = runSimulation(CONFIG, TRACE, FIXED_CONTROLS);
  const planned = runPlannedSimulation(
    CONFIG,
    TRACE,
    FIXED_CONTROLS,
    { forcedReweightingWindows: [] },
  );

  assert.deepEqual(planned, historical);
});

test("a forced window changes ShiftCache adaptation from its start token", () => {
  const noAction = runSimulation(CONFIG, TRACE, FIXED_CONTROLS);
  const forced = runPlannedSimulation(
    CONFIG,
    TRACE,
    FIXED_CONTROLS,
    {
      forcedReweightingWindows: [
        { startToken: 11, endTokenExclusive: 12 },
      ],
    },
  );

  assert.deepEqual(forced.timeline.slice(0, 11), noAction.timeline.slice(0, 11));
  assert.ok(noAction.timeline[12].bytesTransferred > 0);
  assert.equal(forced.timeline[12].bytesTransferred, 0);
  assert.deepEqual(forced.trace.selections, noAction.trace.selections);
  assert.equal(forced.traceFingerprint, noAction.traceFingerprint);
  assert.equal(forced.metrics.semanticRoutingChanges, 0);
});
