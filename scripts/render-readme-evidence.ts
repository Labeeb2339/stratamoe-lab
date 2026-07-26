import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { buildFixedBenchmark } from "./benchmark-fixture";

type PolicyResult = {
  policy: "lru" | "lfu" | "shift-cache";
  metrics: { bytesPerToken: number; semanticRoutingChanges: number };
};

type CapturedEvidence = {
  benchmark: string;
  results: PolicyResult[];
};

type ActionabilityRecord = {
  gpuSlots: number;
  gatedPrimaryPercentChangeVsNoAction: number;
  harmfulAction: boolean;
  seed: number;
  shadowDecision: { act: boolean };
};

type ActionabilityEvidence = {
  payload: {
    carryForward: boolean;
    records: {
      abrupt: ActionabilityRecord[];
      stationary: ActionabilityRecord[];
    };
    summary: {
      abruptCells: number;
      executedActions: number;
      harmfulExecutedActions: number;
      harmfulExecutedActionRate: number;
      maximumRegressionPercent: number;
      stationaryCells: number;
      stationaryDetectorEvents: number;
      stationaryGatedActions: number;
    };
  };
};

const overviewOutputUrl = new URL(
  "../public/stratamoe-evidence.svg",
  import.meta.url,
);
const capacityOutputUrl = new URL(
  "../public/actionability-capacity-map.svg",
  import.meta.url,
);
const capturedUrl = new URL(
  "../evidence/switch-base-8/comparison.json",
  import.meta.url,
);
const actionabilityUrl = new URL(
  "../evidence/actionability-v1/results.json",
  import.meta.url,
);

function median(values: number[]) {
  const sorted = [...values].sort((left, right) => left - right);
  const midpoint = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[midpoint - 1] + sorted[midpoint]) / 2
    : sorted[midpoint];
}

function escapeXml(value: string) {
  return value.replace(
    /[&<>"']/g,
    (character) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&apos;",
      })[character]!,
  );
}

const policyOrder: PolicyResult["policy"][] = ["shift-cache", "lru", "lfu"];
const policyLabel = {
  "shift-cache": "ShiftCache",
  lru: "LRU",
  lfu: "LFU",
};
const policyColor = {
  "shift-cache": "#65d4d1",
  lru: "#bcc8c7",
  lfu: "#efb64a",
};

function resultPanel(
  title: string,
  subtitle: string,
  results: PolicyResult[],
  originX: number,
) {
  const maxValue = 650;
  const barWidth = 410;
  const rows = policyOrder.map((policy, index) => {
    const result = results.find((entry) => entry.policy === policy);
    if (!result) throw new Error(`Missing ${policy} result for ${title}.`);
    if (result.metrics.semanticRoutingChanges !== 0) {
      throw new Error(`${title} no longer preserves router selections.`);
    }
    const value = result.metrics.bytesPerToken / 1_000_000;
    const width = (value / maxValue) * barWidth;
    const y = 226 + index * 86;
    return `
      <text x="${originX}" y="${y - 10}" class="label">${policyLabel[policy]}</text>
      <rect x="${originX}" y="${y}" width="${barWidth}" height="24" rx="3" class="track"/>
      <rect x="${originX}" y="${y}" width="${width.toFixed(2)}" height="24" rx="3" fill="${policyColor[policy]}"/>
      <text x="${originX + barWidth + 18}" y="${y + 19}" class="value">${value.toFixed(2)} MB</text>`;
  });

  return `
    <g>
      <text x="${originX}" y="126" class="panel-title">${escapeXml(title)}</text>
      <text x="${originX}" y="158" class="subtitle">${escapeXml(subtitle)}</text>
      ${rows.join("")}
    </g>`;
}

function renderSvg() {
  const synthetic = buildFixedBenchmark();
  const captured = JSON.parse(
    readFileSync(capturedUrl, "utf8"),
  ) as CapturedEvidence;
  const actionability = JSON.parse(
    readFileSync(actionabilityUrl, "utf8"),
  ) as ActionabilityEvidence;

  const capacities = [8, 16, 32, 64, 96].map((gpuSlots) => {
    const records = actionability.payload.records.abrupt.filter(
      (record) => record.gpuSlots === gpuSlots,
    );
    if (records.length !== 30) {
      throw new Error(`Expected 30 actionability cells at ${gpuSlots} GPU slots.`);
    }
    return {
      gpuSlots,
      change: median(
        records.map((record) => record.gatedPrimaryPercentChangeVsNoAction),
      ),
      actions: records.filter((record) => record.shadowDecision.act).length,
      harmful: records.filter((record) => record.harmfulAction).length,
    };
  });

  const summary = actionability.payload.summary;
  if (actionability.payload.carryForward !== false) {
    throw new Error("The checked-in actionability decision unexpectedly changed.");
  }

  const zeroX = 760;
  const scale = 19;
  const capacityRows = capacities.map((entry, index) => {
    const y = 596 + index * 58;
    const width = Math.abs(entry.change) * scale;
    const barX = entry.change < 0 ? zeroX - width : zeroX;
    const color = entry.change < 0 ? "#65d4d1" : entry.change > 0 ? "#ef6e63" : "#bcc8c7";
    const signed = `${entry.change > 0 ? "+" : ""}${entry.change.toFixed(2)}%`;
    const valueX = entry.change < 0 ? barX - 14 : zeroX + width + 14;
    const anchor = entry.change < 0 ? "end" : "start";
    return `
      <text x="120" y="${y + 17}" class="capacity">${entry.gpuSlots} slots</text>
      <text x="238" y="${y + 17}" class="small">${entry.actions}/30 actions · ${entry.harmful} harmful</text>
      <rect x="${barX.toFixed(2)}" y="${y}" width="${Math.max(width, 2).toFixed(2)}" height="22" rx="3" fill="${color}"/>
      <text x="${valueX.toFixed(2)}" y="${y + 17}" text-anchor="${anchor}" class="value">${signed}</text>`;
  });

  return `<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="930" viewBox="0 0 1500 930" role="img" aria-labelledby="title desc">
  <title id="title">StrataMoE trace and actionability evidence</title>
  <desc id="desc">Modeled link traffic for ShiftCache, LRU, and LFU on one synthetic trace and one captured Switch-Base-8 router trace, followed by the traffic change from a causal shadow gate across five modeled GPU capacities.</desc>
  <style>
    .title{font:700 34px ui-monospace,SFMono-Regular,Consolas,monospace;fill:#f2f3ed;letter-spacing:1px}.panel-title{font:700 24px ui-monospace,SFMono-Regular,Consolas,monospace;fill:#f2f3ed}.subtitle,.small,.footer{font:400 16px Inter,Segoe UI,sans-serif;fill:#91a4a5}.label,.capacity{font:600 17px ui-monospace,SFMono-Regular,Consolas,monospace;fill:#e7ece8}.value{font:600 16px ui-monospace,SFMono-Regular,Consolas,monospace;fill:#f2f3ed}.track{fill:#182629}.rule{stroke:#294044;stroke-width:1}.zero{stroke:#d8dfdb;stroke-width:2}.panel{fill:#0d1719;stroke:#263a3d;stroke-width:1}
  </style>
  <rect width="1500" height="930" fill="#081113"/>
  <path d="M40 68H1460" class="rule"/>
  <text x="70" y="54" class="title">TRACE RESULT / ACTIONABILITY CHECK</text>
  <text x="1430" y="53" text-anchor="end" class="subtitle">modeled link bytes / token · decimal MB · lower is better</text>
  <rect x="70" y="92" width="650" height="380" rx="8" class="panel"/>
  <rect x="780" y="92" width="650" height="380" rx="8" class="panel"/>
  ${resultPanel("Synthetic domain shift", "Seed 2339 · fixed deterministic fixture", synthetic.results, 110)}
  ${resultPanel("Captured router trace", "Switch-Base-8 · placement remains simulated", captured.results, 820)}
  <text x="70" y="506" class="panel-title">CAUSAL SHADOW GATE / FIRST 64 POST-SHIFT TOKENS</text>
  <text x="70" y="535" class="subtitle">Median modeled-link-byte change versus no action · lower is better</text>
  <text x="540" y="566" class="small">SAVINGS ←</text>
  <text x="748" y="566" class="small">0</text>
  <text x="790" y="566" class="small">→ REGRESSION</text>
  <path d="M760 576V866" class="zero"/>
  ${capacityRows.join("")}
  <path d="M40 884H1460" class="rule"/>
  <text x="70" y="914" class="footer">${summary.executedActions}/${summary.abruptCells} actions · ${summary.harmfulExecutedActions} harmful (${(summary.harmfulExecutedActionRate * 100).toFixed(2)}%) · worst regression +${summary.maximumRegressionPercent.toFixed(2)}% · carryForward = false</text>
</svg>
`;
}

function heatmapFill(percentChange: number) {
  if (percentChange === 0) return { color: "#243638", opacity: 1 };
  const opacity = Math.min(0.95, 0.34 + (Math.abs(percentChange) / 22) * 0.61);
  return {
    color: percentChange < 0 ? "#65d4d1" : "#ef6e63",
    opacity,
  };
}

function renderCapacityMap() {
  const actionability = JSON.parse(
    readFileSync(actionabilityUrl, "utf8"),
  ) as ActionabilityEvidence;
  const abrupt = actionability.payload.records.abrupt;
  const seeds = [...new Set(abrupt.map((record) => record.seed))].sort(
    (left, right) => left - right,
  );
  const gpuSlots = [8, 16, 32, 64, 96];
  if (seeds.length !== 30 || abrupt.length !== seeds.length * gpuSlots.length) {
    throw new Error("Expected the frozen 30-seed by 5-capacity abrupt sweep.");
  }
  if (actionability.payload.carryForward !== false) {
    throw new Error("The checked-in actionability decision unexpectedly changed.");
  }

  const startX = 246;
  const startY = 170;
  const cellWidth = 25;
  const cellHeight = 34;
  const columnPitch = 31;
  const rowPitch = 76;
  const rows = gpuSlots.map((slots, rowIndex) => {
    const records = abrupt
      .filter((record) => record.gpuSlots === slots)
      .sort((left, right) => left.seed - right.seed);
    if (
      records.length !== seeds.length ||
      records.some((record, index) => record.seed !== seeds[index])
    ) {
      throw new Error(`Capacity ${slots} does not cover the frozen seed inventory.`);
    }
    const y = startY + rowIndex * rowPitch;
    const changes = records.map(
      (record) => record.gatedPrimaryPercentChangeVsNoAction,
    );
    const actions = records.filter((record) => record.shadowDecision.act).length;
    const harmful = records.filter((record) => record.harmfulAction).length;
    const cells = records.map((record, columnIndex) => {
      const x = startX + columnIndex * columnPitch;
      const fill = heatmapFill(record.gatedPrimaryPercentChangeVsNoAction);
      const signed = `${record.gatedPrimaryPercentChangeVsNoAction > 0 ? "+" : ""}${record.gatedPrimaryPercentChangeVsNoAction.toFixed(2)}%`;
      return `<rect x="${x}" y="${y}" width="${cellWidth}" height="${cellHeight}" rx="3" fill="${fill.color}" fill-opacity="${fill.opacity.toFixed(3)}"><title>Seed ${record.seed}, ${slots} GPU slots: ${signed} modeled link-byte change versus no action</title></rect>`;
    });
    const rowMedian = median(changes);
    const signedMedian = `${rowMedian > 0 ? "+" : ""}${rowMedian.toFixed(2)}%`;
    return `
      <text x="70" y="${y + 23}" class="row-label">${slots} GPU slots</text>
      ${cells.join("")}
      <text x="1210" y="${y + 14}" class="summary-value">${signedMedian} median</text>
      <text x="1210" y="${y + 36}" class="small">${actions}/30 acted / ${harmful} harmful</text>`;
  });

  const seedTicks = seeds
    .map((seed, index) => ({ seed, index }))
    .filter(({ index }) => index % 5 === 0 || index === seeds.length - 1)
    .map(({ seed, index }) => {
      const x = startX + index * columnPitch + cellWidth / 2;
      return `<text x="${x}" y="146" text-anchor="middle" class="tick">${seed}</text>`;
    });
  const summary = actionability.payload.summary;
  if (
    summary.stationaryCells !== 30 ||
    summary.stationaryDetectorEvents !== 0 ||
    summary.stationaryGatedActions !== 0
  ) {
    throw new Error("The frozen stationary control summary unexpectedly changed.");
  }

  return `<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="650" viewBox="0 0 1500 650" role="img" aria-labelledby="capacity-title capacity-desc">
  <title id="capacity-title">StrataMoE actionability outcome by modeled GPU capacity and seed</title>
  <desc id="capacity-desc">A 5 by 30 matrix shows modeled link-byte change versus no action for every preregistered abrupt-shift seed. Savings occur at 8 and 64 slots, regressions occur at 32 slots, and the intervention takes no action at 16 and 96 slots. The candidate failed its carry-forward gates.</desc>
  <style>
    .title{font:700 32px ui-monospace,SFMono-Regular,Consolas,monospace;fill:#f2f3ed;letter-spacing:1px}.subtitle,.small,.footer{font:400 16px Inter,Segoe UI,sans-serif;fill:#91a4a5}.tick{font:400 13px ui-monospace,SFMono-Regular,Consolas,monospace;fill:#718789}.row-label,.summary-value{font:600 17px ui-monospace,SFMono-Regular,Consolas,monospace;fill:#e7ece8}.rule{stroke:#294044;stroke-width:1}.status{font:700 17px ui-monospace,SFMono-Regular,Consolas,monospace;fill:#ef6e63}
  </style>
  <rect width="1500" height="650" fill="#081113"/>
  <path d="M40 72H1460" class="rule"/>
  <text x="70" y="54" class="title">ACTIONABILITY / ALL 150 ABRUPT-SHIFT CELLS</text>
  <text x="1430" y="53" text-anchor="end" class="subtitle">first 64 post-shift tokens / modeled link bytes / lower is better</text>
  <text x="70" y="116" class="subtitle">capacity</text>
  <text x="246" y="116" class="subtitle">untouched seeds 4100-4129</text>
  <text x="1210" y="116" class="subtitle">row result</text>
  ${seedTicks.join("")}
  ${rows.join("")}
  <path d="M40 565H1460" class="rule"/>
  <rect x="70" y="590" width="20" height="20" rx="3" fill="#65d4d1"/><text x="102" y="606" class="footer">savings</text>
  <rect x="210" y="590" width="20" height="20" rx="3" fill="#243638"/><text x="242" y="606" class="footer">no action</text>
  <rect x="378" y="590" width="20" height="20" rx="3" fill="#ef6e63"/><text x="410" y="606" class="footer">regression</text>
  <text x="620" y="606" class="footer">stationary control / 30 seeds at 32 slots / 0 detector events / 0 actions</text>
  <text x="1430" y="606" text-anchor="end" class="status">carryForward = false</text>
</svg>
`;
}

const expectedOutputs = new Map<URL, string>([
  [overviewOutputUrl, renderSvg()],
  [capacityOutputUrl, renderCapacityMap()],
]);
const check = process.argv.includes("--check");
const normalizeLineEndings = (value: string) => value.replace(/\r\n/g, "\n");

if (check) {
  let stale = false;
  for (const [outputUrl, expected] of expectedOutputs) {
    let current = "";
    try {
      current = readFileSync(outputUrl, "utf8");
    } catch {
      // A missing file is reported by the same stale-evidence message below.
    }
    if (normalizeLineEndings(current) !== expected) {
      process.stderr.write(
        `README evidence is stale. Run npm run evidence:render (${fileURLToPath(outputUrl)}).\n`,
      );
      stale = true;
    }
  }
  if (stale) {
    process.exitCode = 1;
  } else {
    process.stdout.write("README evidence matches executable and checked-in results.\n");
  }
} else {
  for (const [outputUrl, expected] of expectedOutputs) {
    writeFileSync(outputUrl, expected, "utf8");
    process.stdout.write(`Wrote ${fileURLToPath(outputUrl)}\n`);
  }
}
