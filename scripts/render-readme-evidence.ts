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
  shadowDecision: { act: boolean };
};

type ActionabilityEvidence = {
  payload: {
    carryForward: boolean;
    records: { abrupt: ActionabilityRecord[] };
    summary: {
      abruptCells: number;
      executedActions: number;
      harmfulExecutedActions: number;
      harmfulExecutedActionRate: number;
      maximumRegressionPercent: number;
    };
  };
};

const outputUrl = new URL("../public/stratamoe-evidence.svg", import.meta.url);
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

const expected = renderSvg();
const check = process.argv.includes("--check");
const normalizeLineEndings = (value: string) => value.replace(/\r\n/g, "\n");

if (check) {
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
    process.exitCode = 1;
  } else {
    process.stdout.write("README evidence matches executable and checked-in results.\n");
  }
} else {
  writeFileSync(outputUrl, expected, "utf8");
  process.stdout.write(`Wrote ${fileURLToPath(outputUrl)}\n`);
}
