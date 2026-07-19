import assert from "node:assert/strict";
import { access, readFile, stat } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the StrataMoE research workbench", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>StrataMoE Lab/i);
  assert.match(html, /Can a shift-aware cache move fewer expert bytes/i);
  assert.match(html, /Detection passed\. Action safety did not\./i);
  assert.match(html, /16 \/ 76/i);
  assert.match(html, /Configure once\. Compare all three policies\./i);
  assert.match(html, /A falsifiable simulator, not a hardware claim\./i);
  assert.match(html, /LRU/i);
  assert.match(html, /LFU/i);
  assert.match(html, /ShiftCache/i);
  assert.match(html, /Import router trace JSON/i);
  assert.match(html, /Export router trace/i);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|Building your site/i);
});

test("removes starter-only surfaces and ships project metadata", async () => {
  const [layout, page, packageJson, socialCard] = await Promise.all([
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    stat(new URL("../public/og.png", import.meta.url)),
  ]);

  assert.match(layout, /StrataMoE Lab/);
  assert.match(layout, /openGraph/);
  assert.match(layout, /twitter/);
  assert.match(layout, /new URL\("\/og\.png", origin\)/);
  assert.match(page, /ExperimentWorkbench/);
  assert.match(packageJson, /"name": "stratamoe-lab"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton|drizzle-orm|drizzle-kit/);
  assert.ok(socialCard.size > 100_000, "social card should be a real raster asset");

  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
  await assert.rejects(access(new URL("../db/schema.ts", import.meta.url)));
  await assert.rejects(access(new URL("../examples/d1/db/schema.ts", import.meta.url)));
});
