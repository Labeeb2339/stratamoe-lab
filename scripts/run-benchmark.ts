import { buildFixedBenchmark } from "./benchmark-fixture";

const output = buildFixedBenchmark();

process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
