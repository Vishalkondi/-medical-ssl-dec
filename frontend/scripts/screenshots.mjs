// Capture README screenshots of the running app (frontend on :5173, backend on :8000).
// Uses the locally installed Microsoft Edge through playwright-core (no browser download).
//
//   node scripts/screenshots.mjs <synthetic-image.png> <non-image-file>
//
// Only use synthetic or public images: screenshots are committed to the repo.
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const [, , sampleImage, badFile] = process.argv;
if (!sampleImage || !badFile) {
  console.error("usage: node scripts/screenshots.mjs <synthetic-image.png> <non-image-file>");
  process.exit(1);
}
const URL = process.env.APP_URL ?? "http://localhost:5173/";
const OUT = resolve(dirname(fileURLToPath(import.meta.url)), "../../docs/screenshots");
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ channel: "msedge" });

async function page({ scheme, width = 1280, height = 820 }) {
  const ctx = await browser.newContext({ viewport: { width, height }, colorScheme: scheme, deviceScaleFactor: 1.5 });
  const p = await ctx.newPage();
  await p.goto(URL);
  await p.getByText("No trained model", { exact: false }).first().waitFor();
  return p;
}

async function shot(p, name) {
  await p.evaluate(() => window.scrollTo(0, 0));
  await p.waitForTimeout(400); // let transitions settle
  await p.screenshot({ path: `${OUT}/${name}.png` });
  console.log("saved", name);
}

let p = await page({ scheme: "dark" });
await shot(p, "analyze-dark");

p = await page({ scheme: "light" });
await shot(p, "analyze-light");

await p.locator('input[type="file"]').setInputFiles(sampleImage);
await p.getByRole("button", { name: /Analyze image/ }).click();
await p.getByText("no model is loaded", { exact: false }).waitFor();
await shot(p, "upload-validated");

p = await page({ scheme: "dark" });
await p.locator('input[type="file"]').setInputFiles(badFile);
await p.getByRole("alert").first().waitFor();
await shot(p, "invalid-file");

p = await page({ scheme: "light" });
await p.getByRole("button", { name: "Model card" }).click();
await p.getByText("Intended use and limitations").waitFor();
await p.screenshot({ path: `${OUT}/model-card.png`, fullPage: true });
console.log("saved model-card");

p = await page({ scheme: "dark", width: 390, height: 844 });
await p.locator('input[type="file"]').setInputFiles(sampleImage);
await shot(p, "mobile");

await browser.close();
