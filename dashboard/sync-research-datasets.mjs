import fs from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";

// Maintenance-only helper.
// The MVP uses the checked-in dataset payloads under dashboard/simulations.

const ROOT = process.cwd();
const DASHBOARD_DIR = path.join(ROOT, "dashboard");
const CATALOG_PATH = path.join(DASHBOARD_DIR, "assets", "research-catalog.js");
const REMOTE_ROOT = "https://geo-decentralization.github.io/simulations";

function buildRemoteUrl(entry) {
  return `${REMOTE_ROOT}/${entry.evaluation}/${entry.paradigm}/${entry.result}/data.json`;
}

async function loadCatalog() {
  const source = await fs.readFile(CATALOG_PATH, "utf8");
  const sandbox = { window: {} };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox);
  const catalog = sandbox.window.RESEARCH_CATALOG;
  if (!catalog || !Array.isArray(catalog.datasets)) {
    throw new Error("Failed to load research catalog");
  }
  return catalog;
}

async function syncEntry(entry) {
  const remoteUrl = buildRemoteUrl(entry);
  const response = await fetch(remoteUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${remoteUrl}: ${response.status} ${response.statusText}`);
  }

  const body = await response.text();
  const outputPath = path.join(DASHBOARD_DIR, entry.path);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, body, "utf8");

  return {
    localPath: entry.path,
    remoteUrl
  };
}

async function main() {
  const catalog = await loadCatalog();
  const synced = [];

  for (const entry of catalog.datasets) {
    synced.push(await syncEntry(entry));
  }

  process.stdout.write(`${JSON.stringify({ syncedCount: synced.length, synced }, null, 2)}\n`);
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
