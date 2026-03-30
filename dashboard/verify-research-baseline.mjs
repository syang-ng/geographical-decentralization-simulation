import fs from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";

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

async function compareEntry(entry) {
  const localPath = path.join(DASHBOARD_DIR, entry.path);
  const remoteUrl = buildRemoteUrl(entry);

  const [localBody, remoteBody] = await Promise.all([
    fs.readFile(localPath, "utf8"),
    fetch(remoteUrl).then(async response => {
      if (!response.ok) {
        throw new Error(`Failed to fetch ${remoteUrl}: ${response.status} ${response.statusText}`);
      }
      return response.text();
    })
  ]);

  return {
    path: entry.path,
    remoteUrl,
    matches: localBody === remoteBody
  };
}

async function main() {
  const catalog = await loadCatalog();
  const results = [];

  for (const entry of catalog.datasets) {
    results.push(await compareEntry(entry));
  }

  const mismatches = results.filter(result => !result.matches);
  process.stdout.write(`${JSON.stringify({
    checked: results.length,
    mismatches
  }, null, 2)}\n`);

  if (mismatches.length > 0) {
    process.exitCode = 1;
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
