import argparse
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
import time

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_HOME = DEFAULT_REPO_ROOT / "tests"
ARTIFACT_ROOT = BENCHMARK_HOME / ".artifacts" / "simulation_benchmarks"
RESULTS_PATH = BENCHMARK_HOME / "performance_results.json"
BASELINE_PATH = BENCHMARK_HOME / "performance_baselines.snapshot"

OUTPUT_FILES = (
    "action_reasons.csv",
    "attest_by_slot.json",
    "avg_mev.json",
    "estimated_mev_by_slot.json",
    "failed_block_proposals.json",
    "mev_by_slot.json",
    "proposal_time_by_slot.json",
    "proposer_strategy_and_mev.json",
    "region_counter_per_slot.json",
    "region_profits.csv",
    "relay_names.json",
    "signal_names.json",
    "supermajority_success.json",
    "utility_increase.json",
)


@dataclass(frozen=True)
class BenchmarkScenario:
    name: str
    model: str
    slots: int
    validators: int
    distribution: str
    info_distribution: str
    seed: int
    gamma: float = 0.6667
    delta: int = 12000
    cutoff: int = 4000
    time_window: int = 10
    cost: float = 0.0001


SCENARIOS = (
    BenchmarkScenario(
        name="ssp_homogeneous_small",
        model="SSP",
        slots=4,
        validators=48,
        distribution="homogeneous",
        info_distribution="homogeneous",
        seed=123,
    ),
    BenchmarkScenario(
        name="msp_homogeneous_medium",
        model="MSP",
        slots=4,
        validators=48,
        distribution="homogeneous",
        info_distribution="homogeneous",
        seed=123,
    ),
    BenchmarkScenario(
        name="ssp_heterogeneous_medium",
        model="SSP",
        slots=6,
        validators=96,
        distribution="heterogeneous",
        info_distribution="homogeneous",
        seed=123,
    ),
)


def scenario_names():
    return [scenario.name for scenario in SCENARIOS]


def resolve_repo_root(repo_root):
    if repo_root is None:
        return DEFAULT_REPO_ROOT
    return Path(repo_root).resolve()


def config_path_for(repo_root):
    return repo_root / "params" / "basic-config.yaml"


def supported_cli_flags(repo_root):
    simulation_path = repo_root / "simulation.py"
    source = simulation_path.read_text(encoding="utf-8")
    supported = set()
    for flag in ("--fast", "--full-history", "--cache-results", "--verbose"):
        if flag in source:
            supported.add(flag)
    return supported


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_file_sha256(relative_name, path):
    if relative_name == "region_profits.csv":
        dataframe = pd.read_csv(path)
        if not dataframe.empty:
            dataframe = dataframe.sort_values(
                by=sorted(dataframe.columns)
            ).reset_index(drop=True)
        normalized = dataframe.to_csv(index=False, lineterminator="\n").encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    return file_sha256(path)


def build_output_hashes(output_dir):
    aggregate = hashlib.sha256()
    file_hashes = {}

    for relative_name in OUTPUT_FILES:
        file_path = output_dir / relative_name
        if not file_path.exists():
            raise FileNotFoundError(f"Expected benchmark output missing: {file_path}")
        file_hash = normalized_file_sha256(relative_name, file_path)
        file_hashes[relative_name] = file_hash
        aggregate.update(relative_name.encode("utf-8"))
        aggregate.update(file_hash.encode("utf-8"))

    return file_hashes, aggregate.hexdigest()


def build_summary(output_dir):
    avg_mev = load_json(output_dir / "avg_mev.json")
    supermajority = load_json(output_dir / "supermajority_success.json")
    failed_block_proposals = load_json(output_dir / "failed_block_proposals.json")
    utility_increase = load_json(output_dir / "utility_increase.json")
    proposer_strategy_and_mev = load_json(output_dir / "proposer_strategy_and_mev.json")
    region_counter_per_slot = load_json(output_dir / "region_counter_per_slot.json")

    return {
        "slots_recorded": len(avg_mev),
        "final_avg_mev": avg_mev[-1] if avg_mev else None,
        "final_supermajority_success": supermajority[-1] if supermajority else None,
        "final_failed_block_proposals": (
            failed_block_proposals[-1] if failed_block_proposals else None
        ),
        "final_utility_increase": utility_increase[-1] if utility_increase else None,
        "proposer_records": len(proposer_strategy_and_mev),
        "region_slots": len(region_counter_per_slot),
    }


def parse_reported_elapsed_s(stdout):
    match = re.search(r"Simulation completed in ([0-9]+(?:\.[0-9]+)?) seconds\.", stdout)
    if match is None:
        return None
    return round(float(match.group(1)), 3)


def run_command(scenario, output_dir, repo_root):
    supported_flags = supported_cli_flags(repo_root)
    command = [
        sys.executable,
        "simulation.py",
        "--config",
        str(config_path_for(repo_root)),
        "--model",
        scenario.model,
        "--slots",
        str(scenario.slots),
        "--validators",
        str(scenario.validators),
        "--distribution",
        scenario.distribution,
        "--info-distribution",
        scenario.info_distribution,
        "--seed",
        str(scenario.seed),
        "--gamma",
        str(scenario.gamma),
        "--delta",
        str(scenario.delta),
        "--cutoff",
        str(scenario.cutoff),
        "--time_window",
        str(scenario.time_window),
        "--cost",
        str(scenario.cost),
        "--output-dir",
        str(output_dir),
    ]

    if "--fast" in supported_flags:
        command.append("--no-fast")
    if "--cache-results" in supported_flags:
        command.append("--no-cache-results")
    if "--full-history" in supported_flags:
        command.append("--no-full-history")
    if "--verbose" in supported_flags:
        command.append("--no-verbose")

    return subprocess.run(
        command,
        cwd=repo_root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def run_scenario(scenario, repeat=1, repo_root=None, artifact_label="current"):
    repo_root = resolve_repo_root(repo_root)
    scenario_root = ARTIFACT_ROOT / artifact_label / scenario.name
    if scenario_root.exists():
        shutil.rmtree(scenario_root)
    scenario_root.mkdir(parents=True, exist_ok=True)

    runs = []
    for run_index in range(1, repeat + 1):
        output_dir = scenario_root / f"run_{run_index}"
        start = time.perf_counter()
        completed = run_command(scenario, output_dir, repo_root)
        elapsed_s = time.perf_counter() - start

        if completed.returncode != 0:
            raise RuntimeError(
                f"Scenario '{scenario.name}' failed on run {run_index}.\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        file_hashes, aggregate_hash = build_output_hashes(output_dir)
        reported_elapsed_s = parse_reported_elapsed_s(completed.stdout)
        runs.append(
            {
                "run_index": run_index,
                "wall_elapsed_s": round(elapsed_s, 3),
                "reported_elapsed_s": reported_elapsed_s,
                "aggregate_output_hash": aggregate_hash,
                "file_hashes": file_hashes,
                "summary": build_summary(output_dir),
            }
        )

    aggregate_hashes = {run["aggregate_output_hash"] for run in runs}
    if len(aggregate_hashes) != 1:
        raise AssertionError(
            f"Scenario '{scenario.name}' produced non-repeatable outputs across runs: {sorted(aggregate_hashes)}"
        )

    wall_timings = [run["wall_elapsed_s"] for run in runs]
    reported_timings = [
        run["reported_elapsed_s"]
        for run in runs
        if run["reported_elapsed_s"] is not None
    ]

    result = {
        "scenario": asdict(scenario),
        "repeat": repeat,
        "aggregate_output_hash": runs[0]["aggregate_output_hash"],
        "file_hashes": runs[0]["file_hashes"],
        "summary": runs[0]["summary"],
        "wall_timings_s": wall_timings,
        "min_wall_elapsed_s": min(wall_timings),
        "median_wall_elapsed_s": round(statistics.median(wall_timings), 3),
        "max_wall_elapsed_s": max(wall_timings),
        "runs": runs,
    }

    if reported_timings:
        result["reported_timings_s"] = reported_timings
        result["min_reported_elapsed_s"] = min(reported_timings)
        result["median_reported_elapsed_s"] = round(statistics.median(reported_timings), 3)
        result["max_reported_elapsed_s"] = max(reported_timings)

    return result


def run_benchmarks(selected_names=None, repeat=1, repo_root=None, artifact_label="current"):
    repo_root = resolve_repo_root(repo_root)
    selected = []
    allowed = set(selected_names or scenario_names())
    for scenario in SCENARIOS:
        if scenario.name in allowed:
            selected.append(scenario)

    if not selected:
        raise ValueError("No benchmark scenarios selected.")

    results = {}
    for scenario in selected:
        results[scenario.name] = run_scenario(
            scenario,
            repeat=repeat,
            repo_root=repo_root,
            artifact_label=artifact_label,
        )
    return results


def load_baselines():
    return load_json(BASELINE_PATH)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def print_results(results):
    for name, result in results.items():
        print(
            f"{name}: wall_median={result['median_wall_elapsed_s']:.3f}s "
            f"wall_min={result['min_wall_elapsed_s']:.3f}s "
            f"wall_max={result['max_wall_elapsed_s']:.3f}s "
            f"hash={result['aggregate_output_hash']}"
        )
        if "median_reported_elapsed_s" in result:
            print(
                f"  sim_median={result['median_reported_elapsed_s']:.3f}s "
                f"sim_min={result['min_reported_elapsed_s']:.3f}s "
                f"sim_max={result['max_reported_elapsed_s']:.3f}s"
            )
        summary = result["summary"]
        print(
            f"  slots={summary['slots_recorded']} "
            f"avg_mev={summary['final_avg_mev']} "
            f"supermajority={summary['final_supermajority_success']} "
            f"failed={summary['final_failed_block_proposals']}"
        )


def compare_results(reference_results, candidate_results, metric="reported"):
    comparisons = {}
    for name, reference in reference_results.items():
        candidate = candidate_results[name]
        if metric == "reported":
            metric_key = "median_reported_elapsed_s"
            metric_label = "reported"
        else:
            metric_key = "median_wall_elapsed_s"
            metric_label = "wall"

        reference_median = reference.get(metric_key)
        candidate_median = candidate.get(metric_key)
        improvement_fraction = 0.0
        if reference_median is not None and candidate_median is not None and reference_median > 0:
            improvement_fraction = (reference_median - candidate_median) / reference_median

        comparisons[name] = {
            "metric": metric_label,
            "reference_median_s": reference_median,
            "candidate_median_s": candidate_median,
            "improvement_fraction": improvement_fraction,
            "improvement_percent": round(improvement_fraction * 100, 2),
            "hash_match": (
                reference["aggregate_output_hash"] == candidate["aggregate_output_hash"]
            ),
            "summary_match": reference["summary"] == candidate["summary"],
        }

    return comparisons


def print_comparisons(comparisons, reference_label, candidate_label):
    for name, comparison in comparisons.items():
        print(
            f"{name}: metric={comparison['metric']} "
            f"{reference_label}={comparison['reference_median_s']:.3f}s "
            f"{candidate_label}={comparison['candidate_median_s']:.3f}s "
            f"improvement={comparison['improvement_percent']:.2f}% "
            f"hash_match={comparison['hash_match']} "
            f"summary_match={comparison['summary_match']}"
        )


def compare_to_baselines(results, baselines):
    mismatches = []
    baseline_entries = baselines.get("scenarios", {})
    for name, result in results.items():
        baseline = baseline_entries.get(name)
        if baseline is None:
            mismatches.append(f"{name}: missing baseline entry")
            continue
        if baseline["aggregate_output_hash"] != result["aggregate_output_hash"]:
            mismatches.append(
                f"{name}: output hash changed "
                f"{baseline['aggregate_output_hash']} -> {result['aggregate_output_hash']}"
            )
        if baseline["summary"] != result["summary"]:
            mismatches.append(f"{name}: summary changed")
    return mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Run repeatable local simulation benchmarks and compare outputs to stored baselines."
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=scenario_names(),
        help="Scenario name to run. Repeat to run more than one. Defaults to all scenarios.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repeats per scenario (default: 1).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write the current output hashes and summaries to the baseline file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if the current outputs differ from the stored baselines.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root containing simulation.py and params/basic-config.yaml (default: current repo).",
    )
    parser.add_argument(
        "--artifact-label",
        type=str,
        default="current",
        help="Artifact subdirectory label for the primary benchmark run (default: current).",
    )
    parser.add_argument(
        "--compare-repo-root",
        type=Path,
        help="Optional second repository root to benchmark for before/after comparisons.",
    )
    parser.add_argument(
        "--compare-artifact-label",
        type=str,
        default="comparison",
        help="Artifact subdirectory label for the comparison benchmark run (default: comparison).",
    )
    parser.add_argument(
        "--min-improvement-pct",
        type=float,
        default=None,
        help="Optional minimum required speedup percentage for comparison runs.",
    )
    parser.add_argument(
        "--compare-metric",
        choices=("reported", "wall"),
        default="reported",
        help="Metric to use for performance comparisons (default: reported simulation runtime).",
    )
    args = parser.parse_args()

    repo_root = resolve_repo_root(args.repo_root)
    results = run_benchmarks(
        selected_names=args.scenario,
        repeat=args.repeat,
        repo_root=repo_root,
        artifact_label=args.artifact_label,
    )
    payload = {
        "repo_root": str(repo_root),
        "config_path": str(config_path_for(repo_root).relative_to(repo_root)),
        "results": results,
    }
    write_json(RESULTS_PATH, payload)
    print_results(results)

    comparisons = None
    if args.compare_repo_root:
        comparison_root = resolve_repo_root(args.compare_repo_root)
        comparison_results = run_benchmarks(
            selected_names=args.scenario,
            repeat=args.repeat,
            repo_root=comparison_root,
            artifact_label=args.compare_artifact_label,
        )
        comparisons = compare_results(
            comparison_results,
            results,
            metric=args.compare_metric,
        )
        print_comparisons(comparisons, "reference", "candidate")

    if args.update_baseline:
        baseline_payload = {
            "repo_root": str(repo_root),
            "config_path": str(config_path_for(repo_root).relative_to(repo_root)),
            "scenarios": {
                name: {
                    "aggregate_output_hash": result["aggregate_output_hash"],
                    "file_hashes": result["file_hashes"],
                    "summary": result["summary"],
                    "scenario": result["scenario"],
                }
                for name, result in results.items()
            },
        }
        write_json(BASELINE_PATH, baseline_payload)
        print(f"Updated baseline: {BASELINE_PATH}")

    if args.strict:
        baselines = load_baselines()
        mismatches = compare_to_baselines(results, baselines)
        if mismatches:
            for mismatch in mismatches:
                print(mismatch)
            raise SystemExit(1)

    if comparisons is not None:
        failures = []
        for name, comparison in comparisons.items():
            if not comparison["hash_match"]:
                failures.append(f"{name}: output hash changed during comparison")
            if not comparison["summary_match"]:
                failures.append(f"{name}: summary changed during comparison")
            if (
                args.min_improvement_pct is not None
                and comparison["reference_median_s"] is not None
                and comparison["candidate_median_s"] is not None
                and comparison["improvement_percent"] < args.min_improvement_pct
            ):
                failures.append(
                    f"{name}: improvement {comparison['improvement_percent']:.2f}% is below "
                    f"required {args.min_improvement_pct:.2f}%"
                )
        if failures:
            for failure in failures:
                print(failure)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
