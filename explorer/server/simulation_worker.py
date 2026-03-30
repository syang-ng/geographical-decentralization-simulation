from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SERVER_DIR = Path(__file__).resolve().parent
EXPLORER_ROOT = SERVER_DIR.parent
REPO_ROOT = EXPLORER_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)

from consensus import ConsensusSettings
from distribution import parse_gcp_latency
from simulation import (
    DEFAULT_SIMULATION_SEED,
    SIMULATION_CACHE_DIRNAME,
    SIMULATION_CACHE_VERSION,
    SIMULATION_CODE_FILES,
    compute_simulation_cache_key,
    file_sha256,
    homogeneous_validators,
    homogeneous_validators_per_gcp,
    random_validators,
    simulation as run_simulation,
)
from source_agent import initialize_relays, initialize_signals

TEMPLATE_PATHS = {
    ("SSP", "homogeneous"): REPO_ROOT / "params" / "SSP-baseline.yaml",
    ("SSP", "latency-aligned"): REPO_ROOT / "params" / "SSP-latency-aligned.yaml",
    ("SSP", "latency-misaligned"): REPO_ROOT / "params" / "SSP-latency-misaligned.yaml",
    ("MSP", "homogeneous"): REPO_ROOT / "params" / "MSP-baseline.yaml",
    ("MSP", "latency-aligned"): REPO_ROOT / "params" / "MSP-latency-aligned.yaml",
    ("MSP", "latency-misaligned"): REPO_ROOT / "params" / "MSP-latency-misaligned.yaml",
}

ATTESTATION_CUTOFF_BY_SLOT_SECONDS = {
    6: 3000,
    8: 4000,
    12: 4000,
}

ARTIFACT_SPECS = (
    {
        "name": "avg_mev.json",
        "label": "Average MEV",
        "kind": "timeseries",
        "description": "Cumulative average MEV earned per slot.",
        "content_type": "application/json",
        "lazy": False,
        "renderable": True,
    },
    {
        "name": "supermajority_success.json",
        "label": "Supermajority Success",
        "kind": "timeseries",
        "description": "Cumulative successful supermajority rate across slots.",
        "content_type": "application/json",
        "lazy": False,
        "renderable": True,
    },
    {
        "name": "failed_block_proposals.json",
        "label": "Failed Block Proposals",
        "kind": "timeseries",
        "description": "Cumulative failed proposal count.",
        "content_type": "application/json",
        "lazy": False,
        "renderable": True,
    },
    {
        "name": "utility_increase.json",
        "label": "Utility Increase",
        "kind": "timeseries",
        "description": "Per-slot proposer utility increase after migration.",
        "content_type": "application/json",
        "lazy": True,
        "renderable": True,
    },
    {
        "name": "proposal_time_avg.json",
        "label": "Average Proposal Time",
        "kind": "timeseries",
        "description": "Per-slot average proposal time derived from the raw slot traces.",
        "content_type": "application/json",
        "lazy": True,
        "renderable": True,
    },
    {
        "name": "attestation_sum.json",
        "label": "Attestation Sum",
        "kind": "timeseries",
        "description": "Per-slot aggregate attestation values derived from the raw slot traces.",
        "content_type": "application/json",
        "lazy": True,
        "renderable": True,
    },
    {
        "name": "top_regions_final.json",
        "label": "Final Top Regions",
        "kind": "map",
        "description": "Final-slot region leaderboard derived from the raw geography trace.",
        "content_type": "application/json",
        "lazy": True,
        "renderable": True,
    },
    {
        "name": "proposal_time_by_slot.json",
        "label": "Proposal Time",
        "kind": "timeseries",
        "description": "Raw per-validator proposal timing traces.",
        "content_type": "application/json",
        "lazy": True,
        "renderable": False,
    },
    {
        "name": "attest_by_slot.json",
        "label": "Attestation Outcomes",
        "kind": "raw",
        "description": "Raw per-validator attestation traces.",
        "content_type": "application/json",
        "lazy": True,
        "renderable": False,
    },
    {
        "name": "region_counter_per_slot.json",
        "label": "Validator Geography Trace",
        "kind": "raw",
        "description": "Raw per-slot validator counts by GCP region.",
        "content_type": "application/json",
        "lazy": True,
        "renderable": False,
    },
    {
        "name": "region_profits.csv",
        "label": "Region Profits",
        "kind": "table",
        "description": "Per-region profit ledger exported by the simulator.",
        "content_type": "text/csv",
        "lazy": True,
        "renderable": False,
    },
    {
        "name": "action_reasons.csv",
        "label": "Action Reasons",
        "kind": "table",
        "description": "Migration action audit trail.",
        "content_type": "text/csv",
        "lazy": True,
        "renderable": False,
    },
)

REQUIRED_OUTPUTS = {
    "avg_mev.json",
    "supermajority_success.json",
    "failed_block_proposals.json",
    "utility_increase.json",
    "proposal_time_avg.json",
    "attestation_sum.json",
    "top_regions_final.json",
}

CACHE_ROOT = REPO_ROOT / SIMULATION_CACHE_DIRNAME

BUNDLE_SPECS = (
    {
        "bundle": "core-outcomes",
        "name": "overview_bundle_core-outcomes.json",
        "label": "Core outcomes",
        "description": "Prebuilt exact overview for MEV, supermajority success, and failed proposals.",
        "source_files": (
            "avg_mev.json",
            "supermajority_success.json",
            "failed_block_proposals.json",
        ),
    },
    {
        "bundle": "timing-and-attestation",
        "name": "overview_bundle_timing-and-attestation.json",
        "label": "Timing and attestations",
        "description": "Prebuilt exact overview for proposal latency and aggregate attestations.",
        "source_files": (
            "proposal_time_avg.json",
            "attestation_sum.json",
        ),
    },
    {
        "bundle": "geography-overview",
        "name": "overview_bundle_geography-overview.json",
        "label": "Geography overview",
        "description": "Prebuilt exact overview for final validator geography and top regions.",
        "source_files": (
            "top_regions_final.json",
        ),
    },
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


TEMPLATES = {key: load_yaml(path) for key, path in TEMPLATE_PATHS.items()}

GCP_REGIONS = pd.read_csv(REPO_ROOT / "data" / "gcp_regions.csv").copy()
GCP_REGIONS["gcp_region"] = GCP_REGIONS["Region"]
GCP_REGIONS["lat"] = GCP_REGIONS["Nearest City Latitude"]
GCP_REGIONS["lon"] = GCP_REGIONS["Nearest City Longitude"]
REGION_INDEX = {
    str(row["gcp_region"]): {
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "city": str(row["location"]).split(",")[0].strip(),
    }
    for _, row in GCP_REGIONS.iterrows()
}

RAW_GCP_LATENCY = pd.read_csv(REPO_ROOT / "data" / "gcp_latency.csv")
PARSED_GCP_LATENCY = parse_gcp_latency(RAW_GCP_LATENCY.copy())
VALIDATOR_DATA = pd.read_csv(REPO_ROOT / "data" / "validators.csv")

BASE_INPUT_HASHES = {
    "gcp_regions.csv": file_sha256(REPO_ROOT / "data" / "gcp_regions.csv"),
    "gcp_latency.csv": file_sha256(REPO_ROOT / "data" / "gcp_latency.csv"),
}
VALIDATOR_INPUT_HASH = file_sha256(REPO_ROOT / "data" / "validators.csv")
CODE_HASHES = {
    relative_path: file_sha256(REPO_ROOT / relative_path)
    for relative_path in SIMULATION_CODE_FILES
}


def write_response(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def round_to(value: float, digits: int) -> float:
    return round(float(value), digits)


def normalize_request_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "paradigm": raw_config.get("paradigm", "SSP"),
        "validators": int(raw_config.get("validators", 100)),
        "slots": int(raw_config.get("slots", 1000)),
        "distribution": raw_config.get("distribution", "homogeneous"),
        "sourcePlacement": raw_config.get("sourcePlacement", "homogeneous"),
        "migrationCost": round_to(float(raw_config.get("migrationCost", 0.0001)), 6),
        "attestationThreshold": round_to(float(raw_config.get("attestationThreshold", 2 / 3)), 6),
        "slotTime": int(raw_config.get("slotTime", 12)),
        "seed": int(raw_config.get("seed", DEFAULT_SIMULATION_SEED)),
    }


def select_template(config: dict[str, Any]) -> dict[str, Any]:
    key = (config["paradigm"], config["sourcePlacement"])
    try:
        return TEMPLATES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported paradigm/source placement combination: {key}") from exc


def derive_attestation_cutoff(slot_time_seconds: int) -> int:
    return ATTESTATION_CUTOFF_BY_SLOT_SECONDS.get(
        slot_time_seconds,
        min(slot_time_seconds * 500, 4000),
    )


def effective_validator_distribution(config: dict[str, Any]) -> str:
    if config["distribution"] == "uniform":
        return "homogeneous-gcp"
    return config["distribution"]


def load_info_profiles(template_config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    signal_profiles = initialize_signals(template_config.get("signal_profiles", []))
    relay_profiles = initialize_relays(template_config.get("relay_profiles", []))
    return signal_profiles, relay_profiles


def build_consensus_settings(template_config: dict[str, Any], config: dict[str, Any]) -> ConsensusSettings:
    consensus_data = dict(template_config.get("consensus_settings", {}))
    slot_duration_ms = int(config["slotTime"]) * 1000
    consensus = ConsensusSettings(**consensus_data)
    consensus.slot_duration_ms = slot_duration_ms
    consensus.attestation_time_ms = derive_attestation_cutoff(int(config["slotTime"]))
    consensus.attestation_threshold = float(config["attestationThreshold"])
    return consensus


def select_validators(config: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    requested_validators = int(config["validators"])
    seed = int(config["seed"])
    distribution = config["distribution"]

    random.seed(seed)
    np.random.seed(seed)

    if distribution in {"uniform", "homogeneous-gcp"}:
        validators = homogeneous_validators_per_gcp(GCP_REGIONS, requested_validators)
    elif distribution == "homogeneous":
        validators = homogeneous_validators(GCP_REGIONS, requested_validators)
    elif distribution == "random":
        validators = random_validators(GCP_REGIONS, requested_validators)
    elif distribution == "heterogeneous":
        if len(VALIDATOR_DATA) > requested_validators:
            validators = VALIDATOR_DATA.sample(n=requested_validators, random_state=seed).copy()
        else:
            validators = VALIDATOR_DATA.copy()
        validators = validators.reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported validator distribution: {distribution}")

    actual_validators = len(validators)
    return validators, actual_validators


def build_cache_context(
    config: dict[str, Any],
    template_config: dict[str, Any],
    consensus_settings: ConsensusSettings,
    number_of_validators: int,
) -> dict[str, Any]:
    timing_strategies = template_config.get("proposer_strategies", [{"type": "optimal_latency"}])
    location_strategies = template_config.get("location_strategies", [{"type": "best_relay"}])

    context = {
        "cache_version": SIMULATION_CACHE_VERSION,
        "explorer_backend_version": 1,
        "model": config["paradigm"],
        "num_slots": int(config["slots"]),
        "num_validators": number_of_validators,
        "distribution": effective_validator_distribution(config),
        "source_placement": config["sourcePlacement"],
        "time_window": int(template_config.get("time_window", 10000)),
        "fast_mode": False,
        "cost": float(config["migrationCost"]),
        "seed": int(config["seed"]),
        "consensus_settings": vars(consensus_settings),
        "timing_strategies": timing_strategies,
        "location_strategies": location_strategies,
        "config": {
            key: value
            for key, value in template_config.items()
            if key != "output_folder"
        },
        "input_hashes": dict(BASE_INPUT_HASHES),
        "code_hashes": CODE_HASHES,
    }
    if config["distribution"] == "heterogeneous":
        context["input_hashes"]["validators.csv"] = VALIDATOR_INPUT_HASH
    return context


def required_outputs_present(output_dir: Path) -> bool:
    return all((output_dir / name).is_file() for name in REQUIRED_OUTPUTS)


def is_up_to_date(target_path: Path, source_paths: list[Path]) -> bool:
    if not target_path.is_file():
        return False

    target_mtime = target_path.stat().st_mtime
    return all(source.is_file() and source.stat().st_mtime <= target_mtime for source in source_paths)


def maybe_gzip(path: Path) -> int | None:
    if path.suffix not in {".json", ".csv"}:
        return None

    gz_path = path.with_name(f"{path.name}.gz")
    if gz_path.exists() and gz_path.stat().st_mtime >= path.stat().st_mtime:
        return int(gz_path.stat().st_size)

    with path.open("rb") as source, gzip.open(gz_path, "wb", compresslevel=6) as target:
        shutil.copyfileobj(source, target)
    return int(gz_path.stat().st_size)


def build_time_series_block(title: str, label: str, values: list[float], y_label: str) -> dict[str, Any]:
    return {
        "type": "timeseries",
        "title": title,
        "xLabel": "Slot",
        "yLabel": y_label,
        "series": [
            {
                "label": label,
                "data": [
                    {
                        "x": index + 1,
                        "y": float(value) if isinstance(value, (int, float)) else 0.0,
                    }
                    for index, value in enumerate(values)
                ],
            }
        ],
    }


def read_numeric_series(path: Path) -> list[float]:
    values = load_json(path)
    if not isinstance(values, list):
        return []
    return [
        float(value) if isinstance(value, (int, float)) else 0.0
        for value in values
    ]


def build_overview_bundle_blocks(bundle: str, output_dir: Path) -> list[dict[str, Any]]:
    if bundle == "core-outcomes":
        return [
            build_time_series_block(
                "Average MEV Earned",
                "Average MEV",
                read_numeric_series(output_dir / "avg_mev.json"),
                "ETH",
            ),
            build_time_series_block(
                "Supermajority Success",
                "Supermajority Success",
                read_numeric_series(output_dir / "supermajority_success.json"),
                "Success Rate (%)",
            ),
            build_time_series_block(
                "Failed Block Proposals",
                "Failed Block Proposals",
                read_numeric_series(output_dir / "failed_block_proposals.json"),
                "Count",
            ),
        ]

    if bundle == "timing-and-attestation":
        return [
            build_time_series_block(
                "Average Proposal Time",
                "Average Proposal Time",
                read_numeric_series(output_dir / "proposal_time_avg.json"),
                "Milliseconds",
            ),
            build_time_series_block(
                "Aggregate Attestations",
                "Attestation Sum",
                read_numeric_series(output_dir / "attestation_sum.json"),
                "Aggregate Attestations",
            ),
        ]

    if bundle == "geography-overview":
        rows = load_json(output_dir / "top_regions_final.json")
        if not isinstance(rows, list):
            return []

        regions = []
        table_rows = []
        for entry in rows[:12]:
            if not isinstance(entry, list) or len(entry) < 2:
                continue

            name = str(entry[0])
            count = int(entry[1])
            table_rows.append([name, str(count)])

            region = REGION_INDEX.get(name)
            if not region:
                continue

            regions.append({
                "name": name,
                "lat": region["lat"],
                "lon": region["lon"],
                "value": count,
                "label": f"{name} - {region['city']}",
            })

        return [
            {
                "type": "map",
                "title": "Final Validator Geography",
                "colorScale": "density",
                "regions": regions,
            },
            {
                "type": "table",
                "title": "Top Final Regions",
                "headers": ["Region", "Validators"],
                "rows": table_rows,
                "highlight": [0, 1, 2],
            },
        ]

    return []


def ensure_summary_outputs(output_dir: Path) -> None:
    proposal_time_avg_path = output_dir / "proposal_time_avg.json"
    attestation_sum_path = output_dir / "attestation_sum.json"
    top_regions_final_path = output_dir / "top_regions_final.json"

    proposal_time_trace_path = output_dir / "proposal_time_by_slot.json"
    attest_trace_path = output_dir / "attest_by_slot.json"
    region_trace_path = output_dir / "region_counter_per_slot.json"

    if proposal_time_trace_path.is_file() and not is_up_to_date(proposal_time_avg_path, [proposal_time_trace_path]):
        proposal_time_by_slot = load_json(proposal_time_trace_path)
        proposal_time_avg = []
        for slot_values in proposal_time_by_slot:
            positive_values = [float(value) for value in slot_values if float(value) > 0]
            proposal_time_avg.append(
                (sum(positive_values) / len(positive_values))
                if positive_values
                else 0.0
            )
        with proposal_time_avg_path.open("w", encoding="utf-8") as handle:
            json.dump(proposal_time_avg, handle)

    if attest_trace_path.is_file() and not is_up_to_date(attestation_sum_path, [attest_trace_path]):
        attest_by_slot = load_json(attest_trace_path)
        attestation_sum = [
            sum(float(value) for value in slot_values)
            for slot_values in attest_by_slot
        ]
        with attestation_sum_path.open("w", encoding="utf-8") as handle:
            json.dump(attestation_sum, handle)

    if region_trace_path.is_file() and not is_up_to_date(top_regions_final_path, [region_trace_path]):
        region_counter_per_slot = load_json(region_trace_path)
        final_slot_key = None
        if region_counter_per_slot:
            final_slot_key = max(region_counter_per_slot.keys(), key=lambda raw_key: int(raw_key))
        top_regions_final = (
            region_counter_per_slot.get(final_slot_key, [])
            if final_slot_key is not None
            else []
        )
        with top_regions_final_path.open("w", encoding="utf-8") as handle:
            json.dump(top_regions_final, handle)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def last_numeric(series: list[Any]) -> float:
    if not series:
        return 0.0
    value = series[-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def ensure_overview_bundle_outputs(output_dir: Path) -> None:
    for spec in BUNDLE_SPECS:
        target_path = output_dir / spec["name"]
        source_paths = [output_dir / name for name in spec["source_files"]]
        if is_up_to_date(target_path, source_paths):
            continue

        blocks = build_overview_bundle_blocks(spec["bundle"], output_dir)
        with target_path.open("w", encoding="utf-8") as handle:
            json.dump(blocks, handle, separators=(",", ":"))


def ensure_derived_outputs(output_dir: Path) -> None:
    ensure_summary_outputs(output_dir)
    ensure_overview_bundle_outputs(output_dir)


def build_manifest(
    *,
    job_id: str,
    config: dict[str, Any],
    cache_key: str,
    output_dir: Path,
    cache_hit: bool,
    runtime_seconds: float,
    consensus_settings: ConsensusSettings,
) -> dict[str, Any]:
    avg_mev = load_json(output_dir / "avg_mev.json")
    supermajority_success = load_json(output_dir / "supermajority_success.json")
    failed_block_proposals = load_json(output_dir / "failed_block_proposals.json")
    utility_increase = load_json(output_dir / "utility_increase.json")
    top_regions_raw = load_json(output_dir / "top_regions_final.json")
    top_regions = [
        {"name": str(region), "count": int(count)}
        for region, count in top_regions_raw[:8]
    ]

    artifacts = []
    for spec in ARTIFACT_SPECS:
        artifact_path = output_dir / spec["name"]
        if not artifact_path.is_file():
            continue
        artifacts.append(
            {
                "name": spec["name"],
                "label": spec["label"],
                "kind": spec["kind"],
                "description": spec["description"],
                "contentType": spec["content_type"],
                "bytes": int(artifact_path.stat().st_size),
                "gzipBytes": maybe_gzip(artifact_path),
                "brotliBytes": None,
                "sha256": file_sha256(artifact_path),
                "lazy": spec["lazy"],
                "renderable": spec["renderable"],
            }
        )

    overview_bundles = []
    for spec in BUNDLE_SPECS:
        bundle_path = output_dir / spec["name"]
        if not bundle_path.is_file():
            continue
        overview_bundles.append(
            {
                "bundle": spec["bundle"],
                "name": spec["name"],
                "label": spec["label"],
                "description": spec["description"],
                "bytes": int(bundle_path.stat().st_size),
                "gzipBytes": maybe_gzip(bundle_path),
                "brotliBytes": None,
                "sha256": file_sha256(bundle_path),
            }
        )

    manifest = {
        "jobId": job_id,
        "configHash": cache_key,
        "cacheKey": cache_key,
        "cacheHit": cache_hit,
        "runtimeSeconds": round(float(runtime_seconds), 4),
        "outputDir": str(output_dir),
        "config": config,
        "summary": {
            "slotsRecorded": int(len(avg_mev)),
            "attestationCutoffMs": int(consensus_settings.attestation_time_ms),
            "finalAverageMev": round(last_numeric(avg_mev), 6),
            "finalSupermajoritySuccess": round(last_numeric(supermajority_success), 6),
            "finalFailedBlockProposals": round(last_numeric(failed_block_proposals), 6),
            "finalUtilityIncrease": round(last_numeric(utility_increase), 6),
            "topRegions": top_regions,
        },
        "artifacts": artifacts,
        "overviewBundles": overview_bundles,
    }

    with (output_dir / "explorer_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


def ensure_manifest(
    *,
    job_id: str,
    config: dict[str, Any],
    cache_key: str,
    output_dir: Path,
    cache_hit: bool,
    runtime_seconds: float,
    consensus_settings: ConsensusSettings,
) -> dict[str, Any]:
    manifest_path = output_dir / "explorer_manifest.json"
    effective_runtime = runtime_seconds
    if cache_hit and manifest_path.is_file() and runtime_seconds <= 0:
        existing_manifest = load_json(manifest_path)
        effective_runtime = float(existing_manifest.get("runtimeSeconds", 0.0))

    return build_manifest(
        job_id=job_id,
        config=config,
        cache_key=cache_key,
        output_dir=output_dir,
        cache_hit=cache_hit,
        runtime_seconds=effective_runtime,
        consensus_settings=consensus_settings,
    )


def run_job(job_id: str, raw_config: dict[str, Any]) -> dict[str, Any]:
    config = normalize_request_config(raw_config)

    template_config = select_template(config)
    consensus_settings = build_consensus_settings(template_config, config)
    timing_strategies = template_config.get("proposer_strategies", [{"type": "optimal_latency"}])
    location_strategies = template_config.get("location_strategies", [{"type": "best_relay"}])
    signal_profiles, relay_profiles = load_info_profiles(template_config)
    validators, actual_validator_count = select_validators(config)

    if actual_validator_count != config["validators"]:
        config["validators"] = actual_validator_count

    cache_context = build_cache_context(
        config=config,
        template_config=template_config,
        consensus_settings=consensus_settings,
        number_of_validators=actual_validator_count,
    )
    cache_key = compute_simulation_cache_key(cache_context)
    cache_dir = CACHE_ROOT / cache_key

    if cache_dir.exists():
        ensure_derived_outputs(cache_dir)

    if required_outputs_present(cache_dir):
        return ensure_manifest(
            job_id=job_id,
            config=config,
            cache_key=cache_key,
            output_dir=cache_dir,
            cache_hit=True,
            runtime_seconds=0.0,
            consensus_settings=consensus_settings,
        )

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    temp_output_dir = Path(tempfile.mkdtemp(prefix=f"{cache_key}-", dir=CACHE_ROOT))

    runtime_seconds = 0.0
    captured_logs = io.StringIO()

    try:
        start_time = time.perf_counter()
        with contextlib.redirect_stdout(captured_logs), contextlib.redirect_stderr(captured_logs):
            run_simulation(
                model=config["paradigm"],
                number_of_validators=actual_validator_count,
                num_slots=int(config["slots"]),
                validators=validators,
                gcp_regions=GCP_REGIONS,
                gcp_latency=PARSED_GCP_LATENCY,
                consensus_settings=consensus_settings,
                relay_profiles=relay_profiles,
                signal_profiles=signal_profiles,
                timing_strategies=timing_strategies,
                location_strategies=location_strategies,
                simulation_name=template_config.get("simulation_name", "Explorer exact mode"),
                output_folder=str(temp_output_dir),
                time_window=int(template_config.get("time_window", 10000)),
                fast_mode=False,
                cost=float(config["migrationCost"]),
                seed=int(config["seed"]),
                collect_full_history=False,
                export_raw_artifacts=False,
                verbose=False,
            )
        runtime_seconds = time.perf_counter() - start_time

        ensure_derived_outputs(temp_output_dir)
        build_manifest(
            job_id=job_id,
            config=config,
            cache_key=cache_key,
            output_dir=temp_output_dir,
            cache_hit=False,
            runtime_seconds=runtime_seconds,
            consensus_settings=consensus_settings,
        )

        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        shutil.move(str(temp_output_dir), str(cache_dir))

        return ensure_manifest(
            job_id=job_id,
            config=config,
            cache_key=cache_key,
            output_dir=cache_dir,
            cache_hit=False,
            runtime_seconds=runtime_seconds,
            consensus_settings=consensus_settings,
        )
    except Exception:
        captured_logs_value = captured_logs.getvalue().strip()
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir, ignore_errors=True)
        raise RuntimeError(
            "\n".join(
                part
                for part in [
                    "Exact simulation job failed.",
                    captured_logs_value,
                ]
                if part
            )
        )


def main() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            request_type = request.get("type")
            payload = request.get("payload", {})

            if request_type != "run":
                raise ValueError(f"Unsupported request type: {request_type}")

            result = run_job(
                job_id=str(payload.get("job_id", f"worker-{request_id}")),
                raw_config=payload.get("config", {}),
            )
            write_response({
                "id": request_id,
                "ok": True,
                "result": result,
            })
        except Exception as exc:
            write_response({
                "id": request_id,
                "ok": False,
                "error": {
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            })


if __name__ == "__main__":
    main()
