import argparse
import dataclasses
import hashlib
import json
import numpy as np
import os
import pandas as pd
import random
import shutil
import time
import traceback
import yaml  # Import yaml library
from collections import defaultdict, Counter

from constants import LinearMEVUtility
from consensus import ConsensusSettings
from distribution import parse_gcp_latency
from models import SingleSourceParadigm, MultiSourceParadigm
from source_agent import initialize_relays, initialize_signals


DEFAULT_SIMULATION_SEED = 0x06511
SIMULATION_CACHE_VERSION = 1
SIMULATION_CACHE_DIRNAME = ".simulation_cache"
SIMULATION_CODE_FILES = (
    "simulation.py",
    "models.py",
    "validator_agent.py",
    "source_agent.py",
    "distribution.py",
    "consensus.py",
    "constants.py",
)


def normalize_for_hash(value):
    if dataclasses.is_dataclass(value):
        return normalize_for_hash(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): normalize_for_hash(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [normalize_for_hash(v) for v in value]
    if isinstance(value, set):
        return sorted(normalize_for_hash(v) for v in value)
    if isinstance(value, pd.DataFrame):
        return dataframe_hash(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        callable_payload = {
            "name": getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__)),
        }
        code = getattr(value, "__code__", None)
        if code is not None:
            callable_payload["code"] = {
                "co_code": code.co_code.hex(),
                "co_consts": normalize_for_hash(code.co_consts),
                "co_names": list(code.co_names),
                "co_varnames": list(code.co_varnames),
            }
        defaults = getattr(value, "__defaults__", None)
        if defaults is not None:
            callable_payload["defaults"] = normalize_for_hash(defaults)
        closure = getattr(value, "__closure__", None)
        if closure:
            callable_payload["closure"] = [
                normalize_for_hash(cell.cell_contents) for cell in closure
            ]
        return {"callable": callable_payload}
    if hasattr(value, "name") and hasattr(value, "value"):
        return {"enum": value.__class__.__name__, "name": value.name}
    return value


def dataframe_hash(df):
    hasher = hashlib.sha256()
    hasher.update(json.dumps(list(df.columns)).encode("utf-8"))
    hasher.update(json.dumps([str(dtype) for dtype in df.dtypes]).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(df, index=True, categorize=True).to_numpy(
        dtype=np.uint64
    )
    hasher.update(row_hashes.tobytes())
    return hasher.hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_simulation_cache_key(cache_context):
    normalized = normalize_for_hash(cache_context)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def copy_directory_contents(source_dir, destination_dir, exclude_names=None):
    exclude_names = set(exclude_names or [])
    os.makedirs(destination_dir, exist_ok=True)
    for entry in os.listdir(source_dir):
        if entry in exclude_names:
            continue
        source_path = os.path.join(source_dir, entry)
        destination_path = os.path.join(destination_dir, entry)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination_path)


def build_export_payloads(model_standard):
    avg_mev_series = [
        slot["Average_MEV_Earned"] for slot in model_standard.slot_model_history
    ]
    supermaj_series = [
        slot["Supermajority_Success_Rate"] for slot in model_standard.slot_model_history
    ]
    failed_block_proposals = [
        slot["Failed_Block_Proposals"] for slot in model_standard.slot_model_history
    ]
    utility_increase_series = [
        slot["Utility_Increase"] for slot in model_standard.slot_model_history
    ]

    mev_by_slot = list(getattr(model_standard, "slot_mev_by_slot", []))
    estimated_mev_by_slot = list(getattr(model_standard, "slot_estimated_mev_by_slot", []))
    attest_by_slot = list(getattr(model_standard, "slot_attest_by_slot", []))
    proposal_time_by_slot = list(getattr(model_standard, "slot_proposal_time_by_slot", []))
    proposal_time_avg = list(getattr(model_standard, "slot_proposal_time_avg", []))
    attestation_sum = list(getattr(model_standard, "slot_attestation_sum", []))
    region_counter_per_slot = dict(getattr(model_standard, "slot_region_counter_per_slot", {}))
    top_regions_final = list(getattr(model_standard, "top_regions_final", []))

    return {
        "avg_mev": avg_mev_series,
        "supermajority_success": supermaj_series,
        "failed_block_proposals": failed_block_proposals,
        "utility_increase": utility_increase_series,
        "mev_by_slot": mev_by_slot,
        "estimated_mev_by_slot": estimated_mev_by_slot,
        "attest_by_slot": attest_by_slot,
        "proposal_time_by_slot": proposal_time_by_slot,
        "proposal_time_avg": proposal_time_avg,
        "attestation_sum": attestation_sum,
        "proposer_strategy_and_mev": list(model_standard.slot_proposer_history),
        "region_counter_per_slot": region_counter_per_slot,
        "top_regions_final": top_regions_final,
    }


# --- Simulation Initialization Functions ---

def load_simulation_config(config_file_path):
    """Loads and parses the simulation's YAML configuration file."""
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(
            f"Configuration file '{config_file_path}' not found. Please ensure the file exists."
        )

    try:
        with open(config_file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        print(f"Successfully loaded configuration from: {config_file_path}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unknown error loading configuration file: {e}")


def initialize_consensus_settings(config_data):
    """Initializes a ConsensusSettings instance from configuration data."""
    consensus_settings_data = config_data.get("consensus_settings", {})
    return ConsensusSettings(**consensus_settings_data)


def random_validators(gcp_regions, number_of_validators):
    """Generates a list of validators with random GCP region assignments."""
    gcp_data = [(region["gcp_region"], region["lat"], region["lon"]) for _, region in gcp_regions.iterrows()]
    validators = [random.choice(gcp_data) for _ in range(number_of_validators)]

    return pd.DataFrame(validators, columns=["gcp_region", "latitude", "longitude"])


def homogeneous_validators_per_gcp(gcp_regions, number_of_validators):
    """Generates a list of validators evenly distributed across GCP regions."""
    gcp_data = [(region["gcp_region"], region["lat"], region["lon"]) for _, region in gcp_regions.iterrows()]
    num_regions = len(gcp_data)
    validators = [gcp_data[i % num_regions] for i in range(number_of_validators)]

    return pd.DataFrame(validators, columns=["gcp_region", "latitude", "longitude"])


def homogeneous_validators(gcp_regions, number_of_validators):
    """Generates a list of validators evenly distributed across major GCP regions."""
    gcp_data = [(region["gcp_region"].split("-")[0], region["gcp_region"], region["lat"], region["lon"]) for _, region in gcp_regions.iterrows()]
    macro_regions = {}
    for region in gcp_data:
        macro_region = region[0]
        if macro_region == 'us':
            macro_region = 'northamerica'

        if macro_region not in macro_regions:
            macro_regions[macro_region] = []
        macro_regions[macro_region].append(region[1:])  # Store (gcp_region, lat, lon)

    macro_region_list = list(macro_regions.keys())
    number_of_macro_regions = len(macro_region_list)
    macro_region_selected_counts = {region: 0 for region in macro_region_list}
    validators = []
    for i in range(number_of_validators):
        selected_macro_region = macro_region_list[i % number_of_macro_regions]
        region_options = macro_regions[selected_macro_region]
        selected_count = macro_region_selected_counts[selected_macro_region]
        selected_region = region_options[selected_count % len(region_options)]
        validators.append(selected_region)
        macro_region_selected_counts[selected_macro_region] += 1
    
    return pd.DataFrame(validators, columns=["gcp_region", "latitude", "longitude"])


def homogeneous_info_sources(gcp_regions):
    gcp_data = [(region["gcp_region"].split("-")[0], region["gcp_region"], region["lat"], region["lon"]) for _, region in gcp_regions.iterrows()]
    macro_regions = {}
    for region in gcp_data:
        macro_region = region[0]
        if macro_region == 'us':
            macro_region = 'northamerica'

        if macro_region not in macro_regions:
            macro_regions[macro_region] = []
        macro_regions[macro_region].append(region[1:])  # Store (gcp_region, lat, lon)

    signal_profiles = []
    relay_profiles = []
    for macro_region, sub_regions in macro_regions.items():
        factor = len(macro_regions) * len(sub_regions)
        for i, sub_region in enumerate(sub_regions):
            signal_profile = {
                "unique_id": f"signal-{macro_region}-{i}",
                "gcp_region": sub_region[0],
                "lat": sub_region[1],
                "lon": sub_region[2],
                "utility_function": LinearMEVUtility(0.4/factor, 0.04/factor, 1.0),
            }
            signal_profiles.append(signal_profile)

            relay_profile = {
                "unique_id": f"relay-{macro_region}-{i}",
                "gcp_region": sub_region[0],
                "lat": sub_region[1],
                "lon": sub_region[2],
                "utility_function": LinearMEVUtility(0.4, 0.04, 1.0),
            }
            relay_profiles.append(relay_profile)
        
    return signal_profiles, relay_profiles


def simulation(
    model,
    number_of_validators,
    num_slots,
    validators,
    gcp_regions,
    gcp_latency,
    consensus_settings,  # Pass the ConsensusSettings object
    relay_profiles,  # Pass the list of Relay profiles
    signal_profiles,
    timing_strategies,  # Pass the list of timing strategies
    location_strategies,  # Pass the list of location strategies
    simulation_name,  # Simulation name from YAML
    output_folder,  # Output folder
    time_window,  # Time window for migration checks
    fast_mode=False,  # Fast mode for latency computation
    cost=0.0001,  # Cost for migration, default to 0.0001
    seed=DEFAULT_SIMULATION_SEED,
    collect_full_history=False,
    export_raw_artifacts=True,
    verbose=False,
):
    # --- Simulation Execution ---
    random.seed(seed)  # For reproducibility
    np.random.seed(seed)  # For reproducibility in NumPy operations

    # --- Define Simulation Parameters ---
    # Calculate total time steps using values from ConsensusSettings
    TOTAL_TIME_STEPS = (
        num_slots
        * (
            consensus_settings.slot_duration_ms
            // consensus_settings.time_granularity_ms
        )
        + 1
    )

    # --- Use Strategies from YAML ---
    all_timing_strategies = timing_strategies
    all_location_strategies = location_strategies

    model_params_standard_nomig = {
        "num_validators": number_of_validators,
        "num_relays": len(relay_profiles),  # Use the actual number of loaded relays
        "num_signals": len(signal_profiles),
        "timing_strategies_pool": all_timing_strategies,
        "location_strategies_pool": all_location_strategies,
        "num_slots": num_slots,
        "proposer_has_optimized_latency": False,  # This could also be a YAML config if needed
        "validator_profiles": validators,
        "gcp_regions": gcp_regions,
        "gcp_latency": gcp_latency,
        "consensus_settings": consensus_settings,  # Pass the ConsensusSettings object to the model
        "relay_profiles": relay_profiles,  # Pass the Relay profiles to the model
        "signal_profiles": signal_profiles,
        "time_window": time_window,  # Time window for migration checks
        "fast_mode": fast_mode,  # Fast mode for latency computation
        "cost": cost,  # Cost for migration
        "collect_full_history": collect_full_history,
        "collect_raw_artifacts": export_raw_artifacts,
        "verbose": verbose,
    }

    # --- Create and Run the Model ---
    print(f"\n--- Starting MEV-Boost Simulation: {simulation_name} ---")
    print(f"Simulation seed: {seed}")
    start_time = time.time()

    model_standard = None
    try:
        if model == "SSP":
            model_standard = SingleSourceParadigm(**model_params_standard_nomig)
        else:
            model_standard = MultiSourceParadigm(**model_params_standard_nomig)

        for _ in range(TOTAL_TIME_STEPS):
            model_standard.step()
            if not model_standard.running:
                print(
                    f"Stopping simulation as no validators moved within the time window ({time_window})."
                )
                break
        
        if model_standard.running:
            print(
                f"Stopping simulation after reaching the maximum time steps: {TOTAL_TIME_STEPS}."
            )

        print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")

        # --- Final Analysis & Plotting ---
        print("\n--- Final Results Summary ---")
        print(f"Total Slots: {model_standard.current_slot_idx + 1}")
        print(f"Total MEV Earned: {model_standard.total_mev_earned:.4f} ETH")
        completed_slots = max(model_standard.current_slot_idx, 1)
        print(
            f"Avg MEV Earned per Slot: {model_standard.total_mev_earned / completed_slots:.4f} ETH"
        )

        # profiles:
        for profiles, output_name in [
            (relay_profiles, "relay_names.json"),
            (signal_profiles, "signal_names.json"),
        ]:
            names = [(profile["unique_id"], profile["gcp_region"]) for profile in profiles]
            with open(f"{output_folder}/{output_name}", "w") as f:
                json.dump(names, f)

        export_payloads = build_export_payloads(model_standard)

        gcp_region_profits = pd.DataFrame(model_standard.region_profits)
        gcp_region_profits.to_csv(f"{output_folder}/region_profits.csv", index=False)

        with open(f"{output_folder}/avg_mev.json", "w") as f:
            json.dump(export_payloads["avg_mev"], f)

        with open(f"{output_folder}/supermajority_success.json", "w") as f:
            json.dump(export_payloads["supermajority_success"], f)

        with open(f"{output_folder}/failed_block_proposals.json", "w") as f:
            json.dump(export_payloads["failed_block_proposals"], f)

        with open(f"{output_folder}/utility_increase.json", "w") as f:
            json.dump(export_payloads["utility_increase"], f)

        action_reasons = model_standard.action_reasons
        action_reasons_df = pd.DataFrame(
            action_reasons, columns=["Action_Reason", "Previous_Region", "New_Region"]
        )
        action_reasons_df.to_csv(f"{output_folder}/action_reasons.csv", index=False)

        with open(f"{output_folder}/proposal_time_avg.json", "w") as f:
            json.dump(export_payloads["proposal_time_avg"], f)
        with open(f"{output_folder}/attestation_sum.json", "w") as f:
            json.dump(export_payloads["attestation_sum"], f)
        with open(f"{output_folder}/proposer_strategy_and_mev.json", "w") as f:
            json.dump(export_payloads["proposer_strategy_and_mev"], f)
        with open(f"{output_folder}/top_regions_final.json", "w") as f:
            json.dump(export_payloads["top_regions_final"], f)

        if export_raw_artifacts:
            with open(f"{output_folder}/mev_by_slot.json", "w") as f:
                json.dump(export_payloads["mev_by_slot"], f)
            with open(f"{output_folder}/estimated_mev_by_slot.json", "w") as f:
                json.dump(export_payloads["estimated_mev_by_slot"], f)
            with open(f"{output_folder}/attest_by_slot.json", "w") as f:
                json.dump(export_payloads["attest_by_slot"], f)
            with open(f"{output_folder}/proposal_time_by_slot.json", "w") as f:
                json.dump(export_payloads["proposal_time_by_slot"], f)
            with open(f"{output_folder}/region_counter_per_slot.json", "w") as f:
                json.dump(export_payloads["region_counter_per_slot"], f)

        print("Saved data in JSON files in the output directory.")
        print("Information Sources:")
        if model == "SSP":
            print("Relays:")
            print("\n".join([f"{i['unique_id']} ({i['gcp_region']})" for i in relay_profiles]))
        else:
            print("Signals:")
            print("\n".join([f"{r['unique_id']} ({r['gcp_region']})" for r in signal_profiles]))
    finally:
        if model_standard is not None:
            model_standard.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the MEV-Boost simulation using YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="params/simulation_config.yaml",
        help="Path to the simulation configuration YAML file (default: 'params/simulation_config.yaml')",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data",
        help="Directory to read input data (default: 'data')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="default",
        help="Directory to save output data (default is configured in the YAML file)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SSP",
        choices=["SSP", "MSP"],
        help="Type of model to simulate: 'SSP' or 'MSP' (default: 'SSP')",
    )
    parser.add_argument(
        "--validators",
        type=int,
        default=1000,
        help="Number of validators to simulate (default: 1000)",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=1000,
        help="Number of slots to simulate (default: 1000)",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.0001,
        help="Cost for migration (default: 0.0001)",
    )
    parser.add_argument(
        "--time_window",
        type=int,
        default=10,
        help="Time window for migration checks (default: 10)",
    )
    parser.add_argument(
        "--fast",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable fast mode for latency computation (default: False)",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="homogeneous",
        choices=["homogeneous", "heterogeneous", "random", "homogeneous-gcp"],
        help="Validator distribution strategy (default: homogeneous)"
    )
    parser.add_argument(
        "--info-distribution",
        type=str,
        default="homogeneous",
        choices=["homogeneous", "heterogeneous"],
        help="Distribution of information sources (default: homogeneous)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.6667,
        help="Attestation threshold (\\gamma, γ) (default: 0.6667)",
    )
    parser.add_argument(
        "--delta",
        type=int,
        default=12000,
        help="Slot time (\\Delta, Δ) in milliseconds (default: 12000)",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=4000,
        help="Cutoff time for attestations in milliseconds (default: 4000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed for the simulation (default: YAML seed or {DEFAULT_SIMULATION_SEED})",
    )
    parser.add_argument(
        "--full-history",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable Mesa DataCollector history in addition to the lightweight slot summaries (default: False)",
    )
    parser.add_argument(
        "--cache-results",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Reuse cached simulation outputs when the inputs, seed, and code version match (default: True)",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable verbose slot and migration logging (default: False)",
    )

    args = parser.parse_args()

    try:
        # Load the entire simulation configuration from YAML
        config = load_simulation_config(args.config)
        # Extract top-level simulation parameters from config
        simulation_name = config.get("simulation_name", "Default Simulation")
        model = args.model if args.model else config.get("model", "SSP")
        # Use 'iterations' from YAML as num_slots
        num_slots = args.slots if args.slots else config.get("iterations", 1000)
        num_validators = (
            args.validators if args.validators else config.get("num_validators", 1000)
        )
        input_folder = config.get("input_folder", args.input_dir)
        output_folder = config.get("output_folder", "output")

        # Initialize Consensus Settings
        consensus_parameters = config.get("consensus_settings", {})
        consensus_settings = ConsensusSettings(**consensus_parameters)
        consensus_settings.attestation_threshold = args.gamma
        consensus_settings.slot_duration_ms = args.delta
        consensus_settings.attestation_time_ms = args.cutoff

        # Time window for migration checks
        time_window = (
            args.time_window if args.time_window else config.get("time_window", 10)
        )  # Default to 10

        # fast mode
        fast_mode = args.fast

        # cost for migration
        cost = args.cost if args.cost is not None else config.get("migration_cost", 0.0001)
        seed = args.seed if args.seed is not None else config.get("seed", DEFAULT_SIMULATION_SEED)

        if args.output_dir == "default":
            output_folder = os.path.join(
                output_folder,
                f"num_slots_{num_slots}_validators_{num_validators}_time_window_{time_window}_cost_{cost}_gamma_{args.gamma}_delta_{args.delta}_cutoff_{args.cutoff}_seed_{seed}",
            )
        else:
            output_folder = args.output_dir

        random.seed(seed)
        np.random.seed(seed)

        gcp_regions = pd.read_csv(os.path.join(input_folder, "gcp_regions.csv"))
        gcp_latency = pd.read_csv(os.path.join(input_folder, "gcp_latency.csv"))
        gcp_latency = parse_gcp_latency(gcp_latency)


        gcp_regions["gcp_region"] = gcp_regions["Region"]
        gcp_regions["lat"] = gcp_regions["Nearest City Latitude"]
        gcp_regions["lon"] = gcp_regions["Nearest City Longitude"]
    
        # heterogeneous distribution of validators
        # Input data (validators, gcp_regions, gcp_latency) are still from CSVs
        validators = pd.read_csv(os.path.join(input_folder, "validators.csv"))
        # Sample validators if the CSV has more than the configured number
        if len(validators) > num_validators:
            validators = validators.sample(n=num_validators, random_state=seed)
        else:
            print(
                f"Using all {len(validators)} validators from CSV as it's less than configured {num_validators}."
            )

        if args.info_distribution == "homogeneous":
            signal_profiles, relay_profiles = homogeneous_info_sources(gcp_regions)
        else:
            signal_profiles_data = config.get("signal_profiles", [])
            relay_profiles_data = config.get("relay_profiles", [])
            signal_profiles = initialize_signals(signal_profiles_data)
            relay_profiles = initialize_relays(relay_profiles_data)


        # Initialize Validator Distribution
        if args.distribution == "homogeneous-gcp": # homogeneous across all GCP regions
            validators = homogeneous_validators_per_gcp(gcp_regions, num_validators)
        elif args.distribution == "homogeneous": # homogeneous across macro regions
            validators = homogeneous_validators(gcp_regions, num_validators)
        elif args.distribution == "random": # random across all GCP regions
            validators = random_validators(gcp_regions, num_validators)
        elif args.distribution == "heterogeneous": # real-world heterogeneous from CSV
            pass

        # Get Proposer Timing Strategies
        timing_strategies = config.get(
            "proposer_strategies", [{"type": "optimal_latency"}]
        )

        # Get Proposer Location Strategies
        location_strategies = config.get(
            "location_strategies", [{"type": "best_relay"}]
        )

        cache_context = {
            "cache_version": SIMULATION_CACHE_VERSION,
            "model": model,
            "num_slots": num_slots,
            "num_validators": num_validators,
            "distribution": args.distribution,
            "info_distribution": args.info_distribution,
            "time_window": time_window,
            "fast_mode": fast_mode,
            "cost": cost,
            "seed": seed,
            "consensus_settings": vars(consensus_settings),
            "timing_strategies": timing_strategies,
            "location_strategies": location_strategies,
            "config": {
                key: value
                for key, value in config.items()
                if key != "output_folder"
            },
            "input_hashes": {
                "gcp_regions.csv": file_sha256(os.path.join(input_folder, "gcp_regions.csv")),
                "gcp_latency.csv": file_sha256(os.path.join(input_folder, "gcp_latency.csv")),
            },
            "code_hashes": {
                path: file_sha256(path) for path in SIMULATION_CODE_FILES
            },
        }
        if args.distribution == "heterogeneous":
            cache_context["input_hashes"]["validators.csv"] = file_sha256(
                os.path.join(input_folder, "validators.csv")
            )

        cache_dir = os.path.join(
            SIMULATION_CACHE_DIRNAME,
            compute_simulation_cache_key(cache_context),
        )

        # Ensure the output directory exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created base output directory: {output_folder}")

        if args.cache_results and os.path.isdir(cache_dir):
            print(f"Using cached simulation outputs from: {cache_dir}")
            copy_directory_contents(
                cache_dir,
                output_folder,
                exclude_names={"cache_manifest.json"},
            )
            raise SystemExit(0)

        # Run the simulation with parameters from YAML and CSVs
        simulation(
            model=model,
            number_of_validators=num_validators,
            num_slots=num_slots,
            validators=validators,
            gcp_regions=gcp_regions,
            gcp_latency=gcp_latency,
            consensus_settings=consensus_settings,
            relay_profiles=relay_profiles,
            signal_profiles=signal_profiles,
            timing_strategies=timing_strategies,
            location_strategies=location_strategies,
            simulation_name=simulation_name,
            output_folder=output_folder,  # Pass output_folder for consistent sub-directory creation
            time_window=time_window,
            fast_mode=fast_mode,
            cost=cost,
            seed=seed,
            collect_full_history=args.full_history,
            verbose=args.verbose,
        )

        if args.cache_results:
            os.makedirs(cache_dir, exist_ok=True)
            copy_directory_contents(output_folder, cache_dir)
            with open(os.path.join(cache_dir, "cache_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(cache_context, f, indent=2)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        traceback.print_exc()
        print(f"\nFatal error during simulation setup or execution: {e}")
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        print(f"\nAn unexpected error occurred: {e}")
