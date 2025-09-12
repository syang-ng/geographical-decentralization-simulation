import argparse
import json
import numpy as np
import os
import pandas as pd
import random
import time
import traceback
import yaml  # Import yaml library
from collections import defaultdict, Counter

from constants import LinearMEVUtility
from consensus import ConsensusSettings
from distribution import convert_to_marco_regions, parse_gcp_latency
from measure import *  # Assuming measure.py contains necessary measurement functions
from models import MEVBoostModel, EthereumWithoutMEVBoostModel
from relay_agent import initialize_relays, get_evenly_distributed_relay_profiles
from info_agent import initialize_infos, get_evenly_distributed_info_profiles


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
        print(f"✅ Successfully loaded configuration from: {config_file_path}")
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

def evenly_distribute_validators(gcp_regions, number_of_validators):
    """Generates a list of validators evenly distributed across GCP regions."""
    gcp_data = [(region["gcp_region"], region["lat"], region["lon"]) for _, region in gcp_regions.iterrows()]
    num_regions = len(gcp_data)
    validators = [gcp_data[i % num_regions] for i in range(number_of_validators)]

    return pd.DataFrame(validators, columns=["gcp_region", "latitude", "longitude"])

def macro_region_evenly_distribute_validators(gcp_regions, number_of_validators):
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


def evenly_distribute_info_profiles(gcp_regions):
    gcp_data = [(region["gcp_region"].split("-")[0], region["gcp_region"], region["lat"], region["lon"]) for _, region in gcp_regions.iterrows()]
    macro_regions = {}
    for region in gcp_data:
        macro_region = region[0]
        if macro_region == 'us':
            macro_region = 'northamerica'

        if macro_region not in macro_regions:
            macro_regions[macro_region] = []
        macro_regions[macro_region].append(region[1:])  # Store (gcp_region, lat, lon)

    info_profiles = []
    for macro_region, sub_regions in macro_regions.items():
        factor = len(macro_regions) * len(sub_regions)
        for i, sub_region in enumerate(sub_regions):
            profile = {
                "unique_id": f"info-{macro_region}-{i}",
                "gcp_region": sub_region[0],
                "lat": sub_region[1],
                "lon": sub_region[2],
                "utility_function": LinearMEVUtility(0.4/factor, 0.04/factor, 1.0),
            }
            info_profiles.append(profile)
        
    return info_profiles


def simulation(
    model,
    number_of_validators,
    num_slots,
    validators,
    gcp_regions,
    gcp_latency,
    consensus_settings,  # Pass the ConsensusSettings object
    relay_profiles,  # Pass the list of Relay profiles
    info_profiles,
    timing_strategies,  # Pass the list of timing strategies
    location_strategies,  # Pass the list of location strategies
    simulation_name,  # Simulation name from YAML
    output_folder,  # Output folder
    time_window,  # Time window for migration checks
    fast_mode=False,  # Fast mode for latency computation
    cost=0.0001,  # Cost for migration, default to 0.0001
):
    # --- Simulation Execution ---
    random.seed(0x06511)  # For reproducibility
    np.random.seed(0x06511)  # For reproducibility in NumPy operations

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
        "num_infos": len(info_profiles),
        "timing_strategies_pool": all_timing_strategies,
        "location_strategies_pool": all_location_strategies,
        "num_slots": num_slots,
        "proposer_has_optimized_latency": False,  # This could also be a YAML config if needed
        "validator_profiles": validators,
        "gcp_regions": gcp_regions,
        "gcp_latency": gcp_latency,
        "consensus_settings": consensus_settings,  # Pass the ConsensusSettings object to the model
        "relay_profiles": relay_profiles,  # Pass the Relay profiles to the model
        "info_profiles": info_profiles,
        "time_window": time_window,  # Time window for migration checks
        "fast_mode": fast_mode,  # Fast mode for latency computation
        "cost": cost,  # Cost for migration
    }

    # --- Create and Run the Model ---
    print(f"\n--- Starting MEV-Boost Simulation: {simulation_name} ---")
    start_time = time.time()

    if model == 'mevboost':
        model_standard = MEVBoostModel(**model_params_standard_nomig)
    else:
        model_standard = EthereumWithoutMEVBoostModel(**model_params_standard_nomig)

    for i in range(TOTAL_TIME_STEPS):
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
    # `dir` already holds the specific output path for this simulation run

    print(f"Total Slots: {model_standard.current_slot_idx + 1}")
    print(f"Total MEV Earned: {model_standard.total_mev_earned:.4f} ETH")
    print(
        f"Avg MEV Earned per Slot: {model_standard.total_mev_earned / (model_standard.current_slot_idx):.4f} ETH"
    )
    model_data = model_standard.datacollector.get_model_vars_dataframe()

    print("\n--- Collected Model Data ---")
    print(model_data.head())
    print(model_data.tail())

    # profiles:
    for profiles, output_name in [
        (relay_profiles, "relay_names.json"),
        (info_profiles, "info_names.json"),
    ]:
        names = [profile["unique_id"] for profile in profiles]
        with open(f"{output_folder}/{output_name}", "w") as f:
            json.dump(names, f)

    avg_mev_series = model_data["Average_MEV_Earned"].tolist()
    supermaj_series = model_data["Supermajority_Success_Rate"].tolist()
    failed_block_proposals = model_data["Failed_Block_Proposals"].tolist()

    with open(f"{output_folder}/avg_mev.json", "w") as f:
        json.dump(avg_mev_series, f)

    with open(f"{output_folder}/supermajority_success.json", "w") as f:
        json.dump(supermaj_series, f)

    with open(f"{output_folder}/failed_block_proposals.json", "w") as f:
        json.dump(failed_block_proposals, f)

    action_reasons = model_standard.action_reasons
    action_reasons_df = pd.DataFrame(
        action_reasons, columns=["Action_Reason", "Previous_Region", "New_Region"]
    )
    action_reasons_df.to_csv(f"{output_folder}/action_reasons.csv", index=False)

    agent_data = model_standard.datacollector.get_agent_vars_dataframe()

    print("\n--- Agent Data Collected ---")
    print("DataFrame Head:")
    print(agent_data.head())

    print("\nDataFrame Tail:")
    print(agent_data.tail())

    print("\nDataFrame Info:")
    agent_data.info()
    if isinstance(agent_data.index, pd.MultiIndex):
        agent_data = agent_data.reset_index()

    validator_agent_data = agent_data[agent_data["Role"] != "relay_agent"].reindex()
    positions_by_slot = (
        validator_agent_data.groupby("Slot")["Position"].apply(list).reset_index()
    )
    nested_array = positions_by_slot["Position"].tolist()
    # Group by slot and collect lists of per-agent values:
    mev_by_slot = (
        validator_agent_data.groupby("Slot")["MEV_Captured_Slot"].apply(list).tolist()
    )
    estimated_mev_by_slot = (
        validator_agent_data.groupby("Slot")["Estimated_Profit"].apply(list).tolist()
    )
    attest_by_slot = (
        validator_agent_data.groupby("Slot")["Attestation_Rate"].apply(list).tolist()
    )
    proposal_time_by_slot = (
        validator_agent_data.groupby("Slot")["Proposal Time"].apply(list).tolist()
    )

    relay_agent_data = agent_data[agent_data["Role"] == "relay_agent"]
    relay_positions = relay_agent_data["Position"].iloc[0:number_of_relays]
    # Since relay is not moving, we can just use the first position and multiply it by the number of slots
    # --------------------------------------------
    # build one relay-point list per slot
    # --------------------------------------------
    relay_pos_list = [
        list(relay_position) for relay_position in relay_positions
    ]  # JSON wants lists
    nested_array_relay = [
        relay_pos_list for _ in range(len(nested_array))  # <- extra [] !
    ]
    # --------------------------------------------
    info_agent_data = agent_data[agent_data["Role"] == "info_agent"]
    info_positions = info_agent_data["Position"].iloc[0 : len(info_profiles)]
    info_pos_list = [list(info_position) for info_position in info_positions]
    nested_array_info = [
        info_pos_list for _ in range(len(nested_array))  # <- extra [] !
    ]

    latest_steps = (
        validator_agent_data.sort_values("Step")
        .groupby(["Slot", "AgentID"], as_index=False)
        .last()
    )
    region_counter_per_slot = defaultdict(list)
    for slot, slot_df in latest_steps.groupby("Slot"):
        region_counts = Counter(slot_df["GCP_Region"])
        region_counter_per_slot[int(slot)] = region_counts.most_common()

    # Proposer data
    proposer_data = agent_data[agent_data["Role"] == "proposer"]
    proposer_strategy_and_mev = proposer_data[
        ["Slot", "Location_Strategy", "MEV_Captured_Slot"]
    ].to_dict(orient="records")

    with open(f"{output_folder}/data.json", "w") as f:
        json.dump(nested_array, f)
    with open(f"{output_folder}/mev_by_slot.json", "w") as f:
        json.dump(mev_by_slot, f)
    with open(f"{output_folder}/estimated_mev_by_slot.json", "w") as f:
        json.dump(estimated_mev_by_slot, f)
    with open(f"{output_folder}/attest_by_slot.json", "w") as f:
        json.dump(attest_by_slot, f)
    with open(f"{output_folder}/proposal_time_by_slot.json", "w") as f:
        json.dump(proposal_time_by_slot, f)
    with open(f"{output_folder}/proposer_strategy_and_mev.json", "w") as f:
        json.dump(proposer_strategy_and_mev, f)

    with open(f"{output_folder}/relay_data.json", "w") as f:
        json.dump(nested_array_relay, f)
    with open(f"{output_folder}/info_data.json", "w") as f:
        json.dump(nested_array_info, f)

    with open(f"{output_folder}/region_counter_per_slot.json", "w") as f:
        json.dump(region_counter_per_slot, f)

    print("Saved data in JSON files in the output directory.")
    print("Information sources:")
    if len(nested_array_info) > 0:
        print("Infos:")
        print("\n".join([f"{i['unique_id']} ({i['gcp_region']})" for i in info_profiles]))
    if len(nested_array_relay) > 0:
        print("Relays:")
        print("\n".join([f"{r['unique_id']} ({r['gcp_region']})" for r in relay_profiles]))

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
        "--input_dir",
        type=str,
        default="data",
        help="Directory to read input data (default: 'data')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mevboost",
        choices=["mevboost", "raw"],
        help="Type of model to simulate: 'mevboost' or 'raw' (default: 'mevboost')",
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
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable fast mode for latency computation (default: False)",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="even",
        choices=["even", "macro-region-even", "macro-region-even-v0", "random", "real-world"],
        help="Assign validators / information sources to GCP regions (default: False)"
    )

    args = parser.parse_args()

    try:
        # Load the entire simulation configuration from YAML
        config = load_simulation_config(args.config)
        # Extract top-level simulation parameters from config
        simulation_name = config.get("simulation_name", "Default Simulation")
        model = args.model if args.model else config.get("model", "mevboost")
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

        # Time window for migration checks
        time_window = (
            args.time_window if args.time_window else config.get("time_window", 10)
        )  # Default to 10

        # fast mode
        fast_mode = args.fast

        # cost for migration
        cost = args.cost if args.cost else config.get("migration_cost", 0.0001)

        output_folder = os.path.join(
            output_folder,
            f"num_slots_{num_slots}_validators_{num_validators}_time_window_{time_window}_cost_{cost}",
        )

        gcp_regions = pd.read_csv(os.path.join(input_folder, "gcp_regions.csv"))
        gcp_latency = pd.read_csv(os.path.join(input_folder, "gcp_latency.csv"))

        gcp_regions["gcp_region"] = gcp_regions["Region"]
        gcp_regions["lat"] = gcp_regions["Nearest City Latitude"]
        gcp_regions["lon"] = gcp_regions["Nearest City Longitude"]
    
        # Input data (validators, gcp_regions, gcp_latency) are still from CSVs
        validators = pd.read_csv(os.path.join(input_folder, "validators.csv"))
        # Sample validators if the CSV has more than the configured number
        if len(validators) > num_validators:
            validators = validators.sample(n=num_validators, random_state=42)
        else:
            print(
                f"Using all {len(validators)} validators from CSV as it's less than configured {num_validators}."
            )

        if args.distribution == "macro-region-even":
            gcp_regions, gcp_latency = convert_to_marco_regions(gcp_regions, gcp_latency)

            number_of_relays = number_of_infos = gcp_regions.shape[0]
            info_profiles = get_evenly_distributed_info_profiles(gcp_regions, number_of_infos)
            relay_profiles = get_evenly_distributed_relay_profiles(gcp_regions, number_of_relays)
        else:
            gcp_latency = parse_gcp_latency(gcp_latency)

            relay_profiles_data = config.get("relay_profiles", [])
            relay_profiles = initialize_relays(relay_profiles_data)
            number_of_relays = len(relay_profiles)

            info_profiles_data = config.get("info_profiles", [])
            info_profiles = initialize_infos(info_profiles_data)
            number_of_infos = len(info_profiles)

        # Initialize Relays/Info
        if args.distribution == "even":
            validators = evenly_distribute_validators(gcp_regions, num_validators)
        elif args.distribution == "macro-region-even":
            validators = evenly_distribute_validators(gcp_regions, num_validators)
        elif args.distribution == "macro-region-even-v0":
            validators = macro_region_evenly_distribute_validators(gcp_regions, num_validators)
            # info_profiles = evenly_distribute_info_profiles(gcp_regions)
        elif args.distribution == "random":
            validators = random_validators(gcp_regions, num_validators)
        elif args.distribution == "real-world":
            pass

        # Get Proposer Timing Strategies
        timing_strategies = config.get(
            "proposer_strategies", [{"type": "optimal_latency"}]
        )

        # Get Proposer Location Strategies
        location_strategies = config.get(
            "location_strategies", [{"type": "best_relay"}]
        )

        # Ensure the output directory exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created base output directory: {output_folder}")

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
            info_profiles=info_profiles,
            timing_strategies=timing_strategies,
            location_strategies=location_strategies,
            simulation_name=simulation_name,
            output_folder=output_folder,  # Pass output_folder for consistent sub-directory creation
            time_window=time_window,
            fast_mode=fast_mode,
            cost=cost,
        )

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        traceback.print_exc()
        print(f"\n❌ Fatal error during simulation setup or execution: {e}")
    except Exception as e:
        traceback.print_exc()
        print(f"\n❌ An unexpected error occurred: {e}")
