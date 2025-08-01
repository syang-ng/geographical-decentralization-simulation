import argparse
import json
import numpy as np
import os
import pandas as pd
import random
import time
import yaml # Import yaml library
from collections import defaultdict, Counter

from consensus import ConsensusSettings
from measure import * # Assuming measure.py contains necessary measurement functions
from mevboost import MEVBoostModel # Assuming MEVBoostModel is defined in mevboost.py
from relay_agent import initialize_relays

# --- Simulation Initialization Functions ---

def load_simulation_config(config_file_path):
    """Loads and parses the simulation's YAML configuration file."""
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file '{config_file_path}' not found. Please ensure the file exists.")

    try:
        with open(config_file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print(f"✅ Successfully loaded configuration from: {config_file_path}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unknown error loading configuration file: {e}")

def initialize_consensus_settings(config_data):
    """Initializes a ConsensusSettings instance from configuration data."""
    consensus_settings_data = config_data.get('consensus_settings', {})
    return ConsensusSettings(**consensus_settings_data)


def simulation(
    number_of_validators,
    num_slots,
    validators,
    gcp_regions,
    gcp_latency,
    consensus_settings, # Pass the ConsensusSettings object
    relay_profiles,    # Pass the list of Relay profiles
    timing_strategies,  # Pass the list of timing strategies
    location_strategies,# Pass the list of location strategies
    simulation_name,    # Simulation name from YAML
    output_folder,      # Output folder
    time_window,        # Time window for migration checks
):
    # --- Simulation Execution ---
    random.seed(0x06511)  # For reproducibility
    np.random.seed(0x06511)  # For reproducibility in NumPy operations

    # --- Define Simulation Parameters ---
    # Calculate total time steps using values from ConsensusSettings
    TOTAL_TIME_STEPS = (
        num_slots * (consensus_settings.slot_duration_ms // consensus_settings.time_granularity_ms) + 1
    )

    # --- Use Strategies from YAML ---
    all_timing_strategies = timing_strategies
    all_location_strategies = location_strategies

    model_params_standard_nomig = {
        "num_validators": number_of_validators,
        "num_relays": len(relay_profiles), # Use the actual number of loaded relays
        "timing_strategies_pool": all_timing_strategies,
        "location_strategies_pool": all_location_strategies,
        "num_slots": num_slots,
        "proposer_has_optimized_latency": False, # This could also be a YAML config if needed
        "validators": validators,
        "gcp_regions": gcp_regions,
        "gcp_latency": gcp_latency,
        "consensus_settings": consensus_settings, # Pass the ConsensusSettings object to the model
        "relay_profiles": relay_profiles, # Pass the Relay profiles to the model
        "time_window": time_window,  # Time window for migration checks
    }

    # --- Create and Run the Model ---
    print(f"\n--- Starting MEV-Boost Simulation: {simulation_name} ---")
    start_time = time.time()
    model_standard = MEVBoostModel(**model_params_standard_nomig)
    for i in range(TOTAL_TIME_STEPS):
        model_standard.step()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

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

    # relay profiles:
    relay_names = [relay['unique_id'] for relay in relay_profiles]
    with open(f"{output_folder}/relay_names.json", "w") as f:
        json.dump(relay_names, f)

    avg_mev_series = model_data["Average_MEV_Earned"].tolist()
    supermaj_series = model_data["Supermajority_Success_Rate"].tolist()
    failed_block_proposals = model_data["Failed_Block_Proposals"].tolist()

    with open(f"{output_folder}/avg_mev.json", "w") as f:
        json.dump(avg_mev_series, f)

    with open(f"{output_folder}/supermajority_success.json", "w") as f:
        json.dump(supermaj_series, f)
    
    with open(f"{output_folder}/failed_block_proposals.json", "w") as f:
        json.dump(failed_block_proposals, f)

    agent_data = model_standard.datacollector.get_agent_vars_dataframe()

    print("\n--- Agent Data Collected ---")
    print("DataFrame Head:")
    print(agent_data.head())

    relay_agent_data = agent_data[agent_data["Role"] == "relay_agent"]

    print("\nDataFrame Tail:")
    print(agent_data.tail())

    print("\nDataFrame Info:")
    agent_data.info()
    if isinstance(agent_data.index, pd.MultiIndex):
        agent_data = agent_data.reset_index()

    validator_agent_data = agent_data[agent_data["Role"] != "relay_agent"].reindex()
    positions_by_slot = validator_agent_data.groupby("Slot")["Position"].apply(list).reset_index()
    nested_array = positions_by_slot["Position"].tolist()
    # Group by slot and collect lists of per-agent values:
    mev_by_slot = validator_agent_data.groupby("Slot")["MEV_Captured_Slot"].apply(list).tolist()
    attest_by_slot = validator_agent_data.groupby("Slot")["Attestation_Rate"].apply(list).tolist()
    proposal_time_by_slot = (
        validator_agent_data.groupby("Slot")["Proposal Time"].apply(list).tolist()
    )
    relay_positions = relay_agent_data["Position"].iloc[0:number_of_relays]
    # Since relay is not moving, we can just use the first position and multiply it by the number of slots
    # --------------------------------------------
    # build one relay-point list per slot
    # --------------------------------------------
    relay_pos_list = [list(relay_position) for relay_position in relay_positions]  # JSON wants lists
    nested_array_relay = [
        relay_pos_list for _ in range(len(nested_array))  # <- extra [] !
    ]

    latest_steps = validator_agent_data.sort_values("Step").groupby(["Slot", "AgentID"], as_index=False).last()
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
    with open(f"{output_folder}/attest_by_slot.json", "w") as f:
        json.dump(attest_by_slot, f)
    with open(f"{output_folder}/proposal_time_by_slot.json", "w") as f:
        json.dump(proposal_time_by_slot, f)
    with open(f"{output_folder}/proposer_strategy_and_mev.json", "w") as f:
        json.dump(proposer_strategy_and_mev, f)

    with open(f"{output_folder}/relay_data.json", "w") as f:
        json.dump(nested_array_relay, f)

    with open(f"{output_folder}/region_counter_per_slot.json", "w") as f:
        json.dump(region_counter_per_slot, f)

    print("Saved data in JSON files in the output directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the MEV-Boost simulation using YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="simulation_config.yaml",
        help="Path to the simulation configuration YAML file (default: 'simulation_config.yaml')",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data",
        help="Directory to read input data (default: 'data')",
    )
    args = parser.parse_args()

    try:
        # Load the entire simulation configuration from YAML
        config = load_simulation_config(args.config)

        # Extract top-level simulation parameters from config
        simulation_name = config.get('simulation_name', 'Default Simulation')
        # Use 'iterations' from YAML as num_slots
        num_slots = config.get('iterations', )
        num_validators_from_config = config.get('validators', 1000)
        input_folder = config.get('input_folder', args.input_dir)
        output_folder = config.get('output_folder', 'output')

        # Initialize Consensus Settings
        consensus_parameters = config.get('consensus_settings', {})
        consensus_settings = ConsensusSettings(**consensus_parameters)

        # Time window for migration checks
        time_window = config.get('time_window', 10)  # Default to 10

        # Initialize Relays
        relay_profiles_data = config.get('relay_profiles', [])
        relay_profiles = initialize_relays(relay_profiles_data)
        # Get actual count of initialized relays
        number_of_relays = len(relay_profiles)

        # Get Proposer Timing Strategies
        timing_strategies = config.get('proposer_strategies', [{"type": "optimal_latency"}])

        # Get Proposer Location Strategies
        location_strategies = config.get('location_strategies', [{"type": "best_relay"}])

        # Ensure the output directory exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created base output directory: {output_folder}")

        # Input data (validators, gcp_regions, gcp_latency) are still from CSVs
        validators = pd.read_csv(os.path.join(input_folder, "validators.csv"))
        # Sample validators if the CSV has more than the configured number
        if len(validators) > num_validators_from_config:
            validators = validators.sample(n=num_validators_from_config, random_state=42)
        else:
            print(f"Using all {len(validators)} validators from CSV as it's less than configured {num_validators_from_config}.")

        gcp_regions = pd.read_csv(os.path.join(input_folder, "gcp_regions.csv"))
        gcp_latency = pd.read_csv(os.path.join(input_folder, "gcp_latency.csv"))

        # Run the simulation with parameters from YAML and CSVs
        simulation(
            number_of_validators=len(validators), # Use the actual number of loaded validators
            num_slots=num_slots,
            validators=validators,
            gcp_regions=gcp_regions,
            gcp_latency=gcp_latency,
            consensus_settings=consensus_settings,
            relay_profiles=relay_profiles,
            timing_strategies=timing_strategies,
            location_strategies=location_strategies,
            simulation_name=simulation_name,
            output_folder=output_folder, # Pass output_folder for consistent sub-directory creation
            time_window=time_window,
        )

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ Fatal error during simulation setup or execution: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
