import argparse
import json

import numpy as np
import os
import pandas as pd
import random
import time

from constants import *
from measure import *

from mevboost import MEVBoostModel


def simulation(
    number_of_validators,
    number_of_relays,
    num_slots,
    dir,
    validators=None,
    gcp_regions=None,
    gcp_latency=None,
):
    # --- Simulation Execution ---
    random.seed(0x06511)  # For reproducibility
    np.random.seed(0x06511)  # For reproducibility in NumPy operations

    # --- Define Simulation Parameters ---
    TOTAL_TIME_STEPS = (
        num_slots * (SLOT_DURATION_MS // TIME_GRANULARITY_MS) + 1
    )  # Total fine-grained steps (200 * 120 = 24000 steps)

    # --- Define Proposer Strategies ---
    # all_timing_strategies = [
    #     {"type": "fixed_delay", "delay_ms": 0},  # Strategy 0: Honest proposer with no delay
    #     {"type": "fixed_delay", "delay_ms": 500}, # Strategy 1: Fixed delay at 0.5 seconds
    #     {"type": "fixed_delay", "delay_ms": 1500}, # Strategy 2: Faster fixed delay at 1.5 second
    #     {"type": "threshold_and_max_delay", "mev_threshold": 0.3, "max_delay_ms": 2500}, # Strategy 3: Threshold 0.4 ETH or max 2.5s delay
    #     {"type": "random_delay", "min_delay_ms": 1000, "max_delay_ms": 3000}, # Strategy 4: Random delay between 1s and 3s
    #     {"type": "threshold_and_max_delay", "mev_threshold": 0.4, "max_delay_ms": 3000}, # Strategy 5: More aggressive threshold 0.4 ETH or max 3s delay,
    #     {"type": "optimal_latency"}, # Strategy 6: Proposer with optimized latency
    # ]
    all_timing_strategies = [
        {"type": "optimal_latency"},
    ]

    # --- Define Migration Strategies ---
    MIGRATION_STRATEGY_NEVER = {"type": "never_migrate"}
    MIGRATION_STRATEGY_RANDOM = {"type": "random_relay"}
    MIGRATION_STRATEGY_TARGET = {"type": "target_relay", "target_relay": "us-central1-a"}
    MIGRATION_STRATEGY_CENTER = {"type": "optimize_to_center"}
    MIGRATION_STRATEGY_BEST = {"type": "best_relay"}

    all_location_strategies = [
        # MIGRATION_STRATEGY_NEVER,
        # MIGRATION_STRATEGY_RANDOM,
        # MIGRATION_STRATEGY_CENTER,
        MIGRATION_STRATEGY_BEST
    ]

    model_params_standard_nomig = {
        "num_validators": number_of_validators,
        "num_relays": number_of_relays,
        "timing_strategies_pool": all_timing_strategies,
        "location_strategies_pool": all_location_strategies,
        "num_slots": num_slots,
        "proposer_has_optimized_latency": False,
        "base_mev_amount": BASE_MEV_AMOUNT,
        "mev_increase_per_second": MEV_INCREASE_PER_SECOND,
        "validators": validators,  # Will be set from CSV
        "gcp_regions": gcp_regions,  # Will be set from CSV
        "gcp_latency": gcp_latency,  # Will be set from CSV
    }

    # --- Create and Run the Model ---
    print("\n--- Starting MEV-Boost Simulation ---")
    start_time = time.time()
    model_standard = MEVBoostModel(**model_params_standard_nomig)
    for i in range(TOTAL_TIME_STEPS):
        model_standard.step()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

    # --- Final Analysis & Plotting ---
    print("\n--- Final Results Summary ---")
    print(f"Total Slots: {model_standard.current_slot_idx + 1}")
    print(f"Total MEV Earned: {model_standard.total_mev_earned:.4f} ETH")
    print(
        f"Avg MEV Earned per Slot: {model_standard.total_mev_earned / (model_standard.current_slot_idx):.4f} ETH"
    )
    model_data = model_standard.datacollector.get_model_vars_dataframe()
    # agent_data = model_standard.datacollector.get_agent_vars_dataframe() # If you were collecting agent data

    print("\n--- Collected Model Data ---")
    print(model_data.head())
    print(model_data.tail())

    # Extract the two series as plain Python lists
    avg_mev_series = model_data["Average_MEV_Earned"].tolist()
    supermaj_series = model_data["Supermajority_Success_Rate"].tolist()

    # Write them out
    with open(f"{dir}/avg_mev.json", "w") as f:
        json.dump(avg_mev_series, f)

    with open(f"{dir}/supermajority_success.json", "w") as f:
        json.dump(supermaj_series, f)

    print("Saved avg_mev.json and supermajority_success.json")

    agent_data = model_standard.datacollector.get_agent_vars_dataframe()

    print("\n--- Agent Data Collected ---")
    print("DataFrame Head:")
    print(agent_data.head())  # Print the first 5 rows to see the structure

    relay_agent_data = agent_data[agent_data["Role"] == "relay_agent"]

    print("\nDataFrame Tail:")
    print(agent_data.tail())  # Print the last 5 rows

    # The DataFrame has a MultiIndex: (Step, AgentID)
    # The 'Step' corresponds to the slot number.
    print("\nDataFrame Info:")
    agent_data.info()

    if isinstance(agent_data.index, pd.MultiIndex):
        agent_data = agent_data.reset_index()

    positions_by_slot = agent_data.groupby("Slot")["Position"].apply(list).reset_index()
    nested_array = positions_by_slot["Position"].tolist()
    # Group by slot and collect lists of per-agent values:
    mev_by_slot = agent_data.groupby("Slot")["MEV_Captured_Slot"].apply(list).tolist()
    attest_by_slot = agent_data.groupby("Slot")["Attestation_Rate"].apply(list).tolist()
    proposal_time_by_slot = (
        agent_data.groupby("Slot")["Proposal Time"].apply(list).tolist()
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

    # Proposer data
    proposer_data = agent_data[agent_data["Role"] == "proposer"]
    proposer_strategy_and_mev = proposer_data[
        ["Slot", "Location_Strategy", "MEV_Captured_Slot"]
    ].to_dict(orient="records")

    with open(f"{dir}/data.json", "w") as f:
        json.dump(nested_array, f)
    with open(f"{dir}/mev_by_slot.json", "w") as f:
        json.dump(mev_by_slot, f)
    with open(f"{dir}/attest_by_slot.json", "w") as f:
        json.dump(attest_by_slot, f)
    with open(f"{dir}/proposal_time_by_slot.json", "w") as f:
        json.dump(proposal_time_by_slot, f)
    with open(f"{dir}/proposer_strategy_and_mev.json", "w") as f:
        json.dump(proposer_strategy_and_mev, f)

    with open(f"{dir}/relay_data.json", "w") as f:
        json.dump(nested_array_relay, f)

    print("Saved data in JSON files in the output directory.")


if __name__ == "__main__":
    # --- Define the Pool of Proposer Strategies ---
    NUM_VALIDATORS = 1000  # Example: Simulate 1000 validators
    # --- Define the number of Relays ---
    NUM_RELAYS = 3  # Example: Simulate 3 relays
    # --- Define the number of slots to simulate ---
    SIM_NUM_SLOTS = 1000  # Example: Simulate 1000 slots
    parser = argparse.ArgumentParser(
        description="Run the MEV-Boost simulation and visualize results."
    )
    parser.add_argument(
        "--num_validators",
        type=int,
        default=NUM_VALIDATORS,
        help="Number of validators to simulate (default: 1000)",
    )
    parser.add_argument(
        "--num_relays",
        type=int,
        default=NUM_RELAYS,
        help="Number of relays to simulate (default: 3)",
    )
    parser.add_argument(
        "--num_slots",
        type=int,
        default=SIM_NUM_SLOTS,
        help="Number of slots to simulate (default: 1000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output data (default: 'output')",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data",
        help="Directory to read input data (default: 'data')",
    )
    args = parser.parse_args()
    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Input data
    validators = pd.read_csv(f"{args.input_dir}/validators.csv")
    # Sample 1000 validators if more than 1000
    if len(validators) > args.num_validators:
        validators = validators.sample(n=args.num_validators, random_state=42)
    gcp_regions = pd.read_csv(f"{args.input_dir}/gcp_regions.csv")
    gcp_latency = pd.read_csv(f"{args.input_dir}/gcp_latency.csv")

    num_validators = len(validators)

    # Run the simulation with the specified parameters
    simulation(
        number_of_validators=num_validators,  # args.num_validators,
        number_of_relays=args.num_relays,
        num_slots=args.num_slots,
        dir=args.output_dir,
        validators=validators,
        gcp_regions=gcp_regions,
        gcp_latency=gcp_latency,
    )
