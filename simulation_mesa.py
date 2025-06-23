import json
import math
import numpy as np
import pandas as pd
import random
import time

from mesa import Agent, Model, DataCollector

from constants import *
from distribution import (
    SphericalSpace,
    init_distance_matrix,
    update_distance_matrix_for_node,
)
from measure import *

# --- Relay Agent Class Definition ---


class RelayAgent(Agent):
    """
    A simple Relay Agent that has a position and provides the current best MEV offer.
    It doesn't have complex strategies; it's a conduit.
    """

    def __init__(self, model):
        super().__init__(model)
        self.current_mev_offer = 0.0

    def set_position(self, position):
        """Sets the Relay's position in the space."""
        self.position = position

    def update_mev_offer(self):
        """Simulates builders providing better offers to the Relay over time."""
        # Get current time from the model's steps
        # Convert model time steps to milliseconds within the current slot
        current_slot_time_ms = (
            self.model.steps * TIME_GRANULARITY_MS
        ) % SLOT_DURATION_MS
        time_in_seconds = current_slot_time_ms / 1000

        self.current_mev_offer = (
            BASE_MEV_AMOUNT + time_in_seconds * MEV_INCREASE_PER_SECOND
        )

    def get_mev_offer(self):
        """Provides the current best MEV offer to a Proposer."""
        return self.current_mev_offer

    def step(self):
        """
        The Relay Agent's behavior in each simulation step.
        Here, it just updates its MEV offer based on the current slot time.
        """
        self.update_mev_offer()


# --- Validator Agent Class Definition ---


class ValidatorAgent(Agent):
    """
    Represents a single validator, which can be a Proposer or an Attester.
    It has a position, network latency, and strategies for proposing and potentially migrating.
    """

    def __init__(self, model):
        super().__init__(model)

        # State variables, will be reset each slot by the model
        self.role = "none"  # "proposer" or "attester"
        self.network_latency_to_target = (
            -1
        )  # Latency to Relay (for Proposer) or from Relay (for Attester)

        # Migration state
        self.migration_cooldown = 0  # In slots
        self.is_migrating = False
        self.migration_end_time_ms = -1

        # Proposer specific attributes
        self.timing_strategy = None  # Assigned when chosen as proposer for a slot
        self.random_propose_time = -1  # For random delay strategy
        self.attestation_time_ms = (
            ATTESTATION_TIME_MS  # Default attestation time for Attesters
        )
        self.location_strategy = None

        # Slot-specific performance tracking (reset by model per slot, or used for decision-making)
        self.has_proposed_block = False
        self.has_attested = (
            False  # True if this validator has attested in the current slot
        )
        self.proposed_time_ms = -1
        self.mev_captured = 0.0  # Actual MEV earned after supermajority check
        self.mev_captured_potential = 0.0  # Potential MEV before supermajority check
        self.attested_to_proposer_block = (
            False  # True if this attester made a valid attestation for Proposer's block
        )

    def reset_for_new_slot(self):
        """
        Resets the validator's ephemeral state for a new slot.
        This is called by the Model at the start of each slot.
        """
        # Decrement migration cooldown
        if self.migration_cooldown > 0:
            self.migration_cooldown -= 1

        # If migration just ended, finalize it
        # This check happens at the start of a slot (model.steps * TIME_GRANULARITY_MS)
        # compared to the end time of migration.
        # The migration can be completed immediately (current assumption)
        if (
            self.is_migrating
            and (self.model.steps * TIME_GRANULARITY_MS) >= self.migration_end_time_ms
        ):
            self.complete_migration()

        # Reset ephemeral state for new slot activities
        self.role = "none"  # Role will be reassigned by the Model
        self.network_latency_to_target = -1
        self.has_proposed_block = False
        self.proposed_time_ms = -1
        self.mev_captured = 0.0
        self.mev_captured_potential = 0.0
        self.has_attested = False
        self.attested_to_proposer_block = False
        self.random_propose_time = -1  # Reset for next potential proposer role
        self.latency_threshold = -1  # Reset for next potential proposer role

    def set_position(self, position):
        """Sets the validator's position in the space."""
        self.position = position

    def set_index(self, index):
        """Sets the validator's index in the model's agent list."""
        self.index = index
        self.unique_id = f"validator_{index}"

    def set_strategy(self, timing_strategy, location_strategy):
        """Sets the validators' strategies."""
        self.timing_strategy = timing_strategy
        self.location_strategy = location_strategy

    # --- Role Assignment Methods (Called by the Model) ---
    def set_proposer_role(self, relay_position, space_instance):
        """Sets this validator as the Proposer for the current slot."""
        self.role = "proposer"
        distance_to_relay = space_instance.distance(self.position, relay_position)
        self.network_latency_to_target = (
            BASE_NETWORK_LATENCY_MS
            + (distance_to_relay / space_instance.get_max_dist())
            * MAX_ADDITIONAL_NETWORK_LATENCY_MS
        )
        # Set random propose time if using random strategy
        if self.timing_strategy["type"] == "random_delay":
            self.random_propose_time = random.randint(
                self.timing_strategy["min_delay_ms"],
                self.timing_strategy["max_delay_ms"],
            )
        if self.timing_strategy["type"] == "optimal_latency":
            # Calculate the latency threshold for optimal latency strategy
            to_relay_latency = self.network_latency_to_target
            required_attesters_for_supermajority = math.ceil(
                (ATTESTATION_THRESHOLD) * len(self.model.current_attesters)
            )
            relay_to_attester_latency = [
                a.network_latency_to_target + 3 * to_relay_latency
                for a in self.model.current_attesters
            ]
            sorted_latencies = sorted(relay_to_attester_latency)
            self.latency_threshold = (
                ATTESTATION_TIME_MS
                - sorted_latencies[required_attesters_for_supermajority]
            )

    def set_attester_role(
        self, relay_position, space_instance, proposer_is_optimized_latency=False
    ):
        """
        Sets this validator as an Attester for the current slot and calculates its specific latency.
        proposer_is_optimized_latency: If True, this attester assumes the current proposer (via relay) has optimized latency.
        """
        self.role = "attester"

        if proposer_is_optimized_latency:
            self.network_latency_to_target = BASE_NETWORK_LATENCY_MS
        else:
            distance_from_relay = space_instance.distance(self.position, relay_position)
            self.network_latency_to_target = (
                BASE_NETWORK_LATENCY_MS
                + (distance_from_relay / space_instance.get_max_dist())
                * MAX_ADDITIONAL_NETWORK_LATENCY_MS
            )

    # --- In-Slot Behavior Methods (Called from step()) ---
    def decide_and_propose(self, current_slot_time_ms_inner, relay_agent_instance):
        """
        Proposer (this validator) decides whether to propose a block based on its strategy.
        Returns (should_propose, mev_offer_if_proposing)
        """
        if self.has_proposed_block:  # Already proposed or migrating, cannot act
            return False, 0.0

        # Q: We may need to reconsider this wrt the latency to the relay.
        # we can just say the marginal value of time is known hence everyone can compute it for themselves
        # therefore no need to query the relay
        mev_offer = relay_agent_instance.get_mev_offer()

        if self.timing_strategy["type"] == "fixed_delay":
            if current_slot_time_ms_inner >= self.timing_strategy["delay_ms"]:
                return True, mev_offer
        elif self.timing_strategy["type"] == "threshold_and_max_delay":
            if (
                mev_offer >= self.timing_strategy["mev_threshold"]
                or current_slot_time_ms_inner >= self.timing_strategy["max_delay_ms"]
            ):
                return True, mev_offer
        elif self.timing_strategy["type"] == "random_delay":
            if (
                self.random_propose_time == -1
            ):  # Should have been set by model, but for safety
                self.random_propose_time = random.randint(
                    self.timing_strategy["min_delay_ms"],
                    self.timing_strategy["max_delay_ms"],
                )
            if current_slot_time_ms_inner >= self.random_propose_time:
                return True, mev_offer
        elif (
            self.timing_strategy["type"] == "optimal_latency"
        ):  # The proposer knows its latency is optimized
            to_relay_latency = self.network_latency_to_target
            required_attesters_for_supermajority = math.ceil(
                (ATTESTATION_THRESHOLD) * len(self.model.current_attesters)
            )
            relay_to_attester_latency = [
                a.network_latency_to_target + to_relay_latency
                for a in self.model.current_attesters
            ]
            sorted_latencies = sorted(relay_to_attester_latency)
            latency_threshold = (
                ATTESTATION_TIME_MS
                - sorted_latencies[required_attesters_for_supermajority]
            )
            if (
                current_slot_time_ms_inner <= latency_threshold
                and current_slot_time_ms_inner + TIME_GRANULARITY_MS > latency_threshold
            ):
                return True, mev_offer

        return False, 0.0

    def propose_block(self, current_slot_time_ms_inner, mev_offer):
        """Executes the block proposal action for the Proposer."""
        self.has_proposed_block = True
        # Apply latency to the target (relay) to the proposed time
        self.proposed_time_ms = (
            current_slot_time_ms_inner + 3 * self.network_latency_to_target
        )
        # TODO: This should be the mev of current step time + 1 x network latency to target (ie proposer querying the relay for header)
        # relay_agent_instance.get_mev_offer(current_slot_time_ms_inner + 1 * self.network_latency_to_target)
        self.mev_captured_potential = (
            mev_offer  # Store potential MEV before supermajority check
        )

    def decide_and_attest(
        self,
        current_slot_time_ms_inner,
        block_proposed_time_ms,
        relay_to_attester_latency,
    ):
        """
        Attester (this validator) decides whether to attest to the Proposer's block.
        """
        if self.has_attested:  # Already attested or migrating, cannot act
            return

        # TODO: We should also account for how other attesters behave
        # i.e., an attester should not attest if it knows that the block is not getting enough attestations
        if current_slot_time_ms_inner >= ATTESTATION_TIME_MS:
            # According to the current MEV-Boost auctions, the relay broadcasts the block
            # Todo: the proposer also broadcasts its block, which might be closer to some validators
            block_arrival_at_this_attester_ms = (
                block_proposed_time_ms + relay_to_attester_latency
            )

            if (
                block_proposed_time_ms != -1
                and block_arrival_at_this_attester_ms <= ATTESTATION_TIME_MS
            ):
                self.attested_to_proposer_block = True
            else:
                self.attested_to_proposer_block = False
            self.has_attested = True

    # --- Migration Methods ---
    def decide_to_migrate(self):
        """
        Validator decides whether to migrate based on its assigned migration strategy.
        This is called by the Model after a slot where this validator was Proposer.
        """
        if self.is_migrating or self.migration_cooldown > 0:
            return False

        if self.location_strategy["type"] == "never_migrate":
            return False

        if self.location_strategy["type"] == "random_explore":
            if random.random() < self.location_strategy.get(
                "migration_chance_per_slot", 0.01
            ):
                new_position = self.model.space.sample_point()
                self.do_migration(new_position)
                return True
            return False

        elif self.location_strategy["type"] == "optimize_to_center":
            target_pos = None

            if self.location_strategy["target_type"] == "relay":
                target_pos = self.model.relay_agent.position
            elif self.location_strategy["target_type"] == "attesters_geometric_center":
                # Need active attesters list from the model
                current_attesters_in_model = [
                    a
                    for a in self.model.agents
                    if isinstance(a, ValidatorAgent)
                    and a.unique_id != self.unique_id
                    and a.role == "attester"
                    and not a.is_migrating
                ]
                if current_attesters_in_model:
                    target_pos = self.model.space.calculate_geometric_center_of_nodes(
                        current_attesters_in_model
                    )
                else:
                    return False
            else:  # Unknown target_type
                return False

            if target_pos:
                self.do_migration(target_pos)
                return True

            return False

        return False

    def do_migration(self, new_position_coords):
        """Completes the migration process."""
        self.is_migrating = True
        self.migration_cooldown = self.model.migration_cooldown_slots
        self.position = new_position_coords
        self.is_migrating = False  # Migration is completed immediately in this model
        # update the distance matrix
        self.model.validator_locations[self.index] = (
            new_position_coords  # Update model's validator locations
        )
        update_distance_matrix_for_node(
            self.model.distance_matrix,
            self.model.validator_locations,
            self.model.space,
            self.index,
        )
        # update the network latency to the relay
        space_instance = self.model.space
        relay_position = self.model.relay_agent.position
        distance_to_relay = space_instance.distance(self.position, relay_position)
        self.network_latency_to_target = (
            BASE_NETWORK_LATENCY_MS
            + 2
            * (distance_to_relay / space_instance.get_max_dist())
            * MAX_ADDITIONAL_NETWORK_LATENCY_MS
        )

    # --- Mesa's core step method ---
    def step(self):
        """
        The main step method for a Validator Agent, called by the Mesa scheduler.
        """
        # Get current time within the slot
        current_slot_time_ms_inner = (
            self.model.steps * TIME_GRANULARITY_MS
        ) % SLOT_DURATION_MS

        if self.is_migrating:
            # If migrating, the validator does not perform any actions
            return

        if self.role == "proposer":
            should_propose, mev_offer = self.decide_and_propose(
                current_slot_time_ms_inner, self.model.relay_agent
            )
            if should_propose:
                self.propose_block(current_slot_time_ms_inner, mev_offer)
        elif self.role == "attester":
            # Attesters need to know the proposer's block proposed time and their latency to the relay
            proposer_agent = self.model.get_current_proposer_agent()
            if proposer_agent:
                self.decide_and_attest(
                    current_slot_time_ms_inner,
                    proposer_agent.proposed_time_ms,
                    self.network_latency_to_target,
                )


# --- MEVBoostModel Class ---


class MEVBoostModel(Model):
    """
    The main simulation model for MEV-Boost, managing validators and relay.
    """

    def __init__(
        self,
        num_validators,
        timing_strategies_pool,
        location_strategies_pool,
        num_slots,
        proposer_has_optimized_latency,
        base_mev_amount=0.1,
        mev_increase_per_second=0.05,
        migration_cooldown_slots=5,
    ):

        # Call the base Model constructor
        super().__init__()

        # --- Store Configuration Parameters (from args or defaults) ---
        self.num_validators = num_validators
        self.timing_strategies_pool = timing_strategies_pool
        self.location_strategies_pool = location_strategies_pool
        self.num_slots = num_slots
        self.proposer_has_optimized_latency = proposer_has_optimized_latency

        # Global time/MEV/network parameters accessible to agents
        self.base_mev_amount = base_mev_amount
        self.mev_increase_per_second = mev_increase_per_second
        self.migration_cooldown_slots = migration_cooldown_slots

        # --- Setup the Space (SphericalSpace) ---
        self.space = SphericalSpace()
        self.distance_matrix = (
            None  # Will be initialized after validator positions are set
        )

        # --- Create Agents ---
        ValidatorAgent.create_agents(model=self, n=num_validators)
        RelayAgent.create_agents(model=self, n=1)

        self.validator_locations = []
        validator_index = 0

        for agent in self.agents:
            if isinstance(agent, ValidatorAgent):
                position = self.space.sample_point()
                self.validator_locations.append(position)
                agent.set_position(position)
                agent.set_index(validator_index)
                agent.set_strategy(
                    random.choice(self.timing_strategies_pool),
                    random.choice(self.location_strategies_pool),
                )
                validator_index += 1
            elif isinstance(agent, RelayAgent):
                agent.set_position(self.space.sample_point())
                agent.unique_id = "relay_agent"
                agent.role = "relay_agent"
                self.relay_agent = agent
            else:
                continue

        # Find all validators after they have been created and assigned positions
        self.validators = self.agents.select(agent_type=ValidatorAgent)
        # Initialize distance matrix now that all validator positions are set
        self.distance_matrix = init_distance_matrix(
            self.validator_locations, self.space
        )

        # --- Model-Level Tracking Variables ---
        self.current_slot_idx = -1  # Will increment at start of each slot
        self.total_mev_earned = 0.0
        self.supermajority_met_slots = 0
        self.proposed_block_times = []
        self.total_successful_attestations = (
            0  # Raw count of individual successful attestations
        )
        self.total_attesters_counted = (
            0  # Total number of attesters in slots where proposer successfully proposed
        )

        # --- Setup DataCollector ---
        self.datacollector = self._setup_datacollector()

        # --- Initial Slot Setup (before first step) ---
        self._setup_new_slot()

    def _setup_datacollector(self):
        """Configures and returns a Mesa DataCollector."""
        return DataCollector(
            model_reporters={
                "Average_MEV_Earned": lambda m: (
                    m.total_mev_earned / (m.current_slot_idx + 1)
                    if m.current_slot_idx >= 0
                    else 0
                ),
                "Supermajority_Success_Rate": lambda m: (
                    (m.supermajority_met_slots / (m.current_slot_idx + 1)) * 100
                    if m.current_slot_idx >= 0
                    else 0
                ),
            },
            agent_reporters={
                "Position": "position",  # Example agent attribute
                "Role": "role",
                "Slot": "current_slot_idx",
                "MEV_Captured_Slot": "mev_captured",  # MEV actually earned in the last slot
            },
        )

    def _setup_new_slot(self):
        """
        Manages the setup for a new logical slot:
        Resets validator states, assigns roles, and updates parameters.
        """
        self.current_slot_idx += 1

        # Reset all validators for the new slot
        for validator in self.validators:
            validator.current_slot_idx = (
                self.current_slot_idx
            )  # Pass current slot index for migration logic
            validator.reset_for_new_slot()  # Handles cooldown, completes migrations, resets ephemeral state

        # Select Proposer (must not be migrating)
        available_validators = [v for v in self.validators if not v.is_migrating]
        if not available_validators:
            self.current_proposer_agent = None  # No proposer this slot
            # print(f"Slot {self.current_slot_idx + 1}: No validators available to propose (all migrating).")
            return

        self.current_proposer_agent = random.choice(available_validators)
        self.current_proposer_agent.set_proposer_role(
            self.relay_agent.position, self.space
        )
        self.current_proposer_agent.decide_to_migrate()  # Check if proposer should migrate

        # Set Attesters and calculate their specific latencies from the Relay
        self.current_attesters = [
            v
            for v in available_validators
            if v.unique_id != self.current_proposer_agent.unique_id
        ]
        for attester in self.current_attesters:
            attester.set_attester_role(
                self.relay_agent.position,
                self.space,
                self.proposer_has_optimized_latency,
            )

        # Reset relay's MEV offer for the new slot start
        self.relay_agent.update_mev_offer()

    def get_current_proposer_agent(self):
        """Helper to get the current proposer from the model for attesters."""
        return self.current_proposer_agent

    def step(self):
        """
        Advance the simulation by one step (TIME_GRANULARITY_MS).
        """
        # Determine if we are at the start of a new logical slot
        is_new_slot_start = (self.steps * TIME_GRANULARITY_MS) % SLOT_DURATION_MS == 0

        if is_new_slot_start and self.steps > 0:  # Avoid re-setup for time 0
            # --- End of Previous Slot Logic & Rewards ---
            if (
                self.current_proposer_agent
                and self.current_proposer_agent.has_proposed_block
            ):
                slot_successful_attestations = sum(
                    1 for a in self.current_attesters if a.attested_to_proposer_block
                )
                required_attesters_for_supermajority = math.ceil(
                    (ATTESTATION_THRESHOLD) * len(self.current_attesters)
                )

                if slot_successful_attestations >= required_attesters_for_supermajority:
                    self.current_proposer_agent.mev_captured = (
                        self.current_proposer_agent.mev_captured_potential
                    )
                    self.total_mev_earned += self.current_proposer_agent.mev_captured
                    self.supermajority_met_slots += 1
                else:
                    self.current_proposer_agent.mev_captured = (
                        0.0  # No reward if supermajority not met
                    )

                self.proposed_block_times.append(
                    self.current_proposer_agent.proposed_time_ms
                )
                self.total_successful_attestations += slot_successful_attestations

                # Collect data after all agents have acted in this step
                self.datacollector.collect(self)

            # --- Setup for New Slot ---
            self._setup_new_slot()  # This calls reset_for_new_slot on agents

        # --- Agents perform their step actions ---
        self.agents.do("step")
        self.agents.do("advance")

        # Condition to stop simulation early if all agents are migrating or something goes wrong
        if self.steps * TIME_GRANULARITY_MS > (self.num_slots * SLOT_DURATION_MS):
            self.running = False  # Stop the simulation loop


if __name__ == "__main__":
    # --- Simulation Execution ---
    random.seed(0x06511)  # For reproducibility
    np.random.seed(0x06511)  # For reproducibility in NumPy operations
    # --- Define the Pool of Proposer Strategies ---
    NUM_VALIDATORS = 1000  # Example: Simulate 1000 validators
    # --- Define the number of slots to simulate ---
    SIM_NUM_SLOTS = 1000  # Example: Simulate 200 slots
    TOTAL_TIME_STEPS = (
        SIM_NUM_SLOTS * (SLOT_DURATION_MS // TIME_GRANULARITY_MS) + 1
    )  # Total fine-grained steps (200 * 120 = 24000 steps)

    # --- Define Proposer Strategies ---
    # all_timing_strategies = [
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
    MIGRATION_STRATEGY_RANDOM_EXPLORE = {
        "type": "random_explore",
        "migration_chance_per_slot": 0.01,
    }
    MIGRATION_STRATEGY_OPTIMIZE_RELAY = {
        "type": "optimize_to_center",
        "target_type": "relay",
    }
    MIGRATION_STRATEGY_OPTIMIZE_ATTESTERS = {
        "type": "optimize_to_center",
        "target_type": "attesters_geometric_center",
    }

    all_location_strategies = [
        # MIGRATION_STRATEGY_NEVER,
        # MIGRATION_STRATEGY_RANDOM_EXPLORE,
        MIGRATION_STRATEGY_OPTIMIZE_RELAY,
        # MIGRATION_STRATEGY_OPTIMIZE_ATTESTERS,
    ]

    model_params_standard_nomig = {
        "num_validators": NUM_VALIDATORS,
        "timing_strategies_pool": all_timing_strategies,
        "location_strategies_pool": all_location_strategies,
        "num_slots": SIM_NUM_SLOTS,
        "proposer_has_optimized_latency": False,
        "base_mev_amount": BASE_MEV_AMOUNT,
        "mev_increase_per_second": MEV_INCREASE_PER_SECOND,
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

    relay_position = relay_agent_data["Position"].iloc[0]
    # Since relay is not moving, we can just use the first position and multiply it by the number of slots
    # --------------------------------------------
    # build one relay-point list per slot
    # --------------------------------------------
    relay_pos_list = list(relay_position)  # JSON wants lists
    nested_array_relay = [
        [relay_pos_list] for _ in range(len(nested_array))  # <- extra [] !
    ]

    with open("data.json", "w") as f:
        json.dump(nested_array, f)

    with open("relay_data.json", "w") as f:
        json.dump(nested_array_relay, f)
