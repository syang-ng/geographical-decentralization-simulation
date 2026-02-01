import inspect
import math
import random

from collections import deque
from mesa import Model, DataCollector

from consensus import ConsensusSettings
from constants import (
    CLOUD_VALIDATOR_PERCENTAGE,
    NON_COMPLIANT_VALIDATOR_PERCENTAGE,
)
from distribution import LatencyGenerator, GCPLatencyModel
from source_agent import RelayAgent, RELAY_PROFILES, SignalAgent, SIGNAL_PROFILES
from validator_agent import SSPValidator, MSPValidator, MCPValidator, ValidatorType, ValidatorPreference

# --- Basic: Raw Ethereum ---

class EthereumRawModel(Model):
    """
    The main simulation model for the multi-source paradigm (SSP), managing validators and signals.
    """

    def __init__(
        self,
        num_validators,
        timing_strategies_pool,
        location_strategies_pool,
        num_slots,
        validator_profiles=None,
        migration_cooldown_slots=5,
        gcp_regions=None,
        gcp_latency=None,
        consensus_settings=ConsensusSettings(),
        time_window=10,
        fast_mode=False,
        cost=0.0001,
        validator_cloud_percentage=CLOUD_VALIDATOR_PERCENTAGE,
        validator_noncompliant_percentage=NON_COMPLIANT_VALIDATOR_PERCENTAGE,
        num_proposers_per_slot=1,
        proposer_mode=None
    ):

        # Call the base Model constructor
        super().__init__()

        # --- Store Configuration Parameters (from args or defaults) ---
        self.num_validators = num_validators
        self.validator_profiles = validator_profiles # DataFrame with validator (lat/lon)
        self.timing_strategies_pool = timing_strategies_pool
        self.location_strategies_pool = location_strategies_pool
        self.num_slots = num_slots

        # Global time/MEV/network parameters accessible to agents
        self.migration_cooldown_slots = migration_cooldown_slots
        self.cost = cost
        self.validator_cloud_percentage = validator_cloud_percentage
        self.validator_noncompliant_percentage = validator_noncompliant_percentage

        # Consensus parameters
        self.consensus_settings = consensus_settings

        # Set the queue to count the number of validators that have migrated within the last time window
        self.migration_queue = deque(maxlen=time_window)
        self.action_reasons = []

        # Set latency generator fast mode
        self.fast_mode = fast_mode
        self.latency_generator = LatencyGenerator(self.fast_mode)

        # Set GCP latency if provided
        self.gcp_latency_model = GCPLatencyModel(gcp_latency, gcp_regions)

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
        self.attestation_rate = 0.0  # Calculated as a percentage
        self.failed_block_proposals = 0  # Count of failed block proposals
        self.region_profits = []

        # --- Setup DataCollector ---
        self.datacollector = self._setup_datacollector()

        self.num_proposers_per_slot = num_proposers_per_slot
        self.current_proposer_agents = []
        self.current_proposer_agent = None
        self.proposer_mode = proposer_mode


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
                "Failed_Block_Proposals": "failed_block_proposals",
                "Utility_Increase": lambda m: (
                    sum(p.estimated_profit_increase for p in m.current_proposer_agents or [])
                ),
            },
            agent_reporters={
                "Role": "role",
                "Slot": "current_slot_idx",
                "MEV_Captured_Slot": "mev_captured",  # MEV actually earned in the last slot
                "Estimated_Profit": "estimated_profit",  # Estimated profit before migration
                "Attestation_Rate": "attestation_rate",  # Percentage of successful attestations
                "Proposal Time": "proposed_time_ms",  # Time when the block was proposed,
                "Location_Strategy": lambda v: (
                    v.location_strategy["type"] if v.role == "proposer" else "none"
                ),
                "GCP_Region": "gcp_region",
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
        if len(available_validators) < self.num_proposers_per_slot:
            self.current_proposer_agents = []
            self.current_proposer_agent = None
            self.current_attesters = available_validators
            for a in self.current_attesters:
                a.set_attester_role()
            return

        # Randomly select the proposer(s) from available validators
        if self.num_proposers_per_slot == 1:
            self.current_proposer_agent = random.choice(available_validators)
            self.current_proposer_agents = [self.current_proposer_agent]
        else:
            self.current_proposer_agents = random.sample(available_validators, k=self.num_proposers_per_slot)
            self.current_proposer_agent = self.current_proposer_agents[0]

        self.current_attesters = [
            v
            for v in available_validators
            if v.unique_id not in [p.unique_id for p in self.current_proposer_agents]
        ]
        for attester in self.current_attesters:
            attester.set_attester_role()
        
        for proposer in self.current_proposer_agents:
            proposer.set_proposer_role()


    def get_current_proposer_agent(self):
        """Helper to get the current proposer from the model for attesters."""
        return self.current_proposer_agent
    
    def get_current_proposer_agents(self):
        """Helper to get the current proposers from the model for attesters (MCP setting)."""
        return self.current_proposer_agents


    def get_current_attesters(self):
        """Helper to get the current attesters from the model for proposer."""
        return self.current_attesters

    def step(self):
        """
        Advance the simulation by one step (TIME_GRANULARITY_MS).
        """
        # Determine if we are at the start of a new logical slot
        is_new_slot_start = (self.steps * self.consensus_settings.time_granularity_ms) % self.consensus_settings.slot_duration_ms == 0

        if is_new_slot_start and self.steps > 0:  # Avoid re-setup for time 0
            print(f"--- Slot {self.current_slot_idx + 1} Summary ---")
            # --- End of Previous Slot Logic & Rewards ---

            # MCP-compatible (handles >1 proposers while preserving old behaviour for 1 proposer)
            if self.current_proposer_agents:
                required_attesters_for_supermajority = math.ceil(
                    (self.consensus_settings.attestation_threshold) * len(self.current_attesters)
                )

                # Only consider proposers that actually proposed
                proposers_that_proposed = [p for p in self.current_proposer_agents if p.has_proposed_block]

                # update total MEV captured and consensus rewards
                # Attesters get timely source+target once if at least one proposer proposed,
                # and timely head per proposed block if they correctly attested for that block
                if proposers_that_proposed:
                    for attester in self.current_attesters:
                        attester.total_consensus_rewards += (
                            self.consensus_settings.timely_source_reward
                            + self.consensus_settings.timely_target_reward
                        )

                # For each proposer, evaluate attestations and rewards for that proposer's block
                for proposer in proposers_that_proposed:
                    slot_successful_attestations = sum(
                        1 for a in self.current_attesters
                        if a.attested_to_blocks.get(proposer.unique_id, False)
                    )

                    proposer.attestation_rate = (
                        slot_successful_attestations / len(self.current_attesters)
                    ) * 100

                    if slot_successful_attestations >= required_attesters_for_supermajority:
                        proposer.mev_captured = proposer.mev_captured_potential
                        self.total_mev_earned += proposer.mev_captured
                        proposer.total_mev_captured += proposer.mev_captured
                        self.supermajority_met_slots += 1  # NOTE: in MCP this is per proposer, not per slot
                    else:
                        proposer.mev_captured = 0.0  # No reward if supermajority not met
                        self.failed_block_proposals += 1  # NOTE: in MCP this is per proposer, not per slot

                    # Timely head reward is per proposed block and depends on correctness
                    for attester in self.current_attesters:
                        did_attest = attester.attested_to_blocks.get(proposer.unique_id, False)
                        if (
                            (slot_successful_attestations >= required_attesters_for_supermajority and did_attest)
                            or (slot_successful_attestations < required_attesters_for_supermajority and not did_attest)
                        ):
                            attester.total_consensus_rewards += self.consensus_settings.timely_head_reward

                    self.proposed_block_times.append(proposer.proposed_time_ms)
                    self.total_successful_attestations += slot_successful_attestations

            # Collect data after all agents have acted in this step
            self.datacollector.collect(self)

            # --- Setup for New Slot ---
            self._setup_new_slot()  # This calls reset_for_new_slot on agents

        # --- Agents perform their step actions ---
        self.agents.do("step")
        self.agents.do("advance")

        # Condition to stop simulation if no validators are migrating within the time window
        if len(self.migration_queue) == self.migration_queue.maxlen and not any(self.migration_queue):
            self.running = False


    def get_validator_region_percentage(self, gcp_region):
        """
        Returns the percentage of validators in a specific GCP region.
        """
        total_validators = len(self.validators)
        if total_validators == 0:
            return 0.0

        region_count = sum(
            1 for v in self.validators if v.gcp_region == gcp_region
        )
        return (region_count / total_validators)
    

    def update_validator_profiles(self):
        # Initialize the validators list and their positions
        validator_index = 0

        for validator_agent in self.validators:
            if "gcp_region" in self.validator_profiles.columns:
                gcp_region = self.validator_profiles.iloc[validator_index]["gcp_region"]
            else:
                lat = self.validator_profiles.iloc[validator_index]["latitude"]
                lon = self.validator_profiles.iloc[validator_index]["longitude"]
                gcp_region = self.gcp_latency_model.get_nearest_gcp_region(lat, lon)

            validator_agent.set_gcp_region(gcp_region)
            validator_agent.set_index(validator_index)

            validator_agent.set_strategy(
                random.choice(self.timing_strategies_pool),
                random.choice(self.location_strategies_pool),
            )

            validator_index += 1


    def create_validator_agents(self, ValidatorAgent):
        """
        Creates and adds validator agents to the model.
        """
        ValidatorAgent.create_agents(
            model=self,
            n=self.num_validators,
        )

        # Find all validators after they have been created and assigned positions
        self.validators = self.agents.select(agent_type=ValidatorAgent)
        self.update_validator_profiles()
        
        # Set validator type and preferences
        self.validators = list(self.validators)  # Convert to list for shuffling
        random.shuffle(self.validators)
        for validator_agent in self.validators[:int(self.num_validators * self.validator_cloud_percentage)]:
            validator_agent.set_type(ValidatorType.CLOUD)
            validator_agent.migration_cost = self.cost

        random.shuffle(self.validators)
        for validator_agent in self.validators[:int(self.num_validators * self.validator_noncompliant_percentage)]:
            validator_agent.set_validator_preference(ValidatorPreference.NONCOMPLIANT)
    

# Multi-Source Paradigm (MSP) Model
class MultiSourceParadigm(EthereumRawModel):
    """
    A subclass of EthereumRawModel that represents Ethereum without MEV-Boost.
    """
    def __init__(self, *args, **kwargs):
        # inspect the arguments to pass only relevant ones to the parent class
        inspected_args = inspect.getfullargspec(EthereumRawModel.__init__).args
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in inspected_args}
        super().__init__(*args, **filtered_kwargs)

        num_signals = kwargs.get('num_signals', 3)
        signal_profiles = kwargs.get('signal_profiles', SIGNAL_PROFILES)

        # --- Create Agents ---
        ValidatorCls = MCPValidator if self.proposer_mode == "MCP" else MSPValidator
        self.create_validator_agents(ValidatorCls)

        SignalAgent.create_agents(
            model=self,
            n=num_signals
        )

        # --- Initialize Agents ---
        self.signal_agents = self.agents.select(agent_type=SignalAgent)
        for signal_agent, signal_profile in zip(self.signal_agents, signal_profiles):
            signal_agent.initialize_with_profile(signal_profile)

        # --- Initial Slot Setup (before first step) ---
        self._setup_new_slot()
    
    def _log_region_profits_mcp(self):
        # logs region profits once per slot using final signal_user_counts
        if self.signal_user_counts is None:
            return
        # pick any proposer agent to run the simulation helper
        proposer = self.current_proposer_agents[0] if self.current_proposer_agents else None
        if proposer is None:
            return

        simulation_results = proposer.simulation_with_signals(self.signal_user_counts, update_state=False, marginal=False)

        for row in simulation_results:
            row["slot"] = self.current_slot_idx
        self.region_profits += simulation_results

    def _setup_new_slot(self):
        super()._setup_new_slot()

        # moving decision logic here to ensure it happens after proposer is selected
        any_migrated = False
        if self.proposer_mode == "MCP":
            # MCP setting
            # Value from signal sources will depend on number of proposers using it
            self.signal_user_counts = {s.unique_id: 0 for s in self.signal_agents}
            for proposer in self.current_proposer_agents:
                prev_gcp_region = proposer.gcp_region
                is_migrated, action_reason = proposer.decide_to_migrate(
                    signal_user_counts=self.signal_user_counts
                )
                if is_migrated:
                    print(f"Proposer {proposer.unique_id} migrated from {prev_gcp_region} to {proposer.gcp_region} for Slot {self.current_slot_idx}")
                any_migrated = any_migrated or is_migrated
                self.action_reasons.append((action_reason, prev_gcp_region, proposer.gcp_region))

            self._log_region_profits_mcp()             
        else:
            prev_gcp_region = self.current_proposer_agent.gcp_region
            any_migrated, action_reason = self.current_proposer_agent.decide_to_migrate()
            new_gcp_region = self.current_proposer_agent.gcp_region
            self.action_reasons.append((action_reason, prev_gcp_region, new_gcp_region))

        self.migration_queue.append(any_migrated)
        [signal_agent.update_mev_offer() for signal_agent in self.signal_agents]



# --- Single-Source Paradigm (SSP) Model ---
class SingleSourceParadigm(EthereumRawModel):
    """
    A subclass of EthereumRawModel that represents Ethereum with MEV-Boost.
    """
    def __init__(self, *args, **kwargs):
        # inspect the arguments to pass only relevant ones to the parent class
        inspected_args = inspect.getfullargspec(EthereumRawModel.__init__).args
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in inspected_args}
        super().__init__(*args, **filtered_kwargs)

        num_relays = kwargs.get('num_relays', 3)
        relay_profiles = kwargs.get('relay_profiles', RELAY_PROFILES)

        # --- Create Agents ---
        self.create_validator_agents(SSPValidator)

        RelayAgent.create_agents(
            model=self,
            n=num_relays
        )

        # --- Initialize Agents ---
        self.relay_agents = self.agents.select(agent_type=RelayAgent)
        for relay_agent, relay_profile in zip(self.relay_agents, relay_profiles):
            relay_agent.initialize_with_profile(relay_profile)

        # --- Initial Slot Setup (before first step) ---
        self._setup_new_slot()


    def _setup_new_slot(self):
        super()._setup_new_slot()

        any_migrated = False
        for proposer in self.current_proposer_agents:
            prev_gcp_region = proposer.gcp_region
            is_migrated, action_reason = proposer.decide_to_migrate()    # Check if proposer should migrate
            new_gcp_region = proposer.gcp_region
            # Log migration decision
            any_migrated = any_migrated or is_migrated
            self.action_reasons.append((action_reason, prev_gcp_region, new_gcp_region))

        self.migration_queue.append(any_migrated)

        # Reset relay's MEV offer for the new slot start
        [relay_agent.update_mev_offer() for relay_agent in self.relay_agents]
