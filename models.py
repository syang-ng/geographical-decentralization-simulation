import inspect
import math
import random

from collections import Counter, deque
from mesa import Model, DataCollector

from consensus import ConsensusSettings
from constants import (
    CLOUD_VALIDATOR_PERCENTAGE,
    NON_COMPLIANT_VALIDATOR_PERCENTAGE,
)
from distribution import LatencyGenerator, GCPLatencyModel
from source_agent import RelayAgent, RELAY_PROFILES, SignalAgent, SIGNAL_PROFILES, RelayType
from validator_agent import SSPValidator, MSPValidator, ValidatorType, ValidatorPreference

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
        collect_full_history=False,
        collect_raw_artifacts=True,
        verbose=False,
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
        self.all_gcp_regions = tuple(self.gcp_latency_model.gcp_regions["Region"].values)

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
        self.current_proposer_agent = None
        self.current_attesters = []
        self.current_attester_regions = ()
        self.required_attesters_for_supermajority = 0

        # Exact-mode caches and lightweight slot histories.
        self.collect_full_history = collect_full_history
        self.collect_raw_artifacts = collect_raw_artifacts
        self.verbose = verbose
        self.slot_latency_params_cache = {}
        self.slot_minimal_needed_time_cache = {}
        self.slot_sorted_attester_latencies = {}
        self.slot_attester_std_ratios = ()
        self.latency_executor = None
        self.latency_executor_workers = 0
        self.slot_model_history = []
        self.slot_proposer_history = []
        self.slot_mev_by_slot = []
        self.slot_estimated_mev_by_slot = []
        self.slot_attest_by_slot = []
        self.slot_proposal_time_by_slot = []
        self.slot_proposal_time_avg = []
        self.slot_attestation_sum = []
        self.slot_region_counter_per_slot = {}
        self.top_regions_final = []
        self.validator_step_order = []

        # --- Setup DataCollector ---
        self.datacollector = self._setup_datacollector() if self.collect_full_history else None


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
                    m.current_proposer_agent.estimated_profit_increase
                    if m.current_proposer_agent
                    else 0.0
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


    def _record_slot_history(self):
        """Captures the slot-level outputs used by downstream JSON and CSV exports."""
        if self.current_slot_idx < 0:
            return

        completed_slots = self.current_slot_idx + 1
        self.slot_model_history.append(
            {
                "Average_MEV_Earned": (
                    self.total_mev_earned / completed_slots if completed_slots > 0 else 0.0
                ),
                "Supermajority_Success_Rate": (
                    (self.supermajority_met_slots / completed_slots) * 100
                    if completed_slots > 0
                    else 0.0
                ),
                "Failed_Block_Proposals": self.failed_block_proposals,
                "Utility_Increase": (
                    self.current_proposer_agent.estimated_profit_increase
                    if self.current_proposer_agent
                    else 0.0
                ),
            }
        )

        mev_by_slot = []
        estimated_mev_by_slot = []
        attest_by_slot = []
        proposal_time_by_slot = []
        proposal_time_positive_values = []
        attestation_sum = 0.0
        region_counts = Counter()

        for validator in self.validator_step_order:
            mev_by_slot.append(validator.mev_captured)
            estimated_mev_by_slot.append(validator.estimated_profit)
            attest_by_slot.append(validator.attestation_rate)
            proposal_time_by_slot.append(validator.proposed_time_ms)

            if validator.proposed_time_ms > 0:
                proposal_time_positive_values.append(validator.proposed_time_ms)

            attestation_sum += validator.attestation_rate
            region_counts[validator.gcp_region] += 1

        self.slot_proposal_time_avg.append(
            (sum(proposal_time_positive_values) / len(proposal_time_positive_values))
            if proposal_time_positive_values
            else 0.0
        )
        self.slot_attestation_sum.append(attestation_sum)
        top_regions = region_counts.most_common()
        self.slot_region_counter_per_slot[self.current_slot_idx] = top_regions
        self.top_regions_final = top_regions

        if self.collect_raw_artifacts:
            self.slot_mev_by_slot.append(mev_by_slot)
            self.slot_estimated_mev_by_slot.append(estimated_mev_by_slot)
            self.slot_attest_by_slot.append(attest_by_slot)
            self.slot_proposal_time_by_slot.append(proposal_time_by_slot)

        if self.current_proposer_agent:
            self.slot_proposer_history.append(
                {
                    "Slot": self.current_slot_idx,
                    "Location_Strategy": (
                        self.current_proposer_agent.location_strategy["type"]
                        if self.current_proposer_agent.location_strategy
                        else "none"
                    ),
                    "MEV_Captured_Slot": self.current_proposer_agent.mev_captured,
                }
            )


    def _setup_new_slot(self):
        """
        Manages the setup for a new logical slot:
        Resets validator states, assigns roles, and updates parameters.
        """
        self.current_slot_idx += 1
        self.slot_latency_params_cache = {}
        self.slot_minimal_needed_time_cache = {}
        self.slot_sorted_attester_latencies = {}

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
            self.current_attesters = []
            self.current_attester_regions = ()
            self.required_attesters_for_supermajority = 0
            self.slot_attester_std_ratios = ()
            return

        # Randomly select a Proposer from available validators
        self.current_proposer_agent = random.choice(available_validators)

        self.current_attesters = [
            v
            for v in available_validators
            if v.unique_id != self.current_proposer_agent.unique_id
        ]
        self.current_attester_regions = tuple(
            attester.gcp_region for attester in self.current_attesters
        )
        self.required_attesters_for_supermajority = math.ceil(
            self.consensus_settings.attestation_threshold
            * len(self.current_attesters)
        )
        self.slot_attester_std_ratios = tuple([0.5] * len(self.current_attesters))
        self.slot_sorted_attester_latencies = {
            region: tuple(
                sorted(
                    self.gcp_latency_model.get_latency(region, attester_region)
                    for attester_region in self.current_attester_regions
                )
            )
            for region in self.all_gcp_regions
        }
        for attester in self.current_attesters:
            attester.set_attester_role()
        # Set the Proposer's role and prepare for the slot
        self.current_proposer_agent.set_proposer_role()


    def get_current_proposer_agent(self):
        """Helper to get the current proposer from the model for attesters."""
        return self.current_proposer_agent


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
            if self.verbose:
                print(f"--- Slot {self.current_slot_idx + 1} Summary ---")
            # --- End of Previous Slot Logic & Rewards ---
            if (
                self.current_proposer_agent
                and self.current_proposer_agent.has_proposed_block
            ):
                slot_successful_attestations = sum(
                    1 for a in self.current_attesters if a.attested_to_proposer_block
                )
                required_attesters_for_supermajority = math.ceil(
                    (self.consensus_settings.attestation_threshold) * len(self.current_attesters)
                )

                self.current_proposer_agent.attestation_rate = (
                    slot_successful_attestations / len(self.current_attesters)
                ) * 100

                if slot_successful_attestations >= required_attesters_for_supermajority:
                    self.current_proposer_agent.mev_captured = (
                        self.current_proposer_agent.mev_captured_potential
                    )
                    self.total_mev_earned += self.current_proposer_agent.mev_captured
                    self.current_proposer_agent.total_mev_captured += self.current_proposer_agent.mev_captured
                    self.supermajority_met_slots += 1

                else:
                    self.current_proposer_agent.mev_captured = (
                        0.0  # No reward if supermajority not met
                    )
                    
                    self.failed_block_proposals += 1 # count failed block proposals

                # update total MEV captured and consensus rewards
                for attester in self.current_attesters:
                    attester.total_consensus_rewards += (
                        self.consensus_settings.timely_source_reward
                        + self.consensus_settings.timely_target_reward
                    )
                    if (slot_successful_attestations >= required_attesters_for_supermajority and attester.attested_to_proposer_block) \
                        or (slot_successful_attestations < required_attesters_for_supermajority and not attester.attested_to_proposer_block):
                        attester.total_consensus_rewards += self.consensus_settings.timely_head_reward

                self.proposed_block_times.append(
                    self.current_proposer_agent.proposed_time_ms
                )
                self.total_successful_attestations += slot_successful_attestations

            self._record_slot_history()
            if self.datacollector is not None:
                self.datacollector.collect(self)

            # --- Setup for New Slot ---
            self._setup_new_slot()  # This calls reset_for_new_slot on agents

        # --- Validators perform their step actions ---
        for validator in self.validator_step_order:
            validator.step()

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
        self.validator_step_order = list(self.validators)
        
        # Set validator type and preferences
        self.validators = list(self.validators)  # Convert to list for shuffling
        random.shuffle(self.validators)
        for validator_agent in self.validators[:int(self.num_validators * self.validator_cloud_percentage)]:
            validator_agent.set_type(ValidatorType.CLOUD)
            validator_agent.migration_cost = self.cost

        random.shuffle(self.validators)
        for validator_agent in self.validators[:int(self.num_validators * self.validator_noncompliant_percentage)]:
            validator_agent.set_validator_preference(ValidatorPreference.NONCOMPLIANT)


    def close(self):
        if self.latency_executor is not None:
            self.latency_executor.shutdown(wait=True)
            self.latency_executor = None
            self.latency_executor_workers = 0
    

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
        self.create_validator_agents(MSPValidator)

        SignalAgent.create_agents(
            model=self,
            n=num_signals
        )

        # --- Initialize Agents ---
        self.signal_agents = self.agents.select(agent_type=SignalAgent)
        self.signal_agents = list(self.signal_agents)
        for signal_agent, signal_profile in zip(self.signal_agents, signal_profiles):
            signal_agent.initialize_with_profile(signal_profile)
        self.signal_agents.sort(key=lambda agent: (agent.unique_id, agent.gcp_region))
        self.signal_latency_by_region = {
            region: {
                signal_agent.unique_id: self.gcp_latency_model.get_latency(
                    region, signal_agent.gcp_region
                )
                for signal_agent in self.signal_agents
            }
            for region in self.all_gcp_regions
        }

        # --- Initial Slot Setup (before first step) ---
        self._setup_new_slot()


    def _setup_new_slot(self):
        super()._setup_new_slot()

        if self.current_proposer_agent is None:
            self.migration_queue.append(False)
            self.action_reasons.append(("no_available_proposer", None, None))
            return

        # moving decision logic here to ensure it happens after proposer is selected
        prev_gcp_region = self.current_proposer_agent.gcp_region
        is_migrated, action_reason = self.current_proposer_agent.decide_to_migrate()  # Check if proposer should migrate
        new_gcp_region = self.current_proposer_agent.gcp_region
        # Log migration decision
        self.migration_queue.append(is_migrated)
        self.action_reasons.append((action_reason, prev_gcp_region, new_gcp_region))



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
        self.relay_agents = list(self.relay_agents)
        for relay_agent, relay_profile in zip(self.relay_agents, relay_profiles):
            relay_agent.initialize_with_profile(relay_profile)
        self.relay_agents.sort(key=lambda agent: (agent.unique_id, agent.gcp_region))
        self.relay_agents_by_id = {
            relay_agent.unique_id: relay_agent for relay_agent in self.relay_agents
        }
        self.relay_latency_by_region = {
            region: {
                relay_agent.unique_id: self.gcp_latency_model.get_latency(
                    region, relay_agent.gcp_region
                )
                for relay_agent in self.relay_agents
            }
            for region in self.all_gcp_regions
        }
        self.censoring_relays = tuple(
            relay_agent
            for relay_agent in self.relay_agents
            if relay_agent.type == RelayType.CENSORING
        )
        relay_regions = {relay_agent.gcp_region for relay_agent in self.relay_agents}
        self.other_gcp_regions = tuple(
            region for region in self.all_gcp_regions if region not in relay_regions
        )

        # --- Initial Slot Setup (before first step) ---
        self._setup_new_slot()


    def _setup_new_slot(self):
        super()._setup_new_slot()

        if self.current_proposer_agent is None:
            self.migration_queue.append(False)
            self.action_reasons.append(("no_available_proposer", None, None))
            return

        prev_gcp_region = self.current_proposer_agent.gcp_region
        is_migrated, action_reason = self.current_proposer_agent.decide_to_migrate()  # Check if proposer should migrate
        new_gcp_region = self.current_proposer_agent.gcp_region
        # Log migration decision
        self.migration_queue.append(is_migrated)
        self.action_reasons.append((action_reason, prev_gcp_region, new_gcp_region))

