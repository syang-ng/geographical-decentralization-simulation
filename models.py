import inspect
import math
import random

from collections import deque
from mesa import Model, DataCollector, Agent

from consensus import ConsensusSettings
from constants import (
    CLOUD_VALIDATOR_PERCENTAGE,
    NON_COMPLIANT_VALIDATOR_PERCENTAGE,
)
from distribution import (
    SphericalSpace,
    LatencyGenerator,
    init_distance_matrix,
)
from info_agent import InfoAgent, INFO_PROFILES
from relay_agent import RelayAgent, RELAY_PROFILES
from validator_agent import ValidatorWithMEVBoost, ValidatorWithoutMEVBoost, ValidatorType, ValidatorPreference

# --- EthereumRawModel Class ---

class EthereumRawModel(Model):
    """
    The main simulation model for Ethereum without MEV-Boost, managing validators and information sources.
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
    ):

        # Call the base Model constructor
        super().__init__()

        # --- Store Configuration Parameters (from args or defaults) ---
        self.num_validators = num_validators
        self.validator_profiles = validator_profiles # DataFrame with validator info (lat/lon)
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

        # --- Setup the Space (SphericalSpace) ---
        self.space = SphericalSpace()
        self.space.set_gcp_latency_regions(
            gcp_latency, gcp_regions
        )  # Set GCP latency and regions if provided
        self.distance_matrix = (
            None  # Will be initialized after validator positions are set
        )
        # Set latency generator fast mode
        self.fast_mode = fast_mode
        self.latency_generator = LatencyGenerator(self.fast_mode)

        # Set GCP latency if provided
        self.gcp_latency = gcp_latency
        self.gcp_regions = gcp_regions

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

        # --- Setup DataCollector ---
        self.datacollector = self._setup_datacollector()


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
                "Position": "position",  # Example agent attribute
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
        if not available_validators:
            self.current_proposer_agent = None  # No proposer this slot
            return

        # Randomly select a Proposer from available validators
        self.current_proposer_agent = random.choice(available_validators)

        self.current_attesters = [
            v
            for v in available_validators
            if v.unique_id != self.current_proposer_agent.unique_id
        ]
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

        # if (self.steps * self.consensus_settings.time_granularity_ms) > (self.num_slots * self.consensus_settings.slot_duration_ms):
        #     self.running = False  # Stop the simulation loop

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
        self.validator_locations = []
        validator_index = 0
        for validator_agent in self.validators:
            v = (
                self.validator_profiles.iloc[validator_index]
                if self.validator_profiles is not None and validator_index < len(self.validator_profiles)
                else {"latitude": None, "longitude": None}
            )

            position = (
                self.space.get_coordinate_from_lat_lon(
                    v["latitude"], v["longitude"]
                )
                if v["latitude"] is not None and v["longitude"] is not None
                else self.space.sample_point()
            )
            self.validator_locations.append(position)
            gcp_region = (
                self.space.get_nearest_gcp_region(position, self.gcp_regions)
                if self.gcp_regions is not None
                else None
            )
            validator_agent.set_position(position)
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

        # Initialize distance matrix now that all validator positions are set
        self.distance_matrix = init_distance_matrix(
            self.validator_locations, self.space  # , gcp_latency, gcp_regions
        )
    

class EthereumWithoutMEVBoostModel(EthereumRawModel):
    """
    A subclass of EthereumRawModel that represents Ethereum without MEV-Boost.
    """
    def __init__(self, *args, **kwargs):
        # inspect the arguments to pass only relevant ones to the parent class
        inspected_args = inspect.getfullargspec(EthereumRawModel.__init__).args
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in inspected_args}
        super().__init__(*args, **filtered_kwargs)

        num_info = kwargs.get('num_infos', 3)
        info_profiles = kwargs.get('info_profiles', INFO_PROFILES)

        # --- Create Agents ---
        self.create_validator_agents(ValidatorWithoutMEVBoost)

        # if num_info > 3: # Limit to max 3 information sources for now
        InfoAgent.create_agents(
            model=self,
            n=num_info
        )

        # --- Initialize Agents ---
        self.info_agents = self.agents.select(agent_type=InfoAgent)
        for info_agent, info_profile in zip(self.info_agents, info_profiles):
            info_agent.initialize_with_profile(info_profile)

        # --- Initial Slot Setup (before first step) ---
        self._setup_new_slot()


    def _setup_new_slot(self):
        super()._setup_new_slot()

        # moving decision logic here to ensure it happens after proposer is selected
        prev_gcp_region = self.current_proposer_agent.gcp_region
        is_migrated, action_reason = self.current_proposer_agent.decide_to_migrate()  # Check if proposer should migrate
        new_gcp_region = self.current_proposer_agent.gcp_region
        # Log migration decision
        self.migration_queue.append(is_migrated)
        self.action_reasons.append((action_reason, prev_gcp_region, new_gcp_region))

        [info_agent.update_mev_offer() for info_agent in self.info_agents]
    

class MEVBoostModel(EthereumRawModel):
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
        self.create_validator_agents(ValidatorWithMEVBoost)

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

        # relay updates MEV subsidy
        for relay_agent in self.relay_agents:
            relay_agent.update_subsidy()

        self.current_proposer_agent.estimate_profit()
        prev_gcp_region = self.current_proposer_agent.gcp_region
        is_migrated, action_reason = self.current_proposer_agent.decide_to_migrate()  # Check if proposer should migrate
        new_gcp_region = self.current_proposer_agent.gcp_region
        # Log migration decision
        self.migration_queue.append(is_migrated)
        self.action_reasons.append((action_reason, prev_gcp_region, new_gcp_region))

        # Reset relay's MEV offer for the new slot start
        [relay_agent.update_mev_offer() for relay_agent in self.relay_agents]
