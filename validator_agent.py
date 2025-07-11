import math
import random

from collections import defaultdict
from enum import Enum
from mesa import Agent

from constants import (
    BASE_NETWORK_LATENCY_MS,
)
from distribution import update_distance_matrix_for_node
from relay_agent import RelayType

# --- Validator Agent Class Definition ---

class ValidatorType(Enum):
    HOME = 1
    CLOUD = 2

class ValidatorPreference(Enum):
    COMPLIANT = 1
    NONCOMPLIANT = 2

class ValidatorAgent(Agent):
    """
    Represents a single validator, which can be a Proposer or an Attester.
    It has a position, network latency, and strategies for proposing and potentially migrating.
    """

    def __init__(self, model):
        super().__init__(model)

        # State variables, will be reset each slot by the model
        self.role = "none"  # "proposer" or "attester"
        self.type = ValidatorType.HOME  # "home staker" or "cloud"
        self.preference = ValidatorPreference.COMPLIANT  # Compliant or Non-compliant
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
        self.latency_threshold = -1  # For optimal latency strategy
        self.location_strategy = None
        self.target_relay = None  # Target relay for migration or attestation

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
        self.attestation_rate = (
            0.0  # Percentage of successful attestations in the current slot
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
            and (self.model.steps * self.model.consensus_settings.time_granularity_ms) >= self.migration_end_time_ms
        ):
            self.complete_migration()

        # Reset ephemeral state for new slot activities
        self.role = "none"  # Role will be reassigned by the Model
        self.network_latency_to_target = {}
        self.has_proposed_block = False
        self.proposed_time_ms = -1
        self.mev_captured = 0.0
        self.mev_captured_potential = 0.0
        self.attestation_rate = 0.0  # Reset for new slot
        self.has_attested = False
        self.attested_to_proposer_block = False
        self.random_propose_time = -1  # Reset for next potential proposer role
        self.latency_threshold = -1  # Reset for next potential proposer role
        self.relay_id = None  # Reset relay ID for attesters
        self.target_relay = None  # Reset target relay for migration decisions
        

    def set_type(self, validator_type):
        self.type = validator_type

    def set_position(self, position):
        """Sets the validator's position in the space."""
        self.position = position

    def set_gcp_region(self, gcp_region):
        """Sets the validator's GCP region for latency calculations."""
        self.gcp_region = gcp_region

    def set_index(self, index):
        """Sets the validator's index in the model's agent list."""
        self.index = index
        self.unique_id = f"validator_{index}"

    def set_strategy(self, timing_strategy, location_strategy):
        """Sets the validators' strategies."""
        self.timing_strategy = timing_strategy
        self.location_strategy = location_strategy

    def set_target_relay(self, relay_agent):
        """Sets the target relay for this validator."""
        self.target_relay = relay_agent

    def set_validator_preference(self, preference):
        """Sets the validator's preference for relay types."""
        if isinstance(preference, ValidatorPreference):
            self.preference = preference
        else:
            raise ValueError("Preference must be an instance of ValidatorPreference Enum")

    # --- Role Assignment Methods (Called by the Model) ---
    def set_proposer_role(self, gcp_latency):
        """Sets this validator as the Proposer for the current slot."""
        self.role = "proposer"
        # GCP Latency
        self.network_latency_to_target = {}
        for relay_agent in self.model.relay_agents:
            self.network_latency_to_target[relay_agent.unique_id] = self.model.space.get_latency(
                self.gcp_region, relay_agent.gcp_region, gcp_latency
            )

        # Set random propose time if using random strategy
        if self.timing_strategy["type"] == "random_delay":
            self.random_propose_time = random.randint(
                self.timing_strategy["min_delay_ms"],
                self.timing_strategy["max_delay_ms"],
            )

    def calculate_latency_threshold(self):
        """Calculates the latency threshold for the Proposer based on its timing strategy."""
        if self.timing_strategy["type"] == "optimal_latency":
            if self.target_relay is None:
                self.target_relay = list(self.model.relay_agents)[0]  # Default to first relay if none set
            # Calculate the latency threshold for optimal latency strategy
            to_relay_latency = self.network_latency_to_target[self.target_relay.unique_id]
            required_attesters_for_supermajority = math.ceil(
                (self.model.consensus_settings.attestation_threshold) * len(self.model.current_attesters)
            ) 
            relay_to_attester_latency = [
                a.network_latency_to_target[self.target_relay.unique_id] + 3 * to_relay_latency
                for a in self.model.current_attesters
            ]
            sorted_latencies = sorted(relay_to_attester_latency)
            self.latency_threshold = (
                self.model.consensus_settings.attestation_time_ms
                - sorted_latencies[required_attesters_for_supermajority]
                - 50  # 50ms buffer to account for network latency and processing time
            )

    def set_attester_role(
        self,
        proposer_is_optimized_latency=False,
        gcp_latency=None,
    ):
        """
        Sets this validator as an Attester for the current slot and calculates its specific latency.
        proposer_is_optimized_latency: If True, this attester assumes the current proposer (via relay) has optimized latency.
        """
        self.role = "attester"

        if proposer_is_optimized_latency:
            self.network_latency_to_target = defaultdict(lambda: BASE_NETWORK_LATENCY_MS)
        else:
            for relay_agent in self.model.relay_agents:
                self.network_latency_to_target[relay_agent.unique_id] = self.model.space.get_latency(
                    self.gcp_region, relay_agent.gcp_region, gcp_latency
                )

    def get_mev_offer_from_relays(
        self,
        current_time,
        relay_agents
    ):
        """
        Proposer queries all relays for their current MEV offers.
        Returns a list of MEV offers from all relays.
        """
        mev_offers = []
        for r in relay_agents:
            if self.preference == ValidatorPreference.COMPLIANT and r.type != RelayType.CENSORING:
                continue
            mev_offers.append((r.get_mev_offer_at_time(current_time), r.unique_id))
        return mev_offers

    def get_best_mev_offer_from_relays(
        self,
        current_time,
        relay_agents
    ):
        """
        Proposer queries all relays for their current MEV offers.
        Returns the best MEV offer from all relays.
        """
        mev_offers = self.get_mev_offer_from_relays(current_time, relay_agents)
        if len(mev_offers) == 0:
            return 0.0, None
        return max(mev_offers, key=lambda x: x[0])

    # --- In-Slot Behavior Methods (Called from step()) ---
    def decide_and_propose(self, current_slot_time_ms_inner, relay_agents):
        """
        Proposer (this validator) decides whether to propose a block based on its strategy.
        Returns (should_propose, mev_offer_if_proposing)
        """
        if self.has_proposed_block:  # Already proposed or migrating, cannot act
            return False, 0.0, None, 0

        # Q: We may need to reconsider this wrt the latency to the relay.
        # we can just say the marginal value of time is known hence everyone can compute it for themselves
        # therefore no need to query the relay
        mev_offer, relay_id = self.get_best_mev_offer_from_relays(
            current_slot_time_ms_inner, relay_agents
        )
        
        if self.timing_strategy["type"] == "fixed_delay":
            if current_slot_time_ms_inner >= self.timing_strategy["delay_ms"]:
                mev_offer, relay_id = self.get_best_mev_offer_from_relays(
                    self.timing_strategy["delay_ms"],
                    relay_agents
                )

                return True, mev_offer, relay_id, self.timing_strategy["delay_ms"]
        elif self.timing_strategy["type"] == "threshold_and_max_delay":
            if (
                mev_offer >= self.timing_strategy["mev_threshold"]
                or current_slot_time_ms_inner >= self.timing_strategy["max_delay_ms"]
            ):
                proposed_time_ms = min(
                    current_slot_time_ms_inner,
                    self.timing_strategy["max_delay_ms"],
                )
                mev_offer, relay_id = self.get_best_mev_offer_from_relays(
                    proposed_time_ms,
                    relay_agents
                )

                return True, mev_offer, relay_id, proposed_time_ms
        elif self.timing_strategy["type"] == "random_delay":
            if (
                self.random_propose_time == -1
            ):  # Should have been set by model, but for safety
                self.random_propose_time = random.randint(
                    self.timing_strategy["min_delay_ms"],
                    self.timing_strategy["max_delay_ms"],
                )
            if current_slot_time_ms_inner >= self.random_propose_time:
                mev_offer, relay_id = self.get_best_mev_offer_from_relays(
                    self.random_propose_time,
                    relay_agents
                )
                return True, mev_offer, relay_id, self.random_propose_time
        elif (
            self.timing_strategy["type"] == "optimal_latency"
        ):  # The proposer knows its latency is optimized
            if (
                current_slot_time_ms_inner <= self.latency_threshold
                and current_slot_time_ms_inner + self.model.consensus_settings.time_granularity_ms
                > self.latency_threshold
            ):
                # TODO: This should be the mev of latency_threshold + 1 x network latency to relay (ie proposer querying the relay for header)
                mev_offer, relay_id = self.get_best_mev_offer_from_relays(
                    self.latency_threshold,
                    relay_agents
                )
                return True, mev_offer, relay_id, self.latency_threshold

        return False, 0.0, None, 0

    def propose_block(self, proposed_time, mev_offer, relay_id):
        """Executes the block proposal action for the Proposer."""
        self.has_proposed_block = True
        # Apply latency to the target (relay) to the proposed time
        # self.proposed_time_ms = (
        #     current_slot_time_ms_inner + 3 * self.network_latency_to_target
        # )
        self.proposed_time_ms = proposed_time
        self.mev_captured_potential = (
            mev_offer  # Store potential MEV before supermajority check
        )
        self.relay_id = relay_id  # Store the relay ID for attesters

    def decide_and_attest(
        self,
        current_slot_time_ms_inner,
        block_proposed_time_ms,
        proposer_to_relay_latency,
        relay_to_attester_latency,
    ):
        """
        Attester (this validator) decides whether to attest to the Proposer's block.
        """
        if self.has_attested:  # Already attested or migrating, cannot act
            return

        # TODO: We should also account for how other attesters behave
        # i.e., an attester should not attest if it knows that the block is not getting enough attestations
        if current_slot_time_ms_inner >= self.model.consensus_settings.attestation_time_ms:
            # According to the current MEV-Boost auctions, the relay broadcasts the block
            # TODO: the proposer also broadcasts its block, which might be closer to some validators
            block_arrival_at_this_attester_ms = (
                block_proposed_time_ms
                + proposer_to_relay_latency
                + relay_to_attester_latency
            )

            # print(block_proposed_time_ms, block_arrival_at_this_attester_ms, self.model.consensus_settings.attestation_time_ms)

            if (
                block_proposed_time_ms != -1
                and block_arrival_at_this_attester_ms <= self.model.consensus_settings.attestation_time_ms
            ):
                self.attested_to_proposer_block = True
            else:
                self.attested_to_proposer_block = False
            self.has_attested = True

    # --- Migration Methods ---
    def how_to_migrate(self):
        attesters = self.model.current_attesters
        simulation_results = []

        for relay_agent in self.model.relay_agents:
            if relay_agent.type == RelayType.CENSORING and self.preference == ValidatorPreference.COMPLIANT:
                continue
            to_relay_latency = self.network_latency_to_target[relay_agent.unique_id]
            required_attesters_for_supermajority = math.ceil(
                (self.model.consensus_settings.attestation_threshold) * len(attesters)
            )
            
            relay_to_attester_latency = [
                a.network_latency_to_target[relay_agent.unique_id]
                for a in self.model.current_attesters
            ]
            sorted_latencies = sorted(relay_to_attester_latency)
            latency_threshold = (
                self.model.consensus_settings.attestation_time_ms
                - sorted_latencies[required_attesters_for_supermajority]
                - 50 # 50ms buffer to account for network latency and processing time
            )
            mev_offer = relay_agent.get_mev_offer_at_time(latency_threshold)
            simulation_results.append(
                {
                    "relay": relay_agent,
                    "latency_threshold": latency_threshold,
                    "mev_offer": mev_offer,
                    "to_relay_latency": to_relay_latency,
                    "relay_to_attester_latency": sorted_latencies[required_attesters_for_supermajority],
                }
            )
        # Sort by MEV offer, then by latency threshold
        simulation_results.sort(key=lambda x: (-x["mev_offer"], x["latency_threshold"]))
        return simulation_results[0]


    def decide_to_migrate(self):
        """
        Validator decides whether to migrate based on its assigned migration strategy.
        This is called by the Model after a slot where this validator was Proposer.
        Only the validator on the cloud can migrate.
        """
        if self.is_migrating or self.migration_cooldown > 0 or self.type == ValidatorType.HOME:
            return False

        if self.location_strategy["type"] == "never_migrate":
            return False

        elif self.location_strategy["type"] == "random_relay":
            self.target_relay = random.choice(self.model.relay_agents)
            target_pos = self.target_relay.position
            gcp_region = self.model.space.get_nearest_gcp_region(
                target_pos, self.model.gcp_regions
            )
            self.do_migration(target_pos, gcp_region)
            return True
        
        elif self.location_strategy["type"] == "target_relay":
            for relay_agent in self.model.relay_agents:
                if relay_agent.unique_id == self.location_strategy["target_relay"]:
                    self.target_relay = relay_agent
                    break

            target_pos = self.target_relay.position
            gcp_region = self.model.space.get_nearest_gcp_region(
                target_pos, self.model.gcp_regions
            )
            if self.model.space.distance(self.position, target_pos) > 0:
                self.do_migration(target_pos, gcp_region)
                return True
            else:
                return False
            
        elif self.location_strategy["type"] == "optimize_to_center":
            target_region, target_pos = self.model.space.get_best_region_to_targets(
                [relay_agent.gcp_region for relay_agent in self.model.relay_agents if relay_agent.type != RelayType.CENSORING or self.preference != ValidatorPreference.COMPLIANT ],
                self.model.gcp_latency,
                self.model.gcp_regions,
            )

            if target_region is not None and self.model.space.distance(self.position, target_pos) > 0:
                self.do_migration(target_pos, target_region)
                return True
            
        elif self.location_strategy["type"] == "best_relay":
            simulation_result = self.how_to_migrate()
            self.target_relay = simulation_result["relay"]
            target_pos = self.target_relay.position
            gcp_region = self.target_relay.gcp_region
            if self.model.space.distance(self.position, target_pos) > 0:
                self.do_migration(target_pos, gcp_region)
                return True

        return False

    def do_migration(self, new_position_coords, new_gcp_region):
        """Completes the migration process."""
        self.is_migrating = True
        self.migration_cooldown = self.model.migration_cooldown_slots
        self.position = new_position_coords
        self.gcp_region = new_gcp_region
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
        # space_instance = self.model.space
        # relay_position = self.model.relay_agent.position
        # distance_to_relay = space_instance.distance(self.position, relay_position)
        self.network_latency_to_target = {}
        for relay_agent in self.model.relay_agents:
            self.network_latency_to_target[relay_agent.unique_id] = self.model.space.get_latency(
                self.gcp_region, relay_agent.gcp_region, self.model.gcp_latency
            )

    # dummy method for now, we complete migration immediately
    def complete_migration(self):
        pass


    # --- Mesa's core step method ---
    def step(self):
        """
        The main step method for a Validator Agent, called by the Mesa scheduler.
        """
        # Get current time within the slot
        current_slot_time_ms_inner = (
            self.model.steps * self.model.consensus_settings.time_granularity_ms
        ) % self.model.consensus_settings.slot_duration_ms

        if self.is_migrating:
            # If migrating, the validator does not perform any actions
            return

        if self.role == "proposer":
            should_propose, mev_offer, relay_id, proposed_time = self.decide_and_propose(
                current_slot_time_ms_inner, self.model.relay_agents
            )
            if should_propose:
                self.propose_block(proposed_time, mev_offer, relay_id)
        elif self.role == "attester":
            # Attesters need to know the proposer's block proposed time and their latency to the relay
            proposer_agent = self.model.get_current_proposer_agent()
            proposer_to_relay_latency = proposer_agent.network_latency_to_target[proposer_agent.relay_id] if proposer_agent and proposer_agent.relay_id else 0
            avg_latency_to_relay = self.network_latency_to_target.get(proposer_agent.relay_id, BASE_NETWORK_LATENCY_MS)

            if proposer_agent:
                self.decide_and_attest(
                    current_slot_time_ms_inner,
                    proposer_agent.proposed_time_ms,
                    proposer_to_relay_latency,
                    self.model.latency_generator.get_latency(avg_latency_to_relay, 0.5),
                )
