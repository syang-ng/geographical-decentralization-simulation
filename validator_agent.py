import math
import random

from enum import Enum
from mesa import Agent

from constants import (
    ATTESTATION_THRESHOLD,
    ATTESTATION_TIME_MS,
    BASE_NETWORK_LATENCY_MS,
    SLOT_DURATION_MS,
    TIME_GRANULARITY_MS,
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
        self.attestation_rate = 0.0  # Reset for new slot
        self.has_attested = False
        self.attested_to_proposer_block = False
        self.random_propose_time = -1  # Reset for next potential proposer role
        self.latency_threshold = -1  # Reset for next potential proposer role

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

    # --- Role Assignment Methods (Called by the Model) ---
    def set_proposer_role(self, gcp_latency):
        """Sets this validator as the Proposer for the current slot."""
        self.role = "proposer"
        # GCP Latency
        self.network_latency_to_target = self.model.space.get_latency(
            self.gcp_region, self.model.relay_agent.gcp_region, gcp_latency
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
            # Calculate the latency threshold for optimal latency strategy
            to_relay_latency = self.network_latency_to_target
            required_attesters_for_supermajority = math.ceil(
                (ATTESTATION_THRESHOLD) * len(self.model.current_attesters)
            ) + 30  # Add 30 to ensure we have enough for supermajority
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
            self.network_latency_to_target = BASE_NETWORK_LATENCY_MS
        else:
            self.network_latency_to_target = self.model.space.get_latency(
                self.gcp_region, self.model.relay_agent.gcp_region, gcp_latency
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
            mev_offers.append(r.get_mev_offer_at_time(current_time))
        return mev_offers


    # --- In-Slot Behavior Methods (Called from step()) ---
    def decide_and_propose(self, current_slot_time_ms_inner, relay_agents):
        """
        Proposer (this validator) decides whether to propose a block based on its strategy.
        Returns (should_propose, mev_offer_if_proposing)
        """
        if self.has_proposed_block:  # Already proposed or migrating, cannot act
            return False, 0.0, 0

        # Q: We may need to reconsider this wrt the latency to the relay.
        # we can just say the marginal value of time is known hence everyone can compute it for themselves
        # therefore no need to query the relay
        mev_offer = max(self.get_mev_offer_from_relays(
            current_slot_time_ms_inner, relay_agents
        ), default=0.0)

        if self.timing_strategy["type"] == "fixed_delay":
            if current_slot_time_ms_inner >= self.timing_strategy["delay_ms"]:
                mev_offer = max(self.get_mev_offer_from_relays(
                    self.timing_strategy["delay_ms"], relay_agents
                ), default=0.0)

                return True, mev_offer, self.timing_strategy["delay_ms"]
        elif self.timing_strategy["type"] == "threshold_and_max_delay":
            if (
                mev_offer >= self.timing_strategy["mev_threshold"]
                or current_slot_time_ms_inner >= self.timing_strategy["max_delay_ms"]
            ):
                proposed_time_ms = min(
                    current_slot_time_ms_inner,
                    self.timing_strategy["max_delay_ms"],
                )
                mev_offer = max([
                    r.get_mev_offer_at_time(proposed_time_ms) for r in relay_agents
                ])
                return True, mev_offer, proposed_time_ms
        elif self.timing_strategy["type"] == "random_delay":
            if (
                self.random_propose_time == -1
            ):  # Should have been set by model, but for safety
                self.random_propose_time = random.randint(
                    self.timing_strategy["min_delay_ms"],
                    self.timing_strategy["max_delay_ms"],
                )
            if current_slot_time_ms_inner >= self.random_propose_time:
                mev_offer = max([
                    r.get_mev_offer_at_time(self.random_propose_time) for r in relay_agents
                ])
                return True, mev_offer, self.random_propose_time
        elif (
            self.timing_strategy["type"] == "optimal_latency"
        ):  # The proposer knows its latency is optimized
            if (
                current_slot_time_ms_inner <= self.latency_threshold
                and current_slot_time_ms_inner + TIME_GRANULARITY_MS
                > self.latency_threshold
            ):
                # TODO: This should be the mev of latency_threshold + 1 x network latency to relay (ie proposer querying the relay for header)
                mev_offer = max([
                    r.get_mev_offer_at_time(self.latency_threshold) for r in relay_agents
                ])
                return True, mev_offer, self.latency_threshold

        return False, 0.0, 0

    def propose_block(self, proposed_time, mev_offer):
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
            # TODO: the proposer also broadcasts its block, which might be closer to some validators
            block_arrival_at_this_attester_ms = (
                block_proposed_time_ms
                + 3 * self.model.current_proposer_agent.network_latency_to_target
                + relay_to_attester_latency
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
        Only the validator on the cloud can migrate.
        """
        if self.is_migrating or self.migration_cooldown > 0 or self.type == ValidatorType.HOME:
            return False

        if self.location_strategy["type"] == "never_migrate":
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
                gcp_region = self.model.space.get_nearest_gcp_region(
                    target_pos, gcp_regions
                )
                self.do_migration(target_pos, gcp_region)
                return True

            return False

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
        self.network_latency_to_target = self.model.space.get_latency(
            self.gcp_region, self.model.relay_agent.gcp_region, self.model.gcp_latency
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
            should_propose, mev_offer, proposed_time = self.decide_and_propose(
                current_slot_time_ms_inner, self.model.relay_agents
            )
            if should_propose:
                self.propose_block(proposed_time, mev_offer)
        elif self.role == "attester":
            # Attesters need to know the proposer's block proposed time and their latency to the relay
            proposer_agent = self.model.get_current_proposer_agent()
            if proposer_agent:
                self.decide_and_attest(
                    current_slot_time_ms_inner,
                    proposer_agent.proposed_time_ms,
                    self.model.latency_simulator.get_latency(self.network_latency_to_target, 0.5),
                    # self.network_latency_to_target,
                )

