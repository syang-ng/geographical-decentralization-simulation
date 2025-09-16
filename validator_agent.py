import math
import random

from collections import defaultdict
from enum import Enum
from mesa import Agent
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

from constants import (
    BASE_NETWORK_LATENCY_MS,
)
from distribution import update_distance_matrix_for_node, find_min_threshold, find_min_threshold_fast
from relay_agent import RelayType

# --- Validator Agent Class Definition ---

class ValidatorType(Enum):
    HOME = 1
    CLOUD = 2


# a validator in US is compliant
# if it moves to EU, it can be non-compliant
class ValidatorPreference(Enum):
    COMPLIANT = 1
    NONCOMPLIANT = 2


class RawValidatorAgent(Agent):
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

        # Migration state
        self.migration_cooldown = 0  # In slots
        self.is_migrating = False
        self.migration_end_time_ms = -1
        self.migration_cost = 0.0

        # Proposer specific attributes
        self.timing_strategy = None  # Assigned when chosen as proposer for a slot
        self.location_strategy = None

        # Slot-specific performance tracking (reset by model per slot, or used for decision-making)
        self.has_proposed_block = False
        self.has_attested = (
            False  # True if this validator has attested in the current slot
        )
        self.proposed_time_ms = -1
        self.estimated_profit = 0.0
        self.estimated_profit_increase = 0.0
        self.mev_captured = 0.0  # Actual MEV earned after supermajority check
        self.mev_captured_potential = 0.0  # Potential MEV before supermajority check
        self.total_mev_captured = 0.0  # Total MEV captured over the simulation
        self.total_consensus_rewards = (
            0.0  # Total consensus rewards over the simulation
        )
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
            and (self.model.steps * self.model.consensus_settings.time_granularity_ms)
            >= self.migration_end_time_ms
        ):
            self.complete_migration()

        # Reset ephemeral state for new slot activities
        self.role = "none"  # Role will be reassigned by the Model
        self.has_proposed_block = False
        self.proposed_time_ms = -1
        self.estimated_profit = 0.0
        self.estimated_profit_increase = 0.0
        self.mev_captured = 0.0
        self.mev_captured_potential = 0.0
        self.attestation_rate = 0.0  # Reset for new slot
        self.has_attested = False
        self.attested_to_proposer_block = False


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


    def set_validator_preference(self, preference):
        """Sets the validator's preference for relay types."""
        if isinstance(preference, ValidatorPreference):
            self.preference = preference
        else:
            raise ValueError(
                "Preference must be an instance of ValidatorPreference Enum"
            )


    # --- Role Assignment Methods (Called by the Model) ---
    def set_proposer_role(self):
        """
        Sets this validator as the Proposer for the current slot.
        """
        self.role = "proposer"
 

    def set_attester_role(self):
        """
        Sets this validator as an Attester for the current slot and calculates its specific latency.
        """
        self.role = "attester"

    # --- Migration Methods ---
    def how_to_migrate(self):
        pass

    def decide_to_migrate(self):
        pass

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

    # dummy method for now, we complete migration immediately
    def complete_migration(self):
        pass

    def proposing(self, current_slot_time_ms_inner):
        pass

    def make_attestation(self, current_slot_time_ms_inner):
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
            # Proposer needs to decide when to propose and then propose
            self.proposing(current_slot_time_ms_inner)
        elif self.role == "attester":
            # Attesters need to know the proposer's block proposed time and their latency to the relay
            self.make_attestation(current_slot_time_ms_inner)


class ValidatorWithoutMEVBoost(RawValidatorAgent):
    def __init__(self, model):
        super().__init__(model)


    def calculate_minimal_needed_time_params(self, gcp_region):
        """Calculates the latency threshold for the Proposer based on its timing strategy."""
        if self.timing_strategy["type"] == "optimal_latency":
            to_attester_latency = [
                self.model.space.get_latency(gcp_region, a.gcp_region)
                for a in self.model.current_attesters
            ]
            # Sort latencies for threshold calculation
            to_attester_latency.sort()

            required_attesters_for_supermajority = math.ceil(
                (self.model.consensus_settings.attestation_threshold)
                * len(self.model.current_attesters)
            )

            # In fast mode, return a simplified estimate
            if self.model.fast_mode:
                return (
                    to_attester_latency,
                    required_attesters_for_supermajority,
                )
            else:
                return (
                    tuple(to_attester_latency),
                    tuple([0.5] * len(self.model.current_attesters)),
                    required_attesters_for_supermajority,
                    0.99,
                    0.0,
                    self.model.consensus_settings.attestation_time_ms,
                    5.0
                )
    
    def calculate_minimal_needed_time(self, gcp_region):
        params = self.calculate_minimal_needed_time_params(gcp_region)
        if self.model.fast_mode:
            to_attester_latency, required_attesters_for_supermajority = params
            return to_attester_latency[required_attesters_for_supermajority]
        else:
            return find_min_threshold_fast(
                *params
            )


    def simulation_with_info_sources(self):
        simulation_results = []
        time_simulations = []
        region_data = [self.calculate_minimal_needed_time_params(gcp_region) for gcp_region in self.model.gcp_regions["Region"].values]
        if self.model.fast_mode:
            for gcp_region, params in zip(self.model.gcp_regions["Region"].values, region_data):
                to_attester_latency, required_attesters_for_supermajority = params
                time_simulations.append((gcp_region, to_attester_latency[required_attesters_for_supermajority],))
        else:
            thread_number = min(10, cpu_count(), len(self.model.gcp_regions))
            with ThreadPoolExecutor(max_workers=thread_number) as ex:
                time_simulations = list(ex.map(lambda p: find_min_threshold_fast(*p), region_data))
            time_simulations = list(zip(self.model.gcp_regions["Region"].values, time_simulations))

        for gcp_region, required_time in time_simulations:
            base_threshold = self.model.consensus_settings.attestation_time_ms - required_time

            mev_offer = 0.0
            for info_agent in self.model.info_agents:
                to_info_latency = self.model.space.get_latency(gcp_region, info_agent.gcp_region)
                latency_threshold = base_threshold - to_info_latency
                mev_offer += info_agent.get_mev_offer_at_time(latency_threshold)

            simulation_results.append(
                {
                    "gcp_region": gcp_region,
                    "mev_offer": mev_offer,
                    "latency_threshold": base_threshold,
                }
            )

        for result in simulation_results:
            gcp_region = result["gcp_region"]
            if self.gcp_region == gcp_region:
                self.estimated_profit = result["mev_offer"]

        return simulation_results


    def how_to_migrate(self):
        simulation_results = self.simulation_with_info_sources()

        simulation_results.sort(key=lambda x: (
            -x["mev_offer"],
            0 if x["gcp_region"] == self.gcp_region else 1,
            x["latency_threshold"]
        ))

        # import IPython; IPython.embed(colors="neutral")

        return simulation_results[0]
    

    def decide_to_migrate(self):
        """
        Validator decides whether to migrate based on its assigned migration strategy.
        This is called by the Model after a slot where this validator was Proposer.
        Only the validator on the cloud can migrate.
        """
        if (
            self.is_migrating
            or self.migration_cooldown > 0
        ):
            return False, "migrating_or_on_cooldown"

        if self.type == ValidatorType.HOME:
            return False, "home_staker"
        
        elif self.location_strategy["type"] == "best_relay":
            simulation_result = self.how_to_migrate()

            print(f"current: {self.gcp_region}, target: {simulation_result['gcp_region']}, estimated profit: {self.estimated_profit}, target profit: {simulation_result['mev_offer']}, latency threshold: {simulation_result['latency_threshold']}")

            # No migration needed if already at the best relay
            if simulation_result["gcp_region"] == self.gcp_region:
                return False, "utility_not_improved"
            else:
                self.estimated_profit_increase = simulation_result["mev_offer"] - self.estimated_profit
                target_gcp_region = simulation_result["gcp_region"]

                if self.migration_cost >= (simulation_result["mev_offer"] - self.estimated_profit):
                    return False, "migration_cost_high (utility_not_improved)"

                row = self.model.gcp_regions[self.model.gcp_regions["Region"] == target_gcp_region].iloc[
                    0
                ]

                target_pos = self.model.space.get_coordinate_from_lat_lon(
                    row["Nearest City Latitude"], row["Nearest City Longitude"]
                )

                self.do_migration(target_pos, target_gcp_region)
                return True, "utility_improved"

        return False, "no_applicable_strategy"
    

    # --- In-Slot Behavior Methods (Called from step()) ---
    def decide_and_propose(self, current_slot_time_ms_inner):
        """
        Proposer (this validator) decides whether to propose a block based on its strategy.
        Returns (should_propose, mev_offer_if_proposing)
        """
        if self.has_proposed_block:  # Already proposed or migrating, cannot act
            return False, 0.0, 0

        if (
            self.timing_strategy["type"] == "optimal_latency"
        ):  # The proposer knows its latency is optimized
            required_time = self.model.consensus_settings.attestation_time_ms - self.calculate_minimal_needed_time(self.gcp_region)

            if (
                current_slot_time_ms_inner <= required_time
                and current_slot_time_ms_inner
                + self.model.consensus_settings.time_granularity_ms
                > required_time
            ):
                mev_offer = 0.0
                for info_agent in self.model.info_agents:
                    to_info_latency = self.model.space.get_latency(
                        self.gcp_region, info_agent.gcp_region
                    )
                    latency_threshold = (
                        required_time
                        - to_info_latency
                    )
                    info_source_value = info_agent.get_mev_offer_at_time(latency_threshold)
                    mev_offer += info_source_value
               
                return True, mev_offer, required_time

        return False, 0.0, 0
    

    def propose_block(self,proposed_time, mev_offer):
        """Executes the block proposal action for the Proposer."""
        # print(f"Validator {self.unique_id} proposing at {proposed_time} ms with MEV offer {mev_offer:.4f} ETH")
        self.has_proposed_block = True
        self.proposed_time_ms = proposed_time
        self.mev_captured_potential = (
            round(mev_offer, 6)  # Store potential MEV before supermajority check
        )


    def decide_and_attest(
        self,
        current_slot_time_ms_inner,
        block_proposed_time_ms,
        to_attester_latency,
    ):
        """
        Attester (this validator) decides whether to attest to the Proposer's block.
        """
        if self.has_attested:  # Already attested or migrating, cannot act
            return

        # TODO: We should also account for how other attesters behave
        # i.e., an attester should not attest if it knows that the block is not getting enough attestations
        if (
            current_slot_time_ms_inner
            > self.model.consensus_settings.attestation_time_ms
        ):
            # According to the current MEV-Boost auctions, the relay broadcasts the block
            # TODO: the proposer also broadcasts its block, which might be closer to some validators
            block_arrival_at_this_attester_ms = (
                block_proposed_time_ms
                + to_attester_latency
            )

            if (
                block_proposed_time_ms != -1
                and block_arrival_at_this_attester_ms
                <= self.model.consensus_settings.attestation_time_ms
            ):
                self.attested_to_proposer_block = True
            else:
                # if
                # print(f"Validator {self.unique_id} could not attest in time. Block arrival: {block_arrival_at_this_attester_ms} ({block_proposed_time_ms}, {to_attester_latency}), attestation deadline: {self.model.consensus_settings.attestation_time_ms}")
                self.attested_to_proposer_block = False
            self.has_attested = True


    def proposing(self, current_slot_time_ms_inner):
        should_propose, mev_offer, proposed_time = self.decide_and_propose(current_slot_time_ms_inner)

        if should_propose:
             self.propose_block(proposed_time, mev_offer)


    def make_attestation(self, current_slot_time_ms_inner):
        # Attesters need to know the proposer's block proposed time and their latency to the relay
        proposer_agent = self.model.get_current_proposer_agent()
        proposer_gcp_region = proposer_agent.gcp_region if proposer_agent else None
        to_attester_latency = self.model.space.get_latency(
            proposer_gcp_region, self.gcp_region
        ) if proposer_gcp_region else BASE_NETWORK_LATENCY_MS

        if proposer_agent:
            self.decide_and_attest(
                current_slot_time_ms_inner,
                proposer_agent.proposed_time_ms,
                self.model.latency_generator.get_latency(to_attester_latency, 0.5)
            )


class ValidatorWithMEVBoost(RawValidatorAgent):
    def __init__(self, model):
        super().__init__(model)
        self.target_relay = None  # The relay this proposer targets
        self.relay_id = None
        self.network_latency_to_target = {}  # Latency to each relay
        self.latency_threshold = -1 

    # utility functions for MEV-Boost related operations    
    def set_latency_to_relays(self):
        self.network_latency_to_target = {}
        for relay_agent in self.model.relay_agents:
            self.network_latency_to_target[relay_agent.unique_id] = (
                self.model.space.get_latency(
                    self.gcp_region, relay_agent.gcp_region
                )
            )

    def set_proposer_role(self):
        super().set_proposer_role()
        self.set_latency_to_relays()


    def set_attester_role(self):
        super().set_attester_role()
        self.set_latency_to_relays()

    
    def get_mev_offer_from_relays(self, current_time, relay_agents):
        """
        Proposer queries all relays for their current MEV offers.
        Returns a list of MEV offers from all relays.
        """
        mev_offers = []
        for r in relay_agents:
            if (
                self.preference == ValidatorPreference.COMPLIANT
                and r.type != RelayType.CENSORING
            ):
                continue
            mev_offers.append((r.get_mev_offer_at_time(current_time), r.unique_id))
        return mev_offers
    

    def get_best_mev_offer_from_relays(self, current_time, relay_agents):
        """
        Proposer queries all relays for their current MEV offers.
        Returns the best MEV offer from all relays.
        """
        mev_offers = self.get_mev_offer_from_relays(current_time, relay_agents)
        if len(mev_offers) == 0:
            return 0.0, None
        return max(mev_offers, key=lambda x: x[0])

    
    def calculate_minimal_needed_time(self, target_relay=None, relay_latency=None):
        """Calculates the latency threshold for the Proposer based on its timing strategy."""
        if self.timing_strategy["type"] == "optimal_latency":
            if self.target_relay is None:
                self.target_relay = list(self.model.relay_agents)[
                    0
                ]  # Default to first relay if none set

            if target_relay is None:
                target_relay = self.target_relay

            # Calculate the latency threshold for optimal latency strategy
            to_relay_latency = self.network_latency_to_target[target_relay.unique_id]
            if relay_latency is not None:
                to_relay_latency = relay_latency

            relay_to_attester_latency = [
                a.network_latency_to_target[target_relay.unique_id]
                for a in self.model.current_attesters
            ]
            # Sort latencies for threshold calculation
            relay_to_attester_latency.sort()

            required_attesters_for_supermajority = math.ceil(
                (self.model.consensus_settings.attestation_threshold)
                * len(self.model.current_attesters)
            )

            # In fast mode, return a simplified estimate
            if self.model.fast_mode:
                return (
                    relay_to_attester_latency[required_attesters_for_supermajority]
                    + to_relay_latency * 3
                )
            else:
                if to_relay_latency == 0:
                    return self.model.latency_generator.find_min_threshold(
                        tuple(relay_to_attester_latency),
                        tuple([0.5] * len(self.model.current_attesters)),
                        required_attesters_for_supermajority,
                        target_prob=0.95,
                        threshold_low=0.0,
                        threshold_high=self.model.consensus_settings.attestation_time_ms,
                        tolerance=5.0,
                    )
                else:
                    return self.model.latency_generator.find_min_threshold_with_monte_carlo(
                        [to_relay_latency] * 3,
                        [0.5] * 3,
                        relay_to_attester_latency,
                        [0.5] * len(self.model.current_attesters),
                        required_attesters_for_supermajority,
                        target_prob=0.95,
                        samples=10000,
                        threshold_low=0.0,
                        threshold_high=self.model.consensus_settings.attestation_time_ms,
                        tolerance=5.0,
                    )

        
    def set_latency_threshold(self, target_relay=None, to_relay_latency=0):
        """
        Sets the latency threshold for the Proposer based on its timing strategy.
        """
        minimal_needed_time = self.calculate_minimal_needed_time(
            target_relay, to_relay_latency
        )

        self.latency_threshold = (
            self.model.consensus_settings.attestation_time_ms - minimal_needed_time
        )

    
    def simulation_with_relay(self, gcp_region=None):
        simulation_results = []
        for relay_agent in self.model.relay_agents:
            # Skip if the relay is not censoring but the validator is compliant
            if (
                relay_agent.type != RelayType.CENSORING
                and self.preference == ValidatorPreference.COMPLIANT
            ):
                continue

            if gcp_region is None:
                to_relay_latency = 0
            else:
                to_relay_latency = self.model.space.get_latency(
                    gcp_region, relay_agent.gcp_region
                )

            minimal_needed_time = self.calculate_minimal_needed_time(
                relay_agent, to_relay_latency
            )
            latency_threshold = (
                self.model.consensus_settings.attestation_time_ms - minimal_needed_time
            )
            mev_offer = relay_agent.get_mev_offer_at_time(latency_threshold)
            simulation_results.append(
                {
                    "gcp_region": gcp_region if gcp_region else relay_agent.gcp_region,
                    "relay": relay_agent,
                    "latency_threshold": latency_threshold,
                    "mev_offer": round(mev_offer, 6),
                }
            )
        return simulation_results
    

    # --- Migration Methods ---
    def how_to_migrate(self):
        # if the validator co-locates with a relay
        simulation_results = self.simulation_with_relay()

        # Sort by MEV offer, then by latency threshold
        simulation_results.sort(key=lambda x: (
            -x["mev_offer"],
            0 if x["gcp_region"] == self.gcp_region else 1,
            x["latency_threshold"])
        )

        return simulation_results[0]

    def decide_to_migrate(self):
        """
        Validator decides whether to migrate based on its assigned migration strategy.
        This is called by the Model after a slot where this validator was Proposer.
        Only the validator on the cloud can migrate.
        """
        if (
            self.is_migrating
            or self.migration_cooldown > 0
        ):
            return False, "migrating_or_on_cooldown"

        if self.type == ValidatorType.HOME:
            return False, "home_staker"

        if self.location_strategy["type"] == "never_migrate":
            return False, "never_migrate_strategy"
       
        elif self.location_strategy["type"] == "best_relay":
            simulation_result = self.how_to_migrate()
            if simulation_result["gcp_region"] == self.gcp_region:
                return False, "utility_not_improved"
            else:
                self.estimated_profit_increase = simulation_result["mev_offer"] - self.estimated_profit
                # Check if migration cost is acceptable
                # simulation_results[0]["mev_offer"] is the best MEV offer after migration
                # self.estimated_profit is the estimated profit before migration
                if self.migration_cost >= (simulation_result["mev_offer"] - self.estimated_profit):
                    # print(f"Validator {self.unique_id} (at {self.gcp_region}) Migration cost too high, not migrating.")
                    return False, "migration_cost_high (utility_not_improved)"
                
                target_gcp_region = simulation_result["gcp_region"]
                self.target_relay = simulation_result["relay"]

                if self.target_relay.gcp_region != target_gcp_region:
                    row = self.model.gcp_regions[["Region"] == target_gcp_region].iloc[
                        0
                    ]
                    target_pos = self.model.space.get_coordinate_from_lat_lon(
                        row["lat"], row["lon"]
                    )
                else:
                    target_pos = self.target_relay.position

                print(
                    f"Validator {self.unique_id} (at {self.gcp_region}) considering migration to Relay {self.target_relay.unique_id}  with MEV offer {simulation_result['mev_offer']:.4f} ETH and latency threshold {simulation_result['latency_threshold']} ms"
                )
                if self.gcp_region != target_gcp_region:
                    # if self.model.space.distance(self.position, target_pos) > 0:
                    print(f"  Deciding to migrate ({self.target_relay.unique_id}).")
                    self.do_migration(target_pos, target_gcp_region)
                    return True, "utility_improved"

        return False, "no_applicable_strategy"
    

    def do_migration(self, new_position_coords, new_gcp_region):
        super().do_migration(new_position_coords, new_gcp_region)
        self.set_latency_to_relays()

           
    def estimate_profit(self):
        simulation_results = self.simulation_with_relay(self.gcp_region)
        simulation_results.sort(key=lambda x: (-x["mev_offer"], x["latency_threshold"]))
        best_mev = simulation_results[0]["mev_offer"]
        self.estimated_profit = best_mev
        self.target_relay = simulation_results[0]["relay"]

    
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
                    self.timing_strategy["delay_ms"], relay_agents
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
                    proposed_time_ms, relay_agents
                )

                return True, mev_offer, relay_id, proposed_time_ms
        elif (
            self.timing_strategy["type"] == "optimal_latency"
        ):  # The proposer knows its latency is optimized
            if self.latency_threshold == -1:
                # Calculate the latency threshold for optimal latency strategy
                to_relay_latency = self.model.space.get_latency(
                    self.gcp_region, self.target_relay.gcp_region
                )
                self.set_latency_threshold(self.target_relay, to_relay_latency)
            if (
                current_slot_time_ms_inner <= self.latency_threshold
                and current_slot_time_ms_inner
                + self.model.consensus_settings.time_granularity_ms
                > self.latency_threshold
            ):
                # TODO: Depending on relay-proposer latency, proposer will get mev not at current step but at a previous one
                mev_offer = self.target_relay.get_mev_offer_at_time(
                    self.latency_threshold
                )
                relay_id = self.target_relay.unique_id
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
            round(mev_offer, 6)  # Store potential MEV before supermajority check
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
        if (
            current_slot_time_ms_inner
            > self.model.consensus_settings.attestation_time_ms
        ):
            # According to the current MEV-Boost auctions, the relay broadcasts the block
            # TODO: the proposer also broadcasts its block, which might be closer to some validators
            block_arrival_at_this_attester_ms = (
                block_proposed_time_ms
                + proposer_to_relay_latency
                + relay_to_attester_latency
            )

            if (
                block_proposed_time_ms != -1
                and block_arrival_at_this_attester_ms
                <= self.model.consensus_settings.attestation_time_ms
            ):
                self.attested_to_proposer_block = True
            else:
                self.attested_to_proposer_block = False
            self.has_attested = True


    def proposing(self, current_slot_time_ms_inner):
        should_propose, mev_offer, relay_id, proposed_time = (
            self.decide_and_propose(
                current_slot_time_ms_inner, self.model.relay_agents
            )
        )
        if should_propose:
            self.propose_block(proposed_time, mev_offer, relay_id)


    def make_attestation(self, current_slot_time_ms_inner):
        proposer_agent = self.model.get_current_proposer_agent()
        proposer_to_relay_latency = (
            proposer_agent.network_latency_to_target[proposer_agent.relay_id]
            if proposer_agent and proposer_agent.relay_id
            else 0
        )

        # relay id is the relay the proposer used
        avg_latency_to_relay = self.network_latency_to_target.get(
            proposer_agent.relay_id, BASE_NETWORK_LATENCY_MS
        )

        if proposer_agent:
            self.decide_and_attest(
                current_slot_time_ms_inner,
                proposer_agent.proposed_time_ms,
                proposer_to_relay_latency,
                self.model.latency_generator.get_latency(avg_latency_to_relay, 0.5),
            )
