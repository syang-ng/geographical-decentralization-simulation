import math

from enum import Enum
from mesa import Agent
from multiprocessing import cpu_count, Pool
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

from constants import (
    BASE_NETWORK_LATENCY_MS,
)
from distribution import find_min_threshold_fast, find_min_threshold_approx, ThresholdEvaluatorApprox
from source_agent import RelayType


_G_REGION_DATA = None

def _init_worker(region_data):
    global _G_REGION_DATA
    _G_REGION_DATA = region_data

def _run_batch(idxs):
    out = []
    for i in idxs:
        out.append(_run_find_min_threshold(_G_REGION_DATA[i]))
    return out

def _chunk_indices(n, batch_size):
    for start in range(0, n, batch_size):
        yield list(range(start, min(n, start + batch_size)))


def _run_find_min_threshold(args):
    # return find_min_threshold_fast(*args)
    lat_tup, std_tup, required_attesters, target_prob, threshold_low, threshold_high, tolerance = args
    return find_min_threshold_approx(
        ThresholdEvaluatorApprox(lat_tup, std_tup),
        required_attesters,
        target_prob=target_prob,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        tolerance=tolerance
    )

# --- Validator Agent Class Definition ---

class ValidatorType(Enum):
    HOME = 1
    CLOUD = 2


class ValidatorPreference(Enum):
    COMPLIANT = 1
    NONCOMPLIANT = 2


class RawValidatorAgent(Agent):
    """
    Represents a single validator, which can be a Proposer or an Attester.
    It has a region, network latency, and strategies for proposing and potentially migrating.
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


    def do_migration(self, new_gcp_region):
        """Completes the migration process."""
        self.is_migrating = True
        self.migration_cooldown = self.model.migration_cooldown_slots
        self.gcp_region = new_gcp_region
        self.is_migrating = False  # Migration is completed immediately in this model


    def calculate_minimal_needed_time_params(self, gcp_region):
        """Calculates the latency threshold for the Proposer based on its timing strategy."""
        if self.timing_strategy["type"] == "optimal_latency":
            to_attester_latency = [
                self.model.gcp_latency_model.get_latency(gcp_region, a.gcp_region)
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
                    1.0
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

def _run_by_idx(i: int):
    return _run_find_min_threshold(_G_REGION_DATA[i])

class MSPValidator(RawValidatorAgent):
    def __init__(self, model):
        super().__init__(model)


    def simulation_with_signals(self):
        simulation_results = []
        time_simulations = []
        t1 = time.perf_counter()
        region_data = [self.calculate_minimal_needed_time_params(gcp_region) for gcp_region in self.model.gcp_latency_model.gcp_regions["Region"].values]
        print(f"    prepare region data time: {(time.perf_counter() - t1):.3f} s")
        t2 = time.perf_counter()
        if self.model.fast_mode:
            for gcp_region, params in zip(self.model.gcp_latency_model.gcp_regions["Region"].values, region_data):
                to_attester_latency, required_attesters_for_supermajority = params
                time_simulations.append((gcp_region, to_attester_latency[required_attesters_for_supermajority],))
        else:
            # Use parallel processing for non-fast mode
            process_number = min(cpu_count(), len(self.model.gcp_latency_model.gcp_regions))
            with ProcessPoolExecutor(
                max_workers=process_number,
                initializer=_init_worker,
                initargs=(region_data,),
            ) as ex:
                results = list(ex.map(_run_by_idx, range(len(region_data))))
            time_simulations = list(zip(self.model.gcp_latency_model.gcp_regions["Region"].values, results))
        for gcp_region, required_time in time_simulations:
            base_threshold = self.model.consensus_settings.attestation_time_ms - required_time

            mev_offer = 0.0
            for signal_agent in self.model.signal_agents:
                to_signal_latency = self.model.gcp_latency_model.get_latency(gcp_region, signal_agent.gcp_region)
                latency_threshold = base_threshold - to_signal_latency
                mev_offer += signal_agent.get_mev_offer_at_time(latency_threshold)

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
        t1 = time.perf_counter()
        simulation_results = self.simulation_with_signals()
        t2 = time.perf_counter()
        print(f"  simulation time: {(t2 - t1):.3f} s")

        simulation_results.sort(key=lambda x: (
            -x["mev_offer"],
            0 if x["gcp_region"] == self.gcp_region else 1,
            x["latency_threshold"]
        ))


        for i in simulation_results:
            i["slot"] = self.model.current_slot_idx
        
        self.model.region_profits += simulation_results

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

                self.do_migration(target_gcp_region)
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
                for signal_agent in self.model.signal_agents:
                    to_signal_latency = self.model.gcp_latency_model.get_latency(
                        self.gcp_region, signal_agent.gcp_region
                    )
                    latency_threshold = (
                        required_time
                        - to_signal_latency
                    )
                    signal_source_value = signal_agent.get_mev_offer_at_time(latency_threshold)
                    mev_offer += signal_source_value
               
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
        to_attester_latency = self.model.gcp_latency_model.get_latency(
            proposer_gcp_region, self.gcp_region
        ) if proposer_gcp_region else BASE_NETWORK_LATENCY_MS

        if proposer_agent:
            self.decide_and_attest(
                current_slot_time_ms_inner,
                proposer_agent.proposed_time_ms,
                self.model.latency_generator.get_latency(to_attester_latency, 0.5)
            )


class SSPValidator(RawValidatorAgent):
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
                self.model.gcp_latency_model.get_latency(
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
  
        
    def set_latency_threshold(self, target_relay=None, relay_latency=None):
        """
        Sets the latency threshold for the Proposer based on its timing strategy.
        """
        minimal_needed_time = self.calculate_minimal_needed_time(
            target_relay.gcp_region
        )

        self.latency_threshold = (
            self.model.consensus_settings.attestation_time_ms - minimal_needed_time - relay_latency
        )

    
    def simulation_with_relays(self):
        simulation_results = []
        time_simulations = []
        
        relay_regions = set([r.gcp_region for r in self.model.relay_agents])
        other_regions = set(self.model.gcp_latency_model.gcp_regions["Region"].values) - relay_regions

        target_relays = [relay_agent for relay_agent in self.model.relay_agents if (self.preference == ValidatorPreference.NONCOMPLIANT or relay_agent.type == RelayType.CENSORING)]
        target_gcp_regions = [r.gcp_region for r in target_relays]
        region_data = [self.calculate_minimal_needed_time_params(gcp_region) for gcp_region in target_gcp_regions]
        if self.model.fast_mode:
            for target_relay, params in zip(target_relays, region_data):
                to_attester_latency, required_attesters_for_supermajority = params
                time_simulations.append((target_relay, to_attester_latency[required_attesters_for_supermajority],))
        else:
            # Use parallel processing for non-fast mode
            process_number = min(cpu_count(), len(region_data))
            if process_number > 1 and len(region_data) > 1:
                with ProcessPoolExecutor(max_workers=process_number) as ex:
                    results = list(ex.map(_run_find_min_threshold, region_data))
                time_simulations = list(zip(target_relays, results))
            else:
                # Fall back to sequential processing if only one worker or one item
                for target_relay, params in zip(target_relays, region_data):
                    simulation = _run_find_min_threshold(params)
                    time_simulations.append((target_relay, simulation,)) 

        for target_relay, minimal_needed_time in time_simulations:
            latency_threshold = (
                self.model.consensus_settings.attestation_time_ms - minimal_needed_time
            )
            mev_offer = target_relay.get_mev_offer_at_time(latency_threshold)
            simulation_results.append(
                {
                    "gcp_region": target_relay.gcp_region,
                    "relay": target_relay,
                    "latency_threshold": latency_threshold,
                    "mev_offer": round(mev_offer, 6),
                }
            )

            for region in other_regions:
                region_to_relay_latency = self.model.gcp_latency_model.get_latency(region, target_relay.gcp_region)
                new_latency_threshold = latency_threshold - region_to_relay_latency
                mev_offer = target_relay.get_mev_offer_at_time(new_latency_threshold)
                simulation_results.append(
                    {
                        "gcp_region": region,
                        "relay": target_relay,
                        "latency_threshold": new_latency_threshold,
                        "mev_offer": round(mev_offer, 6),
                    }
                )

        sub_results = [result for result in simulation_results if result["gcp_region"] == self.gcp_region]
        sub_results.sort(key=lambda x: (-x["mev_offer"], x["latency_threshold"]))
        self.estimated_profit = sub_results[0]["mev_offer"]
        self.target_relay = sub_results[0]["relay"]

        return simulation_results
    

    # --- Migration Methods ---
    def how_to_migrate(self):
        # if the validator co-locates with a relay
        t1 = time.perf_counter()
        simulation_results = self.simulation_with_relays()
        t2 = time.perf_counter()
        print(f"  simulation time: {(t2 - t1):.3f} s")

        # Sort by MEV offer, then by latency threshold
        simulation_results.sort(key=lambda x: (
            -x["mev_offer"],
            0 if x["gcp_region"] == self.gcp_region else 1,
            x["latency_threshold"])
        )

        for i in simulation_results:
            i["slot"] = self.model.current_slot_idx
        self.model.region_profits += simulation_results

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
            print(f"current: {self.gcp_region}, target: {simulation_result['gcp_region']}, estimated profit: {self.estimated_profit}, target profit: {simulation_result['mev_offer']}, latency threshold: {simulation_result['latency_threshold']}")

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

                print(
                    f"Validator {self.unique_id} (at {self.gcp_region}) considering migration to Relay {self.target_relay.unique_id}  with MEV offer {simulation_result['mev_offer']:.4f} ETH and latency threshold {simulation_result['latency_threshold']} ms"
                )
                if self.gcp_region != target_gcp_region:
                    print(f"  Deciding to migrate ({self.target_relay.unique_id}).")
                    self.do_migration(target_gcp_region)
                    return True, "utility_improved"

        return False, "no_applicable_strategy"
    

    def do_migration(self, new_gcp_region):
        super().do_migration(new_gcp_region)
        self.set_latency_to_relays()


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
                to_relay_latency = self.model.gcp_latency_model.get_latency(
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
