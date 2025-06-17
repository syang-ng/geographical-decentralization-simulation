import json
import math
import numpy as np
import pandas as pd
import random

from abc import ABC, abstractmethod
from mesa import Agent, Model, DataCollector


# Constants
SLOT_DURATION_MS = 12000  # Duration of an Ethereum slot in milliseconds
TIME_GRANULARITY_MS = 100 # Simulation time step in milliseconds
ATTESTATION_TIME_MS = 4000 # Default time for Attesters to attest (can be varied per Attester)

# Network Latency Model Parameters: latency = BASE_NETWORK_LATENCY_MS + (distance_ratio * MAX_ADDITIONAL_NETWORK_LATENCY_MS)
BASE_NETWORK_LATENCY_MS = 50  # Minimum network latency regardless of distance
MAX_ADDITIONAL_NETWORK_LATENCY_MS = 2000 # Max additional latency for max distance on sphere

# MEV yield model (simulating Builder bids increasing over time)
## The number is set randomly now.
BASE_MEV_AMOUNT = 0.2  # Initial MEV in ETH
MEV_INCREASE_PER_SECOND = 0.08 # MEV increase per second (ETH/sec)

# --- Spatial Classes ---
class Space(ABC):
    """
    Abstract base class defining the interface for a 'space'
    where nodes can live. Subclasses must implement:
      - sample_point()
      - distance(p1, p2)
    """

    @abstractmethod
    def sample_point(self):
        """Samples a random point within the space."""
        pass

    @abstractmethod
    def distance(self, p1, p2):
        """Calculates the distance between two points in the space."""
        pass

    @abstractmethod
    def get_area(self):
        """Returns the total 'area' or size of the space."""
        pass

    @abstractmethod
    def get_max_dist(self):
        """Returns the maximum possible distance between any two points in the space."""
        pass

class SphericalSpace(Space):
    """
    Sample points on (or near) the unit sphere.
    distance() returns geodesic distance (great-circle distance).
    """
    def sample_point(self):
        """Samples a random point on the unit sphere (x, y, z)."""
        # Sample (x, y, z) from Normal(0, 1),
        # then normalize to lie on the unit sphere.
        while True:
            x = random.gauss(0, 1)
            y = random.gauss(0, 1)
            z = random.gauss(0, 1)
            r2 = x*x + y*y + z*z
            if r2 > 1e-12: # Avoid division by zero for very small magnitudes
                scale = 1.0 / math.sqrt(r2)
                return (x*scale, y*scale, z*scale)

    def distance(self, p1, p2):
        """
        Calculates the geodesic distance between two points on a unit sphere.
        Distance = arc length = arccos(dot(p1,p2)).
        """
        dotp = p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]
        # Numerical safety clamp for dot product to be within [-1, 1] due to floating point inaccuracies
        dotp = max(-1.0, min(1.0, dotp))
        return math.acos(dotp)
    
    def get_area(self):
        """Returns the surface area of a unit sphere."""
        return 4 * np.pi

    def get_max_dist(self):
        """Returns the maximum possible geodesic distance on a unit sphere (half circumference)."""
        return np.pi # Half the circumference of a unit circle (pi * diameter = pi * 2 * radius = 2*pi * 1 / 2 = pi)
    

    def calculate_geometric_center_of_nodes(self, nodes):
        """
        Calculates the geometric center of a set of nodes in the spherical space.
        Returns a point on the unit sphere that is the average of the node positions.
        """
        if not nodes:
            return None
        
        sum_x = sum(n.position[0] for n in nodes)
        sum_y = sum(n.position[1] for n in nodes)
        sum_z = sum(n.position[2] for n in nodes)
        
        avg_x = sum_x / len(nodes)
        avg_y = sum_y / len(nodes)
        avg_z = sum_z / len(nodes)
        
        temp_center = (avg_x, avg_y, avg_z)
        
        magnitude = math.sqrt(temp_center[0]**2 + temp_center[1]**2 + temp_center[2]**2)
        if magnitude < 1e-12:
            return self.sample_point()
        
        scale = 1.0 / magnitude
        return (temp_center[0]*scale, temp_center[1]*scale, temp_center[2]*scale)

    
def init_distance_matrix(positions, space):
    """
    Build the initial distance matrix for all node pairs.
    Returns a 2D list (or NumPy array) of shape (n, n).
    """
    n = len(positions)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = space.distance(positions[i], positions[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d # Symmetric matrix
    return dist_matrix


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
        current_slot_time_ms = (self.model.steps * TIME_GRANULARITY_MS) % SLOT_DURATION_MS
        time_in_seconds = current_slot_time_ms / 1000
        
        self.current_mev_offer = BASE_MEV_AMOUNT + time_in_seconds * MEV_INCREASE_PER_SECOND

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
        self.role = "none" # "proposer" or "attester"
        self.network_latency_to_target = -1 # Latency to Relay (for Proposer) or from Relay (for Attester)

        # Migration state
        self.migration_cooldown = 0 # In slots
        self.is_migrating = False
        self.migration_end_time_ms = -1

        # Proposer specific attributes
        self.timing_strategy = None # Assigned when chosen as proposer for a slot
        self.random_propose_time = -1 # For random delay strategy
        self.attestation_time_ms = ATTESTATION_TIME_MS # Default attestation time for Attesters
        self.location_strategy = None

        # Slot-specific performance tracking (reset by model per slot, or used for decision-making)
        self.has_proposed_block = False
        self.has_attested = False # True if this validator has attested in the current slot
        self.proposed_time_ms = -1
        self.mev_captured = 0.0 # Actual MEV earned after supermajority check
        self.mev_captured_potential = 0.0 # Potential MEV before supermajority check
        self.attested_to_proposer_block = False # True if this attester made a valid attestation for Proposer's block


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
        if self.is_migrating and (self.model.steps * TIME_GRANULARITY_MS) >= self.migration_end_time_ms:
             self.complete_migration()

        # Reset ephemeral state for new slot activities
        self.role = "none" # Role will be reassigned by the Model
        self.network_latency_to_target = -1
        self.has_proposed_block = False
        self.proposed_time_ms = -1
        self.mev_captured = 0.0 
        self.mev_captured_potential = 0.0 
        self.has_attested = False
        self.attested_to_proposer_block = False 
        self.random_propose_time = -1 # Reset for next potential proposer role
    

    def set_position(self, position):
        """Sets the validator's position in the space."""
        self.position = position

    def set_strategy(self, timing_strategy, location_strategy):
        """Sets the validators' strategies."""
        self.timing_strategy = timing_strategy
        self.location_strategy = location_strategy

    # --- Role Assignment Methods (Called by the Model) ---
    def set_proposer_role(self, relay_position, space_instance):
        """Sets this validator as the Proposer for the current slot."""
        self.role = "proposer"
        distance_to_relay = space_instance.distance(self.position, relay_position)
        self.network_latency_to_target = BASE_NETWORK_LATENCY_MS + \
                                         2 * (distance_to_relay / space_instance.get_max_dist()) * MAX_ADDITIONAL_NETWORK_LATENCY_MS
        # Set random propose time if using random strategy
        if self.timing_strategy["type"] == "random_delay":
            self.random_propose_time = random.randint(self.timing_strategy["min_delay_ms"], self.timing_strategy["max_delay_ms"])

    def set_attester_role(self, relay_position, space_instance, proposer_is_optimized_latency=False):
        """
        Sets this validator as an Attester for the current slot and calculates its specific latency.
        proposer_is_optimized_latency: If True, this attester assumes the current proposer (via relay) has optimized latency.
        """
        self.role = "attester"
        
        if proposer_is_optimized_latency:
            self.network_latency_to_target = BASE_NETWORK_LATENCY_MS
        else:
            distance_from_relay = space_instance.distance(self.position, relay_position)
            self.network_latency_to_target = BASE_NETWORK_LATENCY_MS + \
                                               (distance_from_relay / space_instance.get_max_dist()) * MAX_ADDITIONAL_NETWORK_LATENCY_MS

    # --- In-Slot Behavior Methods (Called from step()) ---
    def decide_and_propose(self, current_slot_time_ms_inner, relay_agent_instance):
        """
        Proposer (this validator) decides whether to propose a block based on its strategy.
        Returns (should_propose, mev_offer_if_proposing)
        """
        if self.has_proposed_block: # Already proposed or migrating, cannot act
            return False, 0.0

        mev_offer = relay_agent_instance.get_mev_offer()

        if self.timing_strategy["type"] == "fixed_delay":
            if current_slot_time_ms_inner >= self.timing_strategy["delay_ms"]:
                return True, mev_offer
        elif self.timing_strategy["type"] == "threshold_and_max_delay":
            if mev_offer >= self.timing_strategy["mev_threshold"] or \
               current_slot_time_ms_inner >= self.timing_strategy["max_delay_ms"]:
                return True, mev_offer
        elif self.timing_strategy["type"] == "random_delay":
            if self.random_propose_time == -1: # Should have been set by model, but for safety
                self.random_propose_time = random.randint(self.timing_strategy["min_delay_ms"], self.timing_strategy["max_delay_ms"])
            if current_slot_time_ms_inner >= self.random_propose_time:
                return True, mev_offer
        
        return False, 0.0

    def propose_block(self, current_slot_time_ms_inner, mev_offer):
        """Executes the block proposal action for the Proposer."""
        self.has_proposed_block = True
        self.proposed_time_ms = current_slot_time_ms_inner
        self.mev_captured_potential = mev_offer # Store potential MEV before supermajority check

    def decide_and_attest(self, current_slot_time_ms_inner, block_proposed_time_ms, relay_to_attester_latency):
        """
        Attester (this validator) decides whether to attest to the Proposer's block.
        """
        if self.has_attested: # Already attested or migrating, cannot act
            return

        # According to the current MEV-Boost auctions, the relay broadcasts the block
        # Todo: the proposer also broadcasts its block, which might be closer to some validators
        block_arrival_at_this_attester_ms = block_proposed_time_ms + relay_to_attester_latency

        if current_slot_time_ms_inner >= ATTESTATION_TIME_MS:
            if block_proposed_time_ms != -1 and \
               block_arrival_at_this_attester_ms <= ATTESTATION_TIME_MS:
                self.attested_to_proposer_block = True
            else:
                self.attested_to_proposer_block = False
            self.has_attested = True
            # print(self.unique_id, "Attester decided to attest", block_arrival_at_this_attester_ms, self.attested_to_proposer_block)


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
            if random.random() < self.location_strategy.get("migration_chance_per_slot", 0.01):
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
                current_attesters_in_model = [a for a in self.model.agents if isinstance(a, ValidatorAgent) and a.unique_id != self.unique_id and a.role == "attester" and not a.is_migrating]
                if current_attesters_in_model:
                    target_pos = self.model.space.calculate_geometric_center_of_nodes(current_attesters_in_model)
                else:
                    return False
            else: # Unknown target_type
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
        self.is_migrating = False


    # --- Mesa's core step method ---
    def step(self):
        """
        The main step method for a Validator Agent, called by the Mesa scheduler.
        """
        # Get current time within the slot
        current_slot_time_ms_inner = (self.model.steps * TIME_GRANULARITY_MS) % SLOT_DURATION_MS

        if self.is_migrating:
            # If migrating, the validator does not perform any actions
            return

        if self.role == "proposer":
            should_propose, mev_offer = self.decide_and_propose(current_slot_time_ms_inner, self.model.relay_agent)
            if should_propose:
                self.propose_block(current_slot_time_ms_inner, mev_offer)
        elif self.role == "attester":
            # Attesters need to know the proposer's block proposed time and their latency to the relay
            proposer_agent = self.model.get_current_proposer_agent()
            if proposer_agent:
                self.decide_and_attest(
                    current_slot_time_ms_inner,
                    proposer_agent.proposed_time_ms,
                    self.network_latency_to_target
                )

# --- MEVBoostModel Class ---

class MEVBoostModel(Model):
    """
    The main simulation model for MEV-Boost, managing validators and relay.
    """

    def __init__(self, num_validators, timing_strategies_pool, location_strategies_pool, num_slots, 
                 proposer_has_optimized_latency,
                 base_mev_amount=0.1, mev_increase_per_second=0.05,
                 migration_cooldown_slots=5):
        
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
        self.distance_matrix = None # Will be initialized after validator positions are set
        
        # --- Create Agents ---
        ValidatorAgent.create_agents(model=self, n=num_validators)
        RelayAgent.create_agents(model=self, n=1)

        validator_positions_initial = []

        for agent in self.agents:
            if isinstance(agent, ValidatorAgent):
                position = self.space.sample_point()
                validator_positions_initial.append(position)
                agent.set_position(position)
                agent.set_strategy(random.choice(self.timing_strategies_pool), random.choice(self.location_strategies_pool))
            elif isinstance(agent, RelayAgent):
                agent.set_position(self.space.sample_point())
                self.relay_agent = agent
            else:
                continue
        
        self.validators = self.agents.select(agent_type=ValidatorAgent)

        # Initialize distance matrix now that all validator positions are set
        self.distance_matrix = init_distance_matrix(validator_positions_initial, self.space)

        # --- Model-Level Tracking Variables ---
        self.current_slot_idx = -1 # Will increment at start of each slot
        self.total_mev_earned = 0.0
        self.supermajority_met_slots = 0
        self.proposed_block_times = []
        self.total_successful_attestations = 0 # Raw count of individual successful attestations
        self.total_attesters_counted = 0 # Total number of attesters in slots where proposer successfully proposed

        # --- Setup DataCollector ---
        self.datacollector = self._setup_datacollector()
        
        # --- Initial Slot Setup (before first step) ---
        self._setup_new_slot()
    
    def _setup_datacollector(self):
        """Configures and returns a Mesa DataCollector."""
        return DataCollector(
            model_reporters={
                "Average_MEV_Earned": lambda m: m.total_mev_earned / (m.current_slot_idx + 1) if m.current_slot_idx >= 0 else 0,
                "Supermajority_Success_Rate": lambda m: (m.supermajority_met_slots / (m.current_slot_idx + 1)) * 100 if m.current_slot_idx >= 0 else 0
            },
            agent_reporters={
                "Position": "position", # Example agent attribute
                "Role": "role",
                "Slot": "current_slot_idx",
                "MEV_Captured_Slot": "mev_captured", # MEV actually earned in the last slot
            }
        )

    def _setup_new_slot(self):
        """
        Manages the setup for a new logical slot:
        Resets validator states, assigns roles, and updates parameters.
        """
        self.current_slot_idx += 1

        # Reset all validators for the new slot
        for validator in self.validators:
            validator.current_slot_idx = self.current_slot_idx # Pass current slot index for migration logic
            validator.reset_for_new_slot() # Handles cooldown, completes migrations, resets ephemeral state

        # Select Proposer (must not be migrating)
        available_validators = [v for v in self.validators if not v.is_migrating]
        if not available_validators:
            self.current_proposer_agent = None # No proposer this slot
            # print(f"Slot {self.current_slot_idx + 1}: No validators available to propose (all migrating).")
            return

        self.current_proposer_agent = random.choice(available_validators)
        self.current_proposer_agent.set_proposer_role(self.relay_agent.position, self.space)
        self.current_proposer_agent.decide_to_migrate() # Check if proposer should migrate

        # Set Attesters and calculate their specific latencies from the Relay
        self.current_attesters = [v for v in available_validators if v.unique_id != self.current_proposer_agent.unique_id]
        for attester in self.current_attesters:
            attester.set_attester_role(self.relay_agent.position, self.space, self.proposer_has_optimized_latency)

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
        
        if is_new_slot_start and self.steps > 0: # Avoid re-setup for time 0
            # --- End of Previous Slot Logic & Rewards ---
            if self.current_proposer_agent and self.current_proposer_agent.has_proposed_block:
                slot_successful_attestations = sum(1 for a in self.current_attesters if a.attested_to_proposer_block)
                required_attesters_for_supermajority = math.ceil((2/3) * len(self.current_attesters))

                if slot_successful_attestations >= required_attesters_for_supermajority:
                    self.current_proposer_agent.mev_captured = self.current_proposer_agent.mev_captured_potential
                    self.total_mev_earned += self.current_proposer_agent.mev_captured
                    self.supermajority_met_slots += 1
                else:
                    self.current_proposer_agent.mev_captured = 0.0 # No reward if supermajority not met
                
                self.proposed_block_times.append(self.current_proposer_agent.proposed_time_ms)
                self.total_successful_attestations += slot_successful_attestations
            
                # Collect data after all agents have acted in this step
                self.datacollector.collect(self)

            # --- Setup for New Slot ---
            self._setup_new_slot() # This calls reset_for_new_slot on agents
            
        # --- Agents perform their step actions ---
        self.agents.do("step")
        self.agents.do("advance")

        # Condition to stop simulation early if all agents are migrating or something goes wrong
        if self.steps * TIME_GRANULARITY_MS > (self.num_slots * SLOT_DURATION_MS):
            self.running = False # Stop the simulation loop


# --- Simulation Execution ---
random.seed(0x06511)  # For reproducibility
np.random.seed(0x06511)  # For reproducibility in NumPy operations
# --- Define the Pool of Proposer Strategies ---
NUM_VALIDATORS = 500 # Example: Simulate 100 validators
# --- Define the number of slots to simulate ---
SIM_NUM_SLOTS = 100 # Example: Simulate 200 slots
TOTAL_TIME_STEPS = SIM_NUM_SLOTS * (SLOT_DURATION_MS // TIME_GRANULARITY_MS) + 1 # Total fine-grained steps (200 * 120 = 24000 steps)

# --- Define Proposer Strategies ---
all_timing_strategies = [
    {"type": "fixed_delay", "delay_ms": 500}, # Strategy 1: Fixed delay at 0.5 seconds
    {"type": "fixed_delay", "delay_ms": 1500}, # Strategy 2: Faster fixed delay at 1.5 second
    {"type": "threshold_and_max_delay", "mev_threshold": 0.3, "max_delay_ms": 2500}, # Strategy 3: Threshold 0.4 ETH or max 2.5s delay
    {"type": "random_delay", "min_delay_ms": 1000, "max_delay_ms": 3000}, # Strategy 4: Random delay between 1s and 3s
    {"type": "threshold_and_max_delay", "mev_threshold": 0.4, "max_delay_ms": 3000}, # Strategy 5: More aggressive threshold 0.4 ETH or max 3s delay
]

# --- Define Migration Strategies ---
MIGRATION_STRATEGY_NEVER = {"type": "never_migrate"}
MIGRATION_STRATEGY_RANDOM_EXPLORE = {"type": "random_explore", "migration_chance_per_slot": 0.01}
MIGRATION_STRATEGY_OPTIMIZE_RELAY = {
    "type": "optimize_to_center", 
    "target_type": "relay", 
}
MIGRATION_STRATEGY_OPTIMIZE_ATTESTERS = {
    "type": "optimize_to_center", 
    "target_type": "attesters_geometric_center", 
}

all_location_strategies = [
    MIGRATION_STRATEGY_NEVER,
    MIGRATION_STRATEGY_RANDOM_EXPLORE,
    MIGRATION_STRATEGY_OPTIMIZE_RELAY,
    MIGRATION_STRATEGY_OPTIMIZE_ATTESTERS,
]

model_params_standard_nomig = {
    "num_validators": NUM_VALIDATORS,
    "timing_strategies_pool": all_timing_strategies,
    "location_strategies_pool": all_location_strategies,
    "num_slots": SIM_NUM_SLOTS,
    "proposer_has_optimized_latency": False,
    "base_mev_amount": BASE_MEV_AMOUNT, "mev_increase_per_second": MEV_INCREASE_PER_SECOND,
}

model_standard = MEVBoostModel(**model_params_standard_nomig)
for i in range(TOTAL_TIME_STEPS):
    model_standard.step()


# --- Final Analysis & Plotting ---
print("\n--- Final Results Summary ---")
print(f"Total Slots: {model_standard.current_slot_idx + 1}")
print(f"Total MEV Earned: {model_standard.total_mev_earned:.4f} ETH")
print(f"Avg MEV Earned per Slot: {model_standard.total_mev_earned / (model_standard.current_slot_idx):.4f} ETH")
model_data = model_standard.datacollector.get_model_vars_dataframe()
# agent_data = model_standard.datacollector.get_agent_vars_dataframe() # If you were collecting agent data

print("\n--- Collected Model Data ---")
print(model_data.head())

agent_data = model_standard.datacollector.get_agent_vars_dataframe()

print("\n--- Agent Data Collected ---")
print("DataFrame Head:")
print(agent_data.head()) # Print the first 5 rows to see the structure

print("\nDataFrame Tail:")
print(agent_data.tail()) # Print the last 5 rows

# The DataFrame has a MultiIndex: (Step, AgentID)
# The 'Step' corresponds to the slot number.
print("\nDataFrame Info:")
agent_data.info()

if isinstance(agent_data.index, pd.MultiIndex):
    agent_data = agent_data.reset_index()
positions_by_slot = agent_data.groupby('Slot')['Position'].apply(list).reset_index()
nested_array = positions_by_slot['Position'].tolist()
with open('data.json', 'w') as f:
    json.dump(nested_array, f)