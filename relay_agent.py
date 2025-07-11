from enum import Enum
from mesa import Agent

from constants import (
    BASE_MEV_AMOUNT,
    MEV_INCREASE_PER_SECOND
)

# --- Relay Types ---

class RelayType(Enum):
    CENSORING = 0
    NONCENSORING = 1

# --- Relay Agent Profiles --- 

RELAY_PROFILES = [
    # Flashbots Relay, aws us-east-1, "Northern Virginia, USA" -- this is close to GCP us-east4, "Ashburn, Virginia, USA"
    {
        "unique_id": "Flashbots",
        "gcp_region": "us-east4",
        "lat": 39.0437,
        "lon": -77.4874,
        "utility_function": lambda x: BASE_MEV_AMOUNT + x * MEV_INCREASE_PER_SECOND,
        "type": RelayType.CENSORING,
    },
    # UltraSound Relay, ovh roubaix, "Roubaix, France" -- this is close to GCP europe-west1, "St. Ghislain, Belgium"
    {
        "unique_id": "UltraSound EU",
        "gcp_region": "europe-west1",
        "lat": 50.4577,
        "lon": 3.8643,
        "utility_function": lambda x: BASE_MEV_AMOUNT + x * MEV_INCREASE_PER_SECOND,
        "type": RelayType.NONCENSORING,
    },
    # UltraSound Relay, ovh vint hill, "Vint Hill, Virginia, USA" -- this is close to GCP us-east4, "Ashburn, Virginia, USA"
    {
        "unique_id": "UltraSound US",
        "gcp_region": "us-east4",
        "lat": 39.0437,
        "lon": -77.4874,
        "utility_function": lambda x: BASE_MEV_AMOUNT + x * MEV_INCREASE_PER_SECOND,
        "type": RelayType.NONCENSORING,
    },
]


# --- Relay Agent Class Definition ---

class RelayAgent(Agent):
    """
    A simple Relay Agent that has a position and provides the current best MEV offer.
    It doesn't have complex strategies; it's a conduit.
    """

    def __init__(self, model):
        super().__init__(model)
        self.current_mev_offer = 0.0
        self.type = RelayType.NONCENSORING

    def initialize_with_profile(self, profile):
        """
        Initializes the Relay Agent with a specific profile.
        The profile should contain 'unique_id', 'gcp_region', 'lat', and 'lon'.
        """
        self.unique_id = profile["unique_id"]
        self.gcp_region = profile["gcp_region"]
        self.position = self.model.space.get_coordinate_from_lat_lon(
            profile["lat"], profile["lon"]
        )
        self.role = "relay_agent"
        self.utility_function = profile.get(
            "utility_function",
            lambda x: BASE_MEV_AMOUNT + x * MEV_INCREASE_PER_SECOND
        )
        self.type = profile.get("type", RelayType.NONCENSORING)

    def set_position(self, position):
        """Sets the Relay's position in the space."""
        self.position = position

    def set_gcp_region(self, gcp_region):
        """Sets the Relay's GCP region for latency calculations."""
        self.gcp_region = gcp_region

    def set_utility_function(self, utility_function):
        """Sets the Relay's utility function for MEV offers."""
        self.utility_function = utility_function

    def update_mev_offer(self):
        """Simulates builders providing better offers to the Relay over time."""
        # Get current time from the model's steps
        # Convert model time steps to milliseconds within the current slot
        current_slot_time_ms = (
            self.model.steps * self.model.consensus_settings.time_granularity_ms
        ) % self.model.consensus_settings.slot_duration_ms
        time_in_seconds = current_slot_time_ms / 1000

        # MEV offer is calculated based on the utility function
        self.current_mev_offer = (
            self.utility_function(time_in_seconds)
        )

    def get_mev_offer(self):
        """Provides the current best MEV offer to a Proposer."""
        return self.current_mev_offer

    def get_mev_offer_at_time(self, time_ms):
        """
        Returns the MEV offer at a specific time in milliseconds.
        This is useful for Proposers to query the Relay for MEV offers.
        """
        time_in_seconds = time_ms / 1000
        return self.utility_function(time_in_seconds)

    def step(self):
        """
        The Relay Agent's behavior in each simulation step.
        Here, it just updates its MEV offer based on the current slot time.
        """
        self.update_mev_offer()
