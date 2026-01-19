from enum import Enum
from mesa import Agent

from constants import (
    BASE_MEV_AMOUNT,
    MEV_INCREASE_PER_SECOND,
    LinearMEVUtility
)

# --- Relay Types ---

class RelayType(Enum):
    CENSORING = 0
    NONCENSORING = 1

# --- Relay Agent Profiles --- 
# Currently, we have three relays with different locations but the same utility functions.

RELAY_PROFILES = [
    {
        "unique_id": "Flashbots",
        "gcp_region": "us-east4",
        "lat": 39.0437,
        "lon": -77.4874,
        "utility_function": lambda x: BASE_MEV_AMOUNT * 0.95 + x * MEV_INCREASE_PER_SECOND * 0.95,
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


SIGNAL_PROFILES = [
    {
        "unique_id": "us-signal",
        "gcp_region": "us-east4",
        "lat": 39.0437,
        "lon": -77.4874,
        "utility_function": lambda x: BASE_MEV_AMOUNT * 0.3 + x * MEV_INCREASE_PER_SECOND * 0.3,
    },
    {
        "unique_id": "eu-signal",
        "gcp_region": "europe-west1",
        "lat": 50.4577,
        "lon": 3.8643,
        "utility_function": lambda x: BASE_MEV_AMOUNT * 0.3 + x * MEV_INCREASE_PER_SECOND * 0.3,
    },
    {
        "unique_id": "as-signal",
        "gcp_region": "asia-northeast1",
        "lat": 35.6895,
        "lon": 139.6917,
        "utility_function": lambda x: BASE_MEV_AMOUNT * 0.3 + x * MEV_INCREASE_PER_SECOND * 0.3,
    },
]


class SourceAgent(Agent):
    """
    A simple Source Agent that has a region and provides the current best MEV offer.
    """

    def __init__(self, model):
        super().__init__(model)
        self.current_mev_offer = 0.0


    def set_gcp_region(self, gcp_region):
        """Sets the Source's GCP region for latency calculations."""
        self.gcp_region = gcp_region


    def set_utility_function(self, utility_function):
        """Sets the Source's utility function for MEV offers."""
        self.utility_function = utility_function


    def update_mev_offer(self):
        """Simulates builders providing better offers to the Signal over time."""
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
        This is useful for Proposers to query the Signal for MEV offers.
        """
        time_in_seconds = time_ms / 1000
        return self.utility_function(time_in_seconds)


    def step(self):
        """
        The Signal Agent's behavior in each simulation step.
        Here, it just updates its MEV offer based on the current slot time.
        """
        self.update_mev_offer()


# --- Singal Agent Class Definition ---
class SignalAgent(SourceAgent):
    """
    A simple Singal Agent that has a region and provides the current best MEV offer.
    """

    def __init__(self, model):
        super().__init__(model)

    def initialize_with_profile(self, profile):
        """
        Initializes the Signal Agent with a specific profile.
        The profile should contain 'unique_id', 'gcp_region', 'lat', and 'lon'.
        """
        self.unique_id = profile["unique_id"]
        self.gcp_region = profile["gcp_region"]
        self.role = "signal_agent"
        self.utility_function = profile.get(
            "utility_function",
            LinearMEVUtility(BASE_MEV_AMOUNT, MEV_INCREASE_PER_SECOND, 1.0)
        )



# --- Relay Agent Class Definition ---
class RelayAgent(SourceAgent):
    """
    A simple Relay Agent that has a region and provides the current best MEV offer.
    """

    def __init__(self, model):
        super().__init__(model)
        self.type = RelayType.NONCENSORING


    def initialize_with_profile(self, profile):
        """
        Initializes the Relay Agent with a specific profile.
        The profile should contain 'unique_id', 'gcp_region', 'lat', and 'lon'.
        """
        self.unique_id = profile["unique_id"]
        self.gcp_region = profile["gcp_region"]
        self.role = "relay_agent"
        self.utility_function = profile.get(
            "utility_function",
            lambda x: BASE_MEV_AMOUNT + x * MEV_INCREASE_PER_SECOND
        )
        self.type = profile.get("type", RelayType.NONCENSORING)
        self.subsidy = profile.get("subsidy", 0.0)  # Default subsidy to 0.0 if not provided
        self.threshold = profile.get("threshold", 0.0)  # Default threshold to  


# ---  Utility Function Factory ---
def create_utility_function(config_data):
    """
    Creates and returns a Relay's utility function (lambda) based on configuration.
    """
    func_type = config_data.get('type')

    if func_type == 'linear_mev':
        base_mev = config_data.get('base_mev', BASE_MEV_AMOUNT) # Get base MEV amount, default to constant
        mev_increase = config_data.get('mev_increase', MEV_INCREASE_PER_SECOND) # Get MEV increase per second, default to constant
        multiplier = config_data.get('multiplier', 1.0) # Get the multiplier, default to 1.0
        # Return a lambda function that calculates MEV utility
        return lambda x: (base_mev * multiplier) + (x * mev_increase * multiplier)
    # Add more utility function types here if needed
    else:
        raise ValueError(f"Unknown or unsupported Relay utility function type: {func_type}")


def initialize_signals(signal_profiles_data):
    """Initializes a list of Signal profiles from YAML data."""
    signal_profiles = []
    for profile_data in signal_profiles_data:
        unique_id = profile_data.get('unique_id')
        gcp_region = profile_data.get('gcp_region')
        lat = profile_data.get('lat')
        lon = profile_data.get('lon')
        utility_func_config = profile_data.get('utility_function')

        if not all([unique_id, gcp_region, lat, lon, utility_func_config]):
            print(f"⚠️ Warning: signal profile for '{unique_id}' is missing required fields. Skipping.")
            continue

        try:
            utility_callable = create_utility_function(utility_func_config)
            info_profile = {
                "unique_id": unique_id,
                "gcp_region": gcp_region,
                "lat": lat,
                "lon": lon,
                "utility_function": utility_callable,
            }
            signal_profiles.append(info_profile)
        except ValueError as e:
            print(f"❌ Failed to initialize Signal '{unique_id}': {e}")
        except Exception as e:
            print(f"❌ Unknown error occurred while initializing Signal '{unique_id}': {e}")
    return signal_profiles


def initialize_relays(relay_profiles_data):
    """Initializes a list of Relay profiles from YAML data."""
    relay_profiles = []
    for profile_data in relay_profiles_data:
        unique_id = profile_data.get('unique_id')
        gcp_region = profile_data.get('gcp_region')
        lat = profile_data.get('lat')
        lon = profile_data.get('lon')
        utility_func_config = profile_data.get('utility_function')
        relay_type_str = profile_data.get('type')
        subsidy = profile_data.get('subsidy', 0.0)  # Default subsidy to 0.0 if not provided
        threshold = profile_data.get('threshold', 0.0)  # Default threshold to 0.0 if not provided

        if not all([unique_id, gcp_region, lat, lon, utility_func_config, relay_type_str]):
            print(f"⚠️ Warning: Relay profile for '{unique_id}' is missing required fields. Skipping.")
            continue

        try:
            utility_callable = create_utility_function(utility_func_config)
            # Convert string type from YAML to RelayType enum member
            relay_type = RelayType[relay_type_str.upper()]

            relay_profile = {
                "unique_id": unique_id,
                "gcp_region": gcp_region,
                "lat": lat,
                "lon": lon,
                "utility_function": utility_callable,
                "type": relay_type,
                "subsidy": subsidy,
                "threshold": threshold
            }
            relay_profiles.append(relay_profile)
        except ValueError as e:
            print(f"❌ Failed to initialize Relay '{unique_id}': {e}")
        except KeyError:
            print(f"❌ Invalid Relay type '{relay_type_str}' for Relay '{unique_id}'. Check RelayType in constants.py.")
        except Exception as e:
            print(f"❌ Unknown error occurred while initializing Relay '{unique_id}': {e}")
    return relay_profiles
