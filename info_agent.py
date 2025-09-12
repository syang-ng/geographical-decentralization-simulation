import random

from mesa import Agent

from constants import (
    BASE_MEV_AMOUNT,
    MEV_INCREASE_PER_SECOND,
    LinearMEVUtility
)


INFO_PROFILES = [
    {
        "unique_id": "us-info",
        "gcp_region": "us-east4",
        "lat": 39.0437,
        "lon": -77.4874,
        "utility_function": lambda x: BASE_MEV_AMOUNT * 0.3 + x * MEV_INCREASE_PER_SECOND * 0.3,
    },
    {
        "unique_id": "eu-info",
        "gcp_region": "europe-west1",
        "lat": 50.4577,
        "lon": 3.8643,
        "utility_function": lambda x: BASE_MEV_AMOUNT * 0.3 + x * MEV_INCREASE_PER_SECOND * 0.3,
    },
    {
        "unique_id": "as-info",
        "gcp_region": "asia-northeast1",
        "lat": 35.6895,
        "lon": 139.6917,
        "utility_function": lambda x: BASE_MEV_AMOUNT * 0.3 + x * MEV_INCREASE_PER_SECOND * 0.3,
    },
]


class InfoAgent(Agent):
    """
    A simple Info Agent that has a position and provides the current best MEV offer.
    It doesn't have complex strategies; it's a conduit.
    """

    def __init__(self, model):
        super().__init__(model)
        self.current_mev_offer = 0.0

    def initialize_with_profile(self, profile):
        """
        Initializes the Info Agent with a specific profile.
        The profile should contain 'unique_id', 'gcp_region', 'lat', and 'lon'.
        """
        self.unique_id = profile["unique_id"]
        self.gcp_region = profile["gcp_region"]
        self.position = self.model.space.get_coordinate_from_lat_lon(
            profile["lat"], profile["lon"]
        )
        self.role = "info_agent"
        self.utility_function = profile.get(
            "utility_function",
            LinearMEVUtility(BASE_MEV_AMOUNT, MEV_INCREASE_PER_SECOND, 1.0)
        )

    def set_position(self, position):
        """Sets the Info's position in the space."""
        self.position = position

    def set_gcp_region(self, gcp_region):
        """Sets the Info's GCP region for latency calculations."""
        self.gcp_region = gcp_region

    def set_utility_function(self, utility_function):
        """Sets the Info's utility function for MEV offers."""
        self.utility_function = utility_function

    def update_mev_offer(self):
        """Simulates builders providing better offers to the Info over time."""
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
        This is useful for Proposers to query the Info for MEV offers.
        """
        time_in_seconds = time_ms / 1000
        return self.utility_function(time_in_seconds)

    def step(self):
        """
        The Info Agent's behavior in each simulation step.
        Here, it just updates its MEV offer based on the current slot time.
        """
        self.update_mev_offer()

# ---  Utility Function Factory ---
def create_info_utility_function(config_data):
    """
    Creates and returns a Info's utility function (lambda) based on configuration.
    """
    func_type = config_data.get('type')

    if func_type == 'linear_mev':
        base_mev = config_data.get('base_mev', BASE_MEV_AMOUNT) # Get base MEV amount, default to constant
        mev_increase = config_data.get('mev_increase', MEV_INCREASE_PER_SECOND) # Get MEV increase per second, default to constant
        multiplier = config_data.get('multiplier', 1.0) # Get the multiplier, default to 1.0
        # Return a lambda function that calculates MEV utility
        return LinearMEVUtility(base_mev, mev_increase, multiplier)
    # Add more utility function types here if needed
    else:
        raise ValueError(f"Unknown or unsupported Info utility function type: {func_type}")


def initialize_infos(info_profiles_data):
    """Initializes a list of Info profiles from YAML data."""
    info_profiles = []
    for profile_data in info_profiles_data:
        unique_id = profile_data.get('unique_id')
        gcp_region = profile_data.get('gcp_region')
        lat = profile_data.get('lat')
        lon = profile_data.get('lon')
        utility_func_config = profile_data.get('utility_function')

        if not all([unique_id, gcp_region, lat, lon, utility_func_config]):
            print(f"⚠️ Warning: Info profile for '{unique_id}' is missing required fields. Skipping.")
            continue

        try:
            utility_callable = create_info_utility_function(utility_func_config)
            info_profile = {
                "unique_id": unique_id,
                "gcp_region": gcp_region,
                "lat": lat,
                "lon": lon,
                "utility_function": utility_callable,
            }
            info_profiles.append(info_profile)
        except ValueError as e:
            print(f"❌ Failed to initialize Info '{unique_id}': {e}")
        except Exception as e:
            print(f"❌ Unknown error occurred while initializing Info '{unique_id}': {e}")
    return info_profiles


def get_random_info_profile(gcp_data_df, num):
    info_profiles = []
    for i, row in gcp_data_df.iterrows():
        profile = {
            "unique_id": f"info-{i}",
            "gcp_region": row['gcp_region'],
            "lat": row['lat'],
            "lon": row['lon'],
            "utility_function": lambda x: BASE_MEV_AMOUNT * 0.2 + x * MEV_INCREASE_PER_SECOND * 0.2,
        }
        info_profiles.append(profile)

    return random.choices(info_profiles, k=num)
    

def get_evenly_distributed_info_profiles(gcp_data_df, num):
    info_profiles = []
    for i, row in gcp_data_df.iterrows():
        profile = {
            "unique_id": f"info-{i}",
            "gcp_region": row['gcp_region'],
            "lat": row['lat'],
            "lon": row['lon'],
            "utility_function": LinearMEVUtility(0.01*40/num, 0.001*40/num, 1.0),
        }
        info_profiles.append(profile)

    if num <= len(info_profiles):
        return info_profiles[:num]
    else:
        # If more profiles are needed than available, repeat the list
        repeats = num // len(info_profiles)
        remainder = num % len(info_profiles)
        return info_profiles * repeats + info_profiles[:remainder]
