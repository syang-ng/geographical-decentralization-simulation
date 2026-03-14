import math
import numpy as np

from scipy.stats import norm, lognorm, poisson_binom
from functools import lru_cache


def distance(p1, p2):
    """
    Calculates the geodesic distance between two points on a unit sphere.
    Distance = arc length = arccos(dot(p1,p2)).
    """
    dotp = p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]
    # Numerical safety clamp for dot product to be within [-1, 1] due to floating point inaccuracies
    dotp = max(-1.0, min(1.0, dotp))
    return math.acos(dotp)


def get_coordinate_from_lat_lon(lat, lon):
    """
    Converts latitude and longitude to Cartesian coordinates on the unit sphere.
    Latitude and longitude are in radians.
    """
    phi = math.radians(lat)
    theta = math.radians(lon)
    x = math.cos(phi) * math.cos(theta)
    y = math.cos(phi) * math.sin(theta)
    z = math.sin(phi)
    return (x, y, z)


class GCPLatencyModel:
    """
    A simple class to hold GCP latency data.
    """
    def __init__(self, gcp_latency, gcp_regions):
        self.gcp_latency = gcp_latency
        self.gcp_regions = gcp_regions


    @lru_cache(maxsize=1024)
    def get_latency(self, gcp1, gcp2):
        """
        Returns the avg latency between two GCP regions according GCP latency data.
        Assumes gcp_latency is a DataFrame with columns 'sending_region', 'receiving_region', and 'milliseconds'.
        """

        if gcp1 == gcp2:
            return 0.0
        
        if (gcp1, gcp2) in self.gcp_latency:
            return self.gcp_latency[(gcp1, gcp2)] / 2
        elif (gcp2, gcp1) in self.gcp_latency:
            return self.gcp_latency[(gcp2, gcp1)] / 2
        else:
            return max(self.gcp_latency.values()) / 2
        

    def get_nearest_gcp_region(self, lat, lon):
        """
        Finds the nearest GCP region to a given position on the unit sphere.
        Returns the GCP region that is closest in terms of geodesic distance.
        """
        min_distance = float("inf")
        nearest_zone = None
        position = get_coordinate_from_lat_lon(lat, lon)
        for index, row in self.gcp_regions.iterrows():
            zone_position = get_coordinate_from_lat_lon(
                row["Nearest City Latitude"], row["Nearest City Longitude"]
            )
            cur_distance = distance(position, zone_position)
            if cur_distance < min_distance:
                min_distance = cur_distance
                nearest_zone = row["Region Name"]
        return nearest_zone if nearest_zone else None


class LatencyGenerator:
    """
    A performance-optimized class for generating latency samples from a given distribution.
    """
    def __init__(self, fast=False, distribution_type="lognormal"):
        """
        Initializes the generator.
        :param distribution_type: The type of distribution to use, either 'normal' or 'lognormal'.
        """
        if distribution_type not in ["normal", "lognormal"]:
            raise ValueError("Unsupported distribution type. Use 'normal' or 'lognormal'.")
        self.distribution_type = distribution_type
        # The cache will store the calculated distribution objects, not large arrays of samples.
        self.dist_cache = {}
        self.fast = fast


    def inititalize_distribution(self, mean_latency, std_dev_ratio=0.1):
        """
        Initializes the distribution object based on the mean latency and standard deviation ratio.
        This method is called once to set up the distribution for subsequent sampling.
        
        :param mean_latency: The target mean for the latency distribution.
        :param std_dev_ratio: The standard deviation as a fraction of the mean.
        """
        if mean_latency <= 0:
            return None

        key = (mean_latency, std_dev_ratio)

        # 1. Check if the distribution object is already cached.
        if key not in self.dist_cache:
            std_dev = mean_latency * std_dev_ratio
            
            # If standard deviation is zero, there's no variance.
            if std_dev <= 0:
                self.dist_cache[key] = None  # Mark as no generation needed.
                return mean_latency

            # 2. If not cached, create and cache the appropriate distribution object.
            if self.distribution_type == "normal":
                # Create a normal distribution object from scipy.stats.
                self.dist_cache[key] = norm(loc=mean_latency, scale=std_dev)
            
            elif self.distribution_type == "lognormal":
                # Parameter conversion for lognormal is required because its native
                # parameters (mu, sigma) are for the underlying normal distribution.
                mu = np.log(mean_latency**2 / np.sqrt(mean_latency**2 + std_dev**2))
                sigma = np.sqrt(np.log(1 + (std_dev**2 / mean_latency**2)))
                
                # Create a lognormal distribution object.
                self.dist_cache[key] = lognorm(s=sigma, scale=np.exp(mu))


    # fast mode: return mean directly if enabled
    def get_latency(self, mean_latency, std_dev_ratio=0.1):
        """
        Directly generates and returns a single latency sample from a statistical distribution.
        This method caches the distribution object itself for efficiency, not the sample data.
        
        :param mean_latency: The target mean for the latency distribution.
        :param std_dev_ratio: The standard deviation as a fraction of the mean.
        :return: A single float representing a latency sample.
        """
        if self.fast:
            return mean_latency

        if mean_latency <= 0:
            return 0.0

        # 1. Check if the distribution object is already cached.
        key = (mean_latency, std_dev_ratio)
        self.inititalize_distribution(mean_latency, std_dev_ratio)
        # 2. Retrieve the cached distribution object.
        distribution = self.dist_cache[key]

        # If the distribution object is None (because std_dev was 0), return the mean.
        if distribution is None:
            return mean_latency
            
        # 3. Generate a single random variate (rvs) from the cached distribution object.
        # This is extremely fast compared to sampling from a large list.
        return distribution.rvs(size=1)[0]


    def get_search_space(self, T):
        """
        Returns the search space for the latency distribution.
        This is a placeholder method that can be overridden in subclasses.
        """
        return None

    
    def compute_the_delay_from_distribution(self):
        pass  # Placeholder for potential methods.


def parse_gcp_latency(latency_df):
    latency_dict = {}
    for _, row in latency_df.iterrows():
        key1 = (row["sending_region"], row["receiving_region"])
        latency_dict[key1] = row["milliseconds"]
        key2 = (row["receiving_region"], row["sending_region"])
        latency_dict[key2] = row["milliseconds"]

    return latency_dict


@lru_cache(maxsize=1024)
def inititalize_distribution(mean_latency, std_dev_ratio=0.1):
    if mean_latency <= 0:
        return None

    std_dev = mean_latency * std_dev_ratio
    mu = np.log(mean_latency**2 / np.sqrt(mean_latency**2 + std_dev**2))
    sigma = np.sqrt(np.log(1 + (std_dev**2 / mean_latency**2)))
    
    return lognorm(s=sigma, scale=np.exp(mu))


@lru_cache(maxsize=1024)
def evaluate_threshold_fast(
        broadcast_latencies, # MUST be a tuple
        broadcast_stds,      # MUST be a tuple
        threshold,
        required_attesters
    ):
        prepared = _prepare_threshold_inputs(broadcast_latencies, broadcast_stds)
        return _evaluate_threshold_from_prepared(
            prepared,
            threshold=threshold,
            required_attesters=required_attesters,
        )


def _prepare_threshold_inputs(broadcast_latencies, broadcast_stds):
    """Precompute arrays and lognormal parameters shared by every binary-search step."""
    if not broadcast_latencies:
        return None

    latencies = np.array(broadcast_latencies, dtype=np.float64)
    stds = np.array(broadcast_stds, dtype=np.float64)

    zero_latency_mask = latencies <= 0
    zero_std_mask = (stds <= 0) & ~zero_latency_mask
    valid_mask = ~zero_latency_mask & ~zero_std_mask

    valid_latencies = latencies[valid_mask]
    valid_scales = None
    valid_sigma = None

    if valid_latencies.size:
        std_dev = valid_latencies * stds[valid_mask]
        mean_sq = valid_latencies**2
        std_dev_sq = std_dev**2
        mu = np.log(mean_sq / np.sqrt(mean_sq + std_dev_sq))
        valid_sigma = np.sqrt(np.log(1 + (std_dev_sq / mean_sq)))
        valid_scales = np.exp(mu)

    return (
        latencies,
        zero_latency_mask,
        zero_std_mask,
        valid_mask,
        valid_sigma,
        valid_scales,
    )


def _evaluate_threshold_from_prepared(prepared, threshold, required_attesters):
    """Evaluate one threshold using precomputed inputs."""
    if prepared is None:
        return 0.0

    (
        latencies,
        zero_latency_mask,
        zero_std_mask,
        valid_mask,
        valid_sigma,
        valid_scales,
    ) = prepared

    probabilities = np.zeros_like(latencies)
    probabilities[zero_latency_mask] = 1.0
    probabilities[zero_std_mask] = np.where(
        latencies[zero_std_mask] < threshold,
        1.0,
        0.0,
    )

    if valid_sigma is not None:
        probabilities[valid_mask] = lognorm.cdf(
            threshold,
            s=valid_sigma,
            scale=valid_scales,
        )

    pb = poisson_binom(probabilities.tolist())
    return pb.sf(required_attesters - 1)


@lru_cache(maxsize=1024)
def find_min_threshold_fast(
        broadcast_latencies, # MUST be a tuple
        broadcast_stds,      # MUST be a tuple
        required_attesters,
        target_prob=0.99,
        threshold_low=0.0,
        threshold_high=4000.0,
        tolerance=1.0
    ):
        prepared = _prepare_threshold_inputs(broadcast_latencies, broadcast_stds)

        while threshold_high - threshold_low > tolerance:
            mid = (threshold_low + threshold_high) / 2
            if mid <= 0: # Avoid getting stuck at 0
                threshold_low = tolerance
                continue
            
            prob = _evaluate_threshold_from_prepared(
                prepared,
                threshold=mid,
                required_attesters=required_attesters,
            )

            if prob >= target_prob:
                threshold_high = mid
            else:
                threshold_low = mid
        
        return (threshold_high + threshold_low) / 2
