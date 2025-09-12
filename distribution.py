import math
import numpy as np
import pandas as pd
import random
import re

from abc import ABC, abstractmethod
from scipy.stats import norm, lognorm, poisson_binom
from functools import lru_cache


_CONTINENT_RULES = [
    (r"^us-|^northamerica-", "North America"),
    (r"^southamerica-",      "South America"),
    (r"^europe-",            "Europe"),
    (r"^asia-",              "Asia"),
    (r"^australia-",         "Oceania"),
    (r"^me-",                "Middle East"),
    (r"^africa-",            "Africa"),
]


def to_continent(region: str) -> str:
    for pat, name in _CONTINENT_RULES:
        if re.match(pat, region):
            return name
    return "Other"

def convert_to_marco_regions(
    regions,
    latency
):
    reg_ids = set(regions["Region"])
    latency = latency[
        latency["sending_region"].isin(reg_ids) &
        latency["receiving_region"].isin(reg_ids)
    ].copy()

    regions["Continent"] = regions["Region"].map(to_continent)
    cont_map = dict(zip(regions["Region"], regions["Continent"]))
    latency["from_c"] = latency["sending_region"].map(cont_map)
    latency["to_c"]   = latency["receiving_region"].map(cont_map)

    # Undirected mean per continent pair
    latency["pair"] = latency.apply(
        lambda r: tuple(sorted([r["from_c"], r["to_c"]])), axis=1
    )
    c_lat = latency.groupby("pair")["milliseconds"].median().reset_index()

    # Build symmetric matrix
    mat = {}

    for _, row in c_lat.iterrows():
        a, b = row["pair"]; ms = row["milliseconds"]
        mat[(a, b)] = ms
        mat[(b, a)] = ms

    regions = (
        regions.groupby("Continent")[["Nearest City Longitude", "Nearest City Latitude"]]
        .mean()
        .rename(columns={"Nearest City Longitude": "lon", "Nearest City Latitude": "lat"})
    )

    regions["Nearest City Longitude"] = regions["lon"]
    regions["Nearest City Latitude"] = regions["lat"]
    regions["gcp_region"] = regions.index
    regions["Region"] = regions.index
    regions["Region Name"] = regions.index

    return regions, mat


def parse_gcp_latency(latency_df):
    latency_dict = {}
    for _, row in latency_df.iterrows():
        key1 = (row["sending_region"], row["receiving_region"])
        latency_dict[key1] = row["milliseconds"]
        key2 = (row["receiving_region"], row["sending_region"])
        latency_dict[key2] = row["milliseconds"]

    return latency_dict


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
            r2 = x * x + y * y + z * z
            if r2 > 1e-12:  # Avoid division by zero for very small magnitudes
                scale = 1.0 / math.sqrt(r2)
                return (x * scale, y * scale, z * scale)

    def distance(self, p1, p2):
        """
        Calculates the geodesic distance between two points on a unit sphere.
        Distance = arc length = arccos(dot(p1,p2)).
        """
        dotp = p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]
        # Numerical safety clamp for dot product to be within [-1, 1] due to floating point inaccuracies
        dotp = max(-1.0, min(1.0, dotp))
        return math.acos(dotp)

    def get_area(self):
        """Returns the surface area of a unit sphere."""
        return 4 * np.pi

    def get_max_dist(self):
        """Returns the maximum possible geodesic distance on a unit sphere (half circumference)."""
        return (
            np.pi
        )  # Half the circumference of a unit circle (pi * diameter = pi * 2 * radius = 2*pi * 1 / 2 = pi)

    def get_coordinate_from_lat_lon(self, lat, lon):
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
    
    def set_gcp_latency_regions(self, gcp_latency, gcp_regions):
        """
        Sets the GCP latency
        """
        self.gcp_latency = gcp_latency
        self.gcp_regions = gcp_regions

    def get_nearest_gcp_region(self, position, gcp_regions):
        """
        Finds the nearest GCP region to a given position on the unit sphere.
        Returns the GCP region that is closest in terms of geodesic distance.
        """
        min_distance = float("inf")
        nearest_zone = None
        for index, row in gcp_regions.iterrows():
            zone_position = self.get_coordinate_from_lat_lon(
                row["Nearest City Latitude"], row["Nearest City Longitude"]
            )
            distance = self.distance(position, zone_position)
            if distance < min_distance:
                min_distance = distance
                nearest_zone = row["Region Name"]
        return nearest_zone if nearest_zone else None


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
            dist_matrix[j][i] = d  # Symmetric matrix
    return dist_matrix


def update_distance_matrix_for_node(dist_matrix, positions, space, moved_idx):
    """
    After node 'moved_idx' has changed its position,
    recalc only row [moved_idx] and column [moved_idx].
    """
    n = len(positions)
    i = moved_idx
    for j in range(n):
        if j == i:
            dist_matrix[i][j] = 0.0
        else:
            d = space.distance(positions[i], positions[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d


# --- Latency Distribution ---
# This function generates a normal distribution of latencies based on a given mean latency.
def generate_normal_latency_distribution(mean_latency, std_dev_ratio=0.1, num_samples=10000):
    """
    Generates a normal distribution of latencies from a given mean latency.

    Parameters:
    mean_latency (float): The desired mean of the latency distribution.
    std_dev_ratio (float): The ratio of standard deviation to the mean.
                           (e.g., 0.1 means std_dev = 10% of mean_latency)
    num_samples (int): The number of latency samples to generate.

    Returns:
    numpy.ndarray: An array of simulated latency values.
    """
    if mean_latency <= 0:
        raise ValueError("Mean latency must be positive.")
    if std_dev_ratio <= 0:
        raise ValueError("Standard deviation ratio must be positive.")

    # Calculate standard deviation based on the ratio
    std_dev = mean_latency * std_dev_ratio

    # Generate samples from a normal distribution
    latencies = np.random.normal(loc=mean_latency, scale=std_dev, size=num_samples)

    # Latency cannot be negative, so cap any negative values at 0.
    # This is a common practical adjustment for normal distributions modeling non-negative quantities.
    latencies[latencies < 0] = 0
    
    return latencies


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


    def evaluate_threshold_with_monte_carlo(
        self,
        shared_means,
        shared_stds,
        broadcast_means,
        broadcast_stds,
        threshold,
        required_attesters,
        samples=10000
    ):
        """
        Estimate the probability that at least `required_attesters` receive the message
        within the given latency threshold.

        Parameters:
        - shared_means: list of means for the first 3 shared segments (A→B→A→B)
        - shared_stds: list of stddevs for the first 3 shared segments
        - broadcast_means: list of means for B→attester_i broadcast (per attester)
        - broadcast_stds: list of stddevs for B→attester_i (per attester)
        - threshold: latency threshold to compare against (float)
        - required_attesters: how many attesters must receive below the threshold
        - samples: number of Monte Carlo samples to use

        Returns:
        - probability of satisfying the threshold condition
        """

        # Step 1: Sample the total shared latency (A -> B -> A -> B)
        shared_latency = np.zeros(samples)
        for mean, std in zip(shared_means, shared_stds):
            if std <= 0:
                shared_latency += mean
            else:
                self.inititalize_distribution(mean, std)
                key = (mean, std)
                if key not in self.dist_cache:
                    continue
                dist = self.dist_cache[key]
                shared_latency += dist.rvs(size=samples)

        # Step 2: For each attester, add their broadcast delay and compute the success prob
        success_probs = []
        for mean, std in zip(broadcast_means, broadcast_stds):
            if std <= 0:
                total_latency = shared_latency + mean
            else:
                self.inititalize_distribution(mean, std)
                key = (mean, std)
                if key in self.dist_cache:
                    dist = self.dist_cache[key]
                    total_latency = shared_latency + dist.rvs(size=samples)
                else:
                    total_latency = shared_latency # mean is 0

            prob = np.mean(total_latency < threshold)
            success_probs.append(prob)

        # Step 3: Use Poisson Binomial to compute probability of at least `required_attesters` successes
        pb = poisson_binom(success_probs)
        return 1 - pb.cdf(required_attesters - 1)
    
    # @lru_cache(maxsize=1024)
    def find_min_threshold_with_monte_carlo(
        self,
        shared_means,
        shared_stds,
        broadcast_means,
        broadcast_stds,
        required_attesters,
        target_prob=0.95,
        samples=10000,
        threshold_low=0.0,
        threshold_high=4000.0,
        tolerance=5.0
    ):
        """
        Binary search for the minimum latency threshold such that
        the success probability is >= target_prob.
        """
        while threshold_high - threshold_low > tolerance:
            mid = (threshold_low + threshold_high) / 2
            prob = self.evaluate_threshold_with_monte_carlo(
                shared_means,
                shared_stds,
                broadcast_means,
                broadcast_stds,
                threshold=mid,
                required_attesters=required_attesters,
                samples=samples
            )

            if prob >= target_prob:
                threshold_high = mid  # try to reduce threshold
            else:
                threshold_low = mid  # need more time

        return (threshold_high + threshold_low) / 2  # or threshold_high / threshold_low, depending on preference
    
    def evaluate_threshold(
        self,
        broadcast_latencies,
        broadcast_stds,
        threshold,
        required_attesters
    ):
        """
        Evaluates the probability that at least one attester receives the broadcast
        within the given latency threshold.

        Parameters:
        - broadcast_latencies: list of latencies for each attester's broadcast
        - broadcast_stds: list of standard deviations for each attester's broadcast
        - threshold: latency threshold to compare against (float)

        Returns:
        - probability of at least one attester receiving within the threshold
        """
        if not broadcast_latencies or not broadcast_stds:
            return 0.0

        probabilities = []
        for latency, std in zip(broadcast_latencies, broadcast_stds):
            if std <= 0:
                prob = 1.0 if latency < threshold else 0.0
                probabilities.append(prob)
            else:
                # Handle zero or negative latency cases
                if latency <= 0:
                    probabilities.append(1.0)
                    continue

                self.inititalize_distribution(latency, std)
                key = (latency, std)
                if key not in self.dist_cache:
                    continue
                dist = self.dist_cache[key]
                probabilities.append(
                    dist.cdf(threshold)  # Probability that this attester receives within threshold
                )
        
        pb = poisson_binom(probabilities)
        return pb.sf(required_attesters - 1)

    @lru_cache(maxsize=1024)
    def find_min_threshold(
        self,
        broadcast_latencies,
        broadcast_stds,
        required_attesters,
        target_prob=0.99,
        threshold_low=0.0,
        threshold_high=4000.0,
        tolerance=5.0
    ):
        while threshold_high - threshold_low > tolerance:
            mid = (threshold_low + threshold_high) / 2
            prob = self.evaluate_threshold(
                broadcast_latencies,
                broadcast_stds,
                threshold=mid,
                required_attesters=required_attesters
            )
            if prob >= target_prob:
                threshold_high = mid
            else:
                threshold_low = mid
            
            if threshold_high - threshold_low < tolerance:
                break
        
        return (threshold_high + threshold_low) / 2  # or threshold_high / threshold_low, depending on preference
    
    def get_search_space(self, T):
        """
        Returns the search space for the latency distribution.
        This is a placeholder method that can be overridden in subclasses.
        """
        return None
    
    def compute_the_delay_from_distribution(self):
        pass  # Placeholder for potential methods.


    

@lru_cache(maxsize=1024)
def inititalize_distribution(mean_latency, std_dev_ratio=0.1):
    if mean_latency <= 0:
        return None

    std_dev = mean_latency * std_dev_ratio
    mu = np.log(mean_latency**2 / np.sqrt(mean_latency**2 + std_dev**2))
    sigma = np.sqrt(np.log(1 + (std_dev**2 / mean_latency**2)))
    
    return lognorm(s=sigma, scale=np.exp(mu))


@lru_cache(maxsize=1024)
def evaluate_threshold(
        broadcast_latencies,
        broadcast_stds,
        threshold,
        required_attesters
    ):
        if not broadcast_latencies or not broadcast_stds:
            return 0.0

        probabilities = []
        for latency, std in zip(broadcast_latencies, broadcast_stds):
            if std <= 0:
                prob = 1.0 if latency < threshold else 0.0
                probabilities.append(prob)
            else:
                if latency <= 0:
                    probabilities.append(1.0)
                    continue
                dist = inititalize_distribution(latency, std)
                probabilities.append(
                    dist.cdf(threshold)
                )
        
        pb = poisson_binom(probabilities)
        return pb.sf(required_attesters - 1)


@lru_cache(maxsize=1024)
def evaluate_threshold_fast(
        broadcast_latencies, # MUST be a tuple
        broadcast_stds,      # MUST be a tuple
        threshold,
        required_attesters
    ):
        if not broadcast_latencies:
            return 0.0

        latencies = np.array(broadcast_latencies, dtype=np.float64)
        stds = np.array(broadcast_stds, dtype=np.float64)
        
        probabilities = np.zeros_like(latencies)

        # Masks for different conditions
        zero_latency_mask = (latencies <= 0)
        zero_std_mask = (stds <= 0) & ~zero_latency_mask
        valid_mask = ~zero_latency_mask & ~zero_std_mask

        # Condition 1: latency <= 0 -> prob = 1.0
        probabilities[zero_latency_mask] = 1.0

        # Condition 2: std <= 0 (and latency > 0) -> prob is 1.0 if latency < threshold, else 0.0
        probabilities[zero_std_mask] = np.where(latencies[zero_std_mask] < threshold, 1.0, 0.0)

        # Condition 3: Regular calculation for valid entries
        if np.any(valid_mask):
            valid_latencies = latencies[valid_mask]
            # Assuming broadcast_stds represents the std_dev_ratio
            std_dev = valid_latencies * stds[valid_mask]

            mean_sq = valid_latencies**2
            std_dev_sq = std_dev**2
            
            mu = np.log(mean_sq / np.sqrt(mean_sq + std_dev_sq))
            sigma = np.sqrt(np.log(1 + (std_dev_sq / mean_sq)))
            
            probabilities[valid_mask] = lognorm.cdf(threshold, s=sigma, scale=np.exp(mu))
        
        # Use a PoissonBinomial library to calculate the survival function
        pb = poisson_binom(probabilities.tolist())
        # pb.sf(k) is P(X > k). We want P(X >= k), which is P(X > k-1).
        return pb.sf(required_attesters - 1)



@lru_cache(maxsize=1024)
def find_min_threshold(
        broadcast_latencies,
        broadcast_stds,
        required_attesters,
        target_prob=0.99,
        threshold_low=0.0,
        threshold_high=4000.0,
        tolerance=5.0
    ):
        while threshold_high - threshold_low > tolerance:
            mid = (threshold_low + threshold_high) / 2
            prob = evaluate_threshold(
                broadcast_latencies,
                broadcast_stds,
                threshold=mid,
                required_attesters=required_attesters
            )
            if prob >= target_prob:
                threshold_high = mid
            else:
                threshold_low = mid
            
            if threshold_high - threshold_low < tolerance:
                break
        
        return (threshold_high + threshold_low) / 2


@lru_cache(maxsize=1024)
def find_min_threshold_fast(
        broadcast_latencies, # MUST be a tuple
        broadcast_stds,      # MUST be a tuple
        required_attesters,
        target_prob=0.99,
        threshold_low=0.0,
        threshold_high=4000.0,
        tolerance=5.0
    ):
        # The binary search logic is already efficient. The main speedup comes
        # from calling the fast version of evaluate_threshold.
        while threshold_high - threshold_low > tolerance:
            mid = (threshold_low + threshold_high) / 2
            if mid <= 0: # Avoid getting stuck at 0
                threshold_low = tolerance
                continue
            
            prob = evaluate_threshold_fast( # Call the fast version!
                broadcast_latencies,
                broadcast_stds,
                threshold=mid,
                required_attesters=required_attesters
            )

            if prob >= target_prob:
                threshold_high = mid
            else:
                threshold_low = mid
        
        return (threshold_high + threshold_low) / 2