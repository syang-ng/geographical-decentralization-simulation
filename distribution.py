import math
import numpy as np
import random

from abc import ABC, abstractmethod
from scipy.stats import norm, lognorm, poisson_binom
from functools import lru_cache

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
        gcp_latency = self.gcp_latency

        if gcp1 == gcp2:
            return 0.0
        latency_row = gcp_latency[
            (gcp_latency["sending_region"] == gcp1)
            & (gcp_latency["receiving_region"] == gcp2)
        ]
        if not latency_row.empty:
            return latency_row["milliseconds"].values[0] / 2 # Convert to one-way latency
        # If no direct latency data is found, return the max latency for the pair

        return gcp_latency["milliseconds"].max() / 2
    
    @lru_cache(maxsize=1024)
    def get_best_region_to_targets(self, targets):
        """
        Given a list of target GCP regions, finds the one with the lowest average latency to all targets.
        """
        gcp_latency = self.gcp_latency
        gcp_regions = self.gcp_regions

        subset = gcp_latency[
            (gcp_latency["sending_region"].isin(targets))
            & (gcp_latency["receiving_region"].isin(targets))
        ]
        if subset.empty:
            return None, (0, 0)
        
        region_to_target = {}
        for sending_region, receiving_region, latency in subset[["sending_region", "receiving_region", "milliseconds"]].values:
            latency = latency / 2 # Convert to one-way latency
            region_to_target[(sending_region, receiving_region)] = latency
            region_to_target[(receiving_region, sending_region)] = latency
        
        candidates = set(subset["sending_region"].unique()).union(
            set(subset["receiving_region"].unique())
        )

        candidates_avg_latency = [
            (
                candidate,
                sum(
                    region_to_target.get((candidate, target), float("inf"))
                    for target in targets
                )
                / len(targets),
            )
            for candidate in candidates
        ]

        candidates_avg_latency.sort(key=lambda x: x[1])
        region = candidates_avg_latency[0][0]
        row = gcp_regions[gcp_regions["Region Name"] == region]
        location = (
            row["Nearest City Latitude"].values[0],
            row["Nearest City Longitude"].values[0]
        )
        return region, location

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

        magnitude = math.sqrt(
            temp_center[0] ** 2 + temp_center[1] ** 2 + temp_center[2] ** 2
        )
        if magnitude < 1e-12:
            return self.sample_point()

        scale = 1.0 / magnitude
        return (temp_center[0] * scale, temp_center[1] * scale, temp_center[2] * scale)


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
                if key not in self.dist_cache:
                    continue
                dist = self.dist_cache[key]
                total_latency = shared_latency + dist.rvs(size=samples)
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
        print(f"Finding min threshold with Monte Carlo: target_prob={target_prob}, samples={samples}")
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
            else:
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
        braodcast_latencies,
        broadcast_stds,
        required_attesters,
        target_prob=0.95,
        threshold_low=0.0,
        threshold_high=4000.0,
        tolerance=5.0
    ):
        while threshold_high - threshold_low > tolerance:
            mid = (threshold_low + threshold_high) / 2
            prob = self.evaluate_threshold(
                braodcast_latencies,
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