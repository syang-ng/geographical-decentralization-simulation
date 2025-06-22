import numpy as np

from scipy.stats import norm
from sklearn.cluster import DBSCAN


# cluster a distance matrix using DBSCAN (from Quintus's code)
def cluster_matrix(dist_matrix, eps=0.2, min_samples=10):
    """
    Cluster a distance matrix using DBSCAN.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    clusters = db.fit_predict(dist_matrix)
    unique_clusters = np.unique(clusters)
    return len(unique_clusters) if -1 not in unique_clusters else len(unique_clusters) - 1


def total_distance(dist_matrix):
    """
    Calculate the total distance in a distance matrix.
    This is the sum of all distances between pairs of points.
    """
    return np.sum(dist_matrix) / 2  # Each pair is counted twice, so divide by 2


def average_nearest_neighbor_distance(dist_matrix):
    """
    Calculate the average nearest neighbor distance for each point in the distance matrix.
    """
    n = dist_matrix.shape[0]
    total_distance = 0.0
    for i in range(n):
        # Get distances to all other points
        distances = dist_matrix[i, :]
        temp_distances = np.copy(distances)
        temp_distances[i] = np.inf  # Set self-distance to infinity to ignore it
        if np.any(temp_distances < np.inf) and len(temp_distances) > 0: 
            total_distance += np.min(temp_distances)  # Find the nearest neighbor distance
    return total_distance / n if n > 0 else 0.0


# --- NNI implementation for SphericalSpace ---
def nearest_neighbor_index_spherical(dist_matrix, spherical_space_instance):
    """
    Calculate the Nearest Neighbor Index (NNI) for points on a unit sphere.
    
    dist_matrix: Pairwise distance matrix (distances in radians).
    spherical_space_instance: An instance of SphericalSpace to get area and max_dist.
    """
    n = dist_matrix.shape[0]
    if n <= 1:
        return 0.0, 0.0, 1.0 # NNI, Z-score, P-value (default for single point)

    # 1. Observed Mean Nearest Neighbor Distance (D_obs)
    d_obs = average_nearest_neighbor_distance(dist_matrix)

    # 2. Expected Mean Nearest Neighbor Distance (D_exp) for a 2D random distribution on a sphere
    area_sphere = spherical_space_instance.get_area() # Which is 4 * pi for unit sphere
    
    # Expected mean nearest neighbor distance formula for a 2D random distribution
    # This formula is for planar, but often used as a strong approximation for spherical large N
    d_exp = 0.5 / np.sqrt(n / area_sphere) 

    # NNI
    nni = d_obs / d_exp if d_exp > 0 else 0.0

    # 3. Z-score for statistical significance (Approximation for large N)
    # Standard Error of the Mean Nearest Neighbor Distance (SED)
    # This formula is for planar, but often used as an approximation for spherical large N
    sed = 0.26136 / (np.sqrt(n * n / area_sphere))
    
    z_score = (d_obs - d_exp) / sed if sed > 0 else 0.0
    
    # P-value (using standard normal distribution, two-tailed test)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    return nni, z_score, p_value


# --- Reuse helper for building weights matrix ---
def build_distance_threshold_weights_from_matrix(dist_matrix, threshold):
    """
    Build a binary spatial weight matrix based on a distance threshold.
    
    dist_matrix: Pairwise distance matrix.
    threshold: Distance threshold.
    """
    n = dist_matrix.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue # No self-loops
            if dist_matrix[i, j] <= threshold:
                W[i, j] = 1
    return W

# --- global Moran's I ---
def global_morans_i_from_matrix(dist_matrix, attribute_values=None, threshold=np.pi / 4):
    """
    Calculate Global Moran's I from a pre-calculated distance matrix.
    
    dist_matrix: Pairwise distance matrix (distances in radians).
    attribute_values: 1D array of attribute values for each point.
                      If only analyzing point pattern, can be array of ones.
    threshold: Distance threshold in radians for spatial weights.
    """
    n = dist_matrix.shape[0]
    if attribute_values is None:
        attribute_values = np.array([1.0] * n)  # Default to ones if no attributes provided
    attribute_values = np.asarray(attribute_values)

    if n <= 1 or len(attribute_values) != n:
        return 0.0, "N/A", "N/A"

    W_raw = build_distance_threshold_weights_from_matrix(dist_matrix, threshold)
    
    S0 = np.sum(W_raw) # Sum of all weights
    
    if S0 == 0:
        print("Warning: No neighbors found with the given threshold. Moran's I cannot be computed.")
        return 0.0, "N/A", "N/A"

    x_mean = np.mean(attribute_values)

    numerator = np.sum(W_raw * (attribute_values[:, np.newaxis] - x_mean) * (attribute_values[np.newaxis, :] - x_mean))
    denominator = np.sum((attribute_values - x_mean)**2)

    moran_i = (n / S0) * (numerator / denominator) if denominator != 0 else 0.0

    return moran_i, "N/A", "N/A" # Placeholder for Z/P

# --- global Geary's C ---
def global_gearys_c_from_matrix(dist_matrix, attribute_values=None, threshold=np.pi / 4):
    """
    Calculate Global Geary's C from a pre-calculated distance matrix.
    
    dist_matrix: Pairwise distance matrix (distances in radians).
    attribute_values: 1D array of attribute values for each point.
                      If only analyzing point pattern, can be array of ones.
    threshold: Distance threshold in radians for spatial weights.
    """
    n = dist_matrix.shape[0]
    if attribute_values is None:
        attribute_values = np.array([1.0] * n)
    attribute_values = np.asarray(attribute_values)

    if n <= 1 or len(attribute_values) != n:
        return 0.0, "N/A", "N/A"

    W_raw = build_distance_threshold_weights_from_matrix(dist_matrix, threshold)
    S0 = np.sum(W_raw)
    
    if S0 == 0:
        print("Warning: No neighbors found with the given threshold. Geary's C cannot be computed.")
        return 0.0, "N/A", "N/A"

    numerator = np.sum(W_raw * (attribute_values[:, np.newaxis] - attribute_values[np.newaxis, :])**2)
    denominator = np.sum((attribute_values - np.mean(attribute_values))**2)

    gearys_c = ((n - 1) * numerator) / (2 * S0 * denominator) if denominator != 0 else 0.0

    return gearys_c, "N/A", "N/A" # Placeholder for Z/P
