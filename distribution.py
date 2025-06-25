import math
import numpy as np
import random

from abc import ABC, abstractmethod


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

    def get_nearest_gcp_zone(self, position, gcp_zones):
        """
        Finds the nearest GCP zone to a given position on the unit sphere.
        Returns the GCP zone that is closest in terms of geodesic distance.
        """
        min_distance = float("inf")
        nearest_zone = None
        for index, row in gcp_zones.iterrows():
            zone_position = (row["x"], row["y"], row["z"])
            distance = self.distance(position, zone_position)
            if distance < min_distance:
                min_distance = distance
                nearest_zone = row["region"]
        return nearest_zone if nearest_zone else None

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


def init_distance_matrix(positions, space):  # , gcp_latency, gcp_zones):
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
