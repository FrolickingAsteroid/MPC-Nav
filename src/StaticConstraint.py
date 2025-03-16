import numpy as np

class StaticObstacle:
    def __init__(self, obstacle_map, grid_size=20):

        # List of (x, y) obstacle positions
        # Each point is a grid cell
        self.obstacle_map = obstacle_map
        self.grid_size = grid_size

        # Check where to shift the previous trajectory forward
        # and extrapolate last point (warm start?)

    def get_obstacle_positions(self):
        """
        Extracts obstacle positions from the grid-based map.
        """
        obstacle_positions = []
        for x, y in self.obstacle_map:
            obstacle_positions.append((x, y))

        return np.array(obstacle_positions)


    def graham_scan_algorithm():
        """
        Compute convex free region around the robot's trajectory
        using graham scan algorithm. Use the closest obstacles
        to define a safe convex polygon (recheck)
        """
        pass

    def compute_linear_constraints():
        """
        Compute linear constraints from convex free region
        """
        pass

    def shift_previous_trajectory():
        """
        Shift the optimal trajectory computed at time t − 1, namely,
        q_{0:N} = [p^∗_{1:N} |t−1 , q_N ], where q_N is an extrapolation of 
        the last two points, that is, q_N = 2p^∗_N |t−1 − p^∗_{N−1}|t−1
        """
        pass
