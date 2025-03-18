"""
Filename: StaticConstraint.py
Description:
    Makes the necessary computations for the generation of
    static obstacle avoidance lin constraints
"""

import numpy as np
from app import ObstacleMap

class StaticObstacle:
    def __init__(self, grid_size=20):

        # List of (x, y) obstacle positions
        # Each point is a grid cell
        self.grid_size = grid_size
        self.obstacle_map = ObstacleMap("Maps/obstacle_grid_map.csv")


        # Check where to shift the previous trajectory forwardz
        # and extrapolate last point (warm start?)

    def get_obstacle_positions(self):
        """
        Extracts obstacle positions from the grid-based map.
        """
        obstacle_positions = []
        for x, y in self.obstacle_map:
            obstacle_positions.append((x, y))

        return np.array(obstacle_positions)


    def graham_scan_algorithm(self):
        """
        Compute convex free region around the robot's trajectory
        using graham scan algorithm. Use the closest obstacles
        to define a safe convex polygon (recheck)
        """
        pass

    def compute_linear_constraints(self):
        """
        Compute linear constraints from convex free region
        """
        pass

    def shift_previous_trajectory(self, p_prev, N):
        """
        Shift the optimal trajectory computed at time t − 1, namely,
        q_{0:N} = [p^∗_{1:N} |t−1 , q_N ], where q_N is an extrapolation of
        the last two points, that is, q_N = 2p^∗_N |t−1 − p^∗_{N−1}|t−1
        """
        q_new = []
        # Shift each step forward
        for k in range(N - 1):
            q_new.append(p_prev[k + 1])

        # Extrapolate the last point
        last_point = (2 * np.array(p_prev[-1]) - np.array(p_prev[-2]))
        q_new.append(tuple(last_point))
        
        return q_new
