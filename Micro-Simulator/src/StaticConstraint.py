"""
Filename: StaticConstraint.py
Description:
    Makes the necessary computations for the generation of
    static obstacle avoidance lin constraints
"""

import numpy as np
from app import ObstacleMap
import timeit
import time


class StaticObstacle:
    def __init__(self, map, grid_size=20):

        # List of (x, y) obstacle positions
        # Each point is a grid cell
        self.grid_size = grid_size
        self.obstacle_map = ObstacleMap(map)

        self.radius = 15

        # Time variables for printing
        self.solve_time = ""
        self.elapsed = 0

    def compute_collision_free_area(self, pred_traj):
        """
        Wrapper for static constraint calc.
        Computes a collision-free regions for each step in the trajectory.
        """
        free_regions = []
        corner_list = []
        start_time = timeit.default_timer()
        for i, (x_path, y_path, theta) in enumerate(pred_traj):
            # Find the largest free space by checking obstacles
            free_space, corners = self.compute_constraints(x_path, y_path, theta)
            free_regions.append(free_space)
            corner_list.append(corners)

        # measure elapsed time
        self.elapsed = timeit.default_timer() - start_time
        self.solve_time = "To find free area took {:.4f} seconds".format(self.elapsed)
        print(self.solve_time)

        return free_regions, corner_list

    def get_rotated_occupancy(self, x_i, search_x, y_i, search_y, theta):
        """
        Rotates the search direction according to the robot's heading angle
        and checks whether the transformed position is occupied.
        """
        # Apply 2D rotation
        x_search_rotated = round(
            np.cos(theta) * search_x - np.sin(theta) * search_y)
        y_search_rotated = round(
            np.sin(theta) * search_x + np.cos(theta) * search_y)

        # Compute absolute grid positions
        x_new = x_i + x_search_rotated
        y_new = y_i - y_search_rotated

        # Check if the rotated position is out of bounds
        if (x_new < 0 or y_new < 0
                or x_new >= 800 or y_new >= 600):           # Assuming 800x600 grid size, change to variable
            return True                              	    # Treat as occupied

        # Check if the new position is an obstacle
        if (round(x_new), round(y_new)) in self.obstacle_map.obstacles:
            return True
        return False                                        # else its free space

    def compute_constraints(self, x_path, y_path, theta):
        """
        Compute linear constraints from convex free region
        """
        theta = self.normalize_theta(-theta)
        x_min, x_max, y_min, y_max = self.compute_bounds(x_path, y_path, theta)

        # Apply safety margin
        x_min += self.radius
        x_max -= self.radius
        y_min += self.radius
        y_max -= self.radius

        # Rotate the bounding box to the robots orientation
        sqx = [
            x_path + np.cos(theta) * x_min - np.sin(theta)
            * y_min,  # Bottom-left corner
            x_path + np.cos(theta) * x_min - np.sin(theta)
            * y_max,  # Top-left corner
            x_path + np.cos(theta) * x_max - np.sin(theta)
            * y_max,  # Top-right corner
            x_path + np.cos(theta) * x_max - np.sin(theta)
            * y_min   # Bottom-right corner
        ]

        sqy = [
            y_path - (np.sin(theta) * x_min + np.cos(theta)
                      * y_min),  # Bottom-left corner
            y_path - (np.sin(theta) * x_min + np.cos(theta)
                      * y_max),  # Top-left corner
            y_path - (np.sin(theta) * x_max + np.cos(theta)
                      * y_max),  # Top-right corner
            y_path - (np.sin(theta) * x_max + np.cos(theta)
                      * y_min)   # Bottom-right corner
        ]

        # Compute direction vectors (edges of the rectangle)
        # Left Edge (Bottom-left -> Top-left)
        t1 = np.array([sqy[1] - sqy[0], sqx[1] - sqx[0]])
        # Top Edge (Top-left -> Top-right)
        t2 = np.array([sqy[2] - sqy[1], sqx[2] - sqx[1]])
        # Right Edge (Top-right -> Bottom-right)
        t3 = np.array([sqy[3] - sqy[2], sqx[3] - sqx[2]])
        # Bottom Edge (Bottom-right -> Bottom-left)
        t4 = np.array([sqy[0] - sqy[3], sqx[0] - sqx[3]])

        # Normalize direction vectors
        t1 = t1 / np.linalg.norm(t1)
        t2 = t2 / np.linalg.norm(t2)
        t3 = t3 / np.linalg.norm(t3)
        t4 = t4 / np.linalg.norm(t4)

        # Rotate by 90 degrees to get outward normal vectors
        a1y, a1x =  t1[1], t1[0]
        a2y, a2x = -t2[1], t2[0]
        a3y, a3x =  t3[1], t3[0]
        a4y, a4x = -t4[1], t4[0]

        C1 = a1x * sqx[0] + a1y * sqy[0]  # Left boundary
        C2 = a2x * sqx[1] + a2y * sqy[1]  # Top boundary (Flipped)
        C3 = a3x * sqx[2] + a3y * sqy[2]  # Right boundary
        C4 = a4x * sqx[3] + a4y * sqy[3]  # Bottom boundary (Flipped)

        # Computed constraint values
        # in practice Ax + By <= C --> line segment equation
        # where n = (A, B)

        # Return constraints as a list of (A, B, C)
        constraints = [
            (a1x, a1y, C1),  # Left
            (a2x, a2y, C2),  # Top
            (a3x, a3y, C3),  # Right edge
            (a4x, a4y, C4)   # Bottom edge
        ]
        return constraints, list(zip(sqx, sqy))

    def compute_bounds(self, x_path, y_path, theta):
        max_search_distance = 40

        # Initialize search bounds with max search range
        x_min, x_max = -max_search_distance, max_search_distance
        y_min, y_max = -max_search_distance, max_search_distance

        search_distance = 15
        step_size = self.grid_size

        left = right = top = bottom = True

        while search_distance < max_search_distance:
            # Check top boundary
            if top:
                search_y = search_distance
                x_range = range(max(-search_distance, x_min), min(search_distance, x_max), step_size)
                if any(self.get_rotated_occupancy(x_path, x, y_path, search_y, theta) for x in x_range):
                    y_max = search_y
                    top = False

            # Check bottom boundary
            if bottom:
                search_y = -search_distance
                x_range = range(max(-search_distance, x_min), min(search_distance, x_max), step_size)
                if any(self.get_rotated_occupancy(x_path, x, y_path, search_y, theta) for x in x_range):
                    y_min = search_y
                    bottom = False

            # Check left boundary
            if left:
                search_x = -search_distance
                y_range = range(max(-search_distance, y_min), min(search_distance, y_max), step_size)
                if any(self.get_rotated_occupancy(x_path, search_x, y_path, y, theta) for y in y_range):
                    x_min = search_x
                    left = False

            # Check right boundary
            if right:
                search_x = search_distance
                y_range = range(max(-search_distance, y_min), min(search_distance, y_max), step_size)
                if any(self.get_rotated_occupancy(x_path, search_x, y_path, y, theta) for y in y_range):
                    x_max = search_x
                    right = False

            if left == False and right == False and top == False and bottom == False:
                break;

            search_distance += 1

        return x_min, x_max, y_min, y_max

    def shift_previous_trajectory(self, p_prev, N):
        """
        Shift the optimal trajectory computed at time t − 1, namely,
        q_{0:N} = [p^∗_{1:N} |t−1 , q_N ], where q_N is an extrapolation of
        the last two points, that is, q_N = 2p^∗_N |t−1 − p^∗_{N−1}|t−1
        """
        q_new = []
        # Shift each step forward
        for k in range(N):
            q_new.append(p_prev[k + 1])

        # Extrapolate the last point
        last_point = (2 * np.array(p_prev[-1]) - np.array(p_prev[-2]))
        q_new.append(tuple(last_point))

        return q_new

    def normalize_theta(self, theta):
        """
        Normalize robot's orientation angle (wrap between -pi and pi)
        """
        theta_norm = theta - np.pi / 2
        return np.arctan2(np.sin(theta_norm), np.cos(theta_norm))
