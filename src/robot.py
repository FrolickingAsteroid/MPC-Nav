"""
Filename: robot.py
Description:
    Defines a Robot class to simulate and visualize a robot's
    movement in a 2D environment using Pygame.
    (https://github.com/KoKoLates/ddrive-robots/blob/main/robot/robot.py)
"""

import numpy as np
import pygame

class Robot:
    def __init__(self, x=400, y=350, theta=-np.pi/2, b=15, wheel_radius=5):
        self.position = np.array([x, y], dtype=float)
        self.wheel_radius = wheel_radius
        self.angle = theta
        self.radius = 15
        self.b = b

        self.color = (51, 255, 51)

        self.position_history = []

    def update_polygon(self):
        """
        Updates the robot's polygon
        """
        # Define corners of the robot relative to the center
        offset = np.array([
            [-self.b, -self.b],
            [self.b, -self.b],
            [self.b, self.b],
            [-self.b, self.b],
        ])

        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)

        # Rotate and translate the corners
        self.polygon = [self.position + np.dot(corner, [[cos_angle, -sin_angle], [sin_angle, cos_angle]]) for corner in offset]

        # Calculate the endpoint
        orientation_end = self.position + np.array([np.cos(self.angle), -np.sin(self.angle)]) * 14
        self.orientation_line = [self.position, orientation_end]

    def draw(self, screen):
        """
        Draws the robot's polygon and orientation line
        """
        pygame.draw.polygon(screen, self.color, np.array(self.polygon, dtype=int), width=4)
        pygame.draw.line(screen, self.color, self.orientation_line[0], self.orientation_line[1], width=4)

    def update_state(self, state):
        """
        Updates the robot's position
        """
        # Update state from an external source (mpc)
        self.position = np.array([state['x'], state['y']], dtype=float)
        self.angle = -state['theta']
        self.update_polygon()

        self.position_history.append(self.position.copy())
