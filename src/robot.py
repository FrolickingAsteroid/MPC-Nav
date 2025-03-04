import numpy as np
import pygame

class Robot:
    def __init__(self, x=400, y=350, theta=-np.pi/2, b=15, wheel_radius=5):
        self.position = np.array([x, y], dtype=float)
        self.angle = theta  # Initial orientation angle in radians
        self.b = b  # Half the width of the robot
        self.wheel_radius = wheel_radius
        self.color = (51, 255, 51)  # Color of the robot
        self.orientation_color = (51, 255, 51)

        self.position_history = []

        self.update_polygon()

    def update_polygon(self):
        # Define corners of the robot relative to the center
        offset = np.array([
            [-self.b, -self.b],
            [self.b, -self.b],
            [self.b, self.b],
            [-self.b, self.b],
        ])

        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)

        # Rotate and translate the corners to get the actual vertices
        self.polygon = [self.position + np.dot(corner, [[cos_angle, -sin_angle], [sin_angle, cos_angle]]) for corner in offset]

        # Calculate the endpoint for the orientation line (e.g., 10 pixels long)
        orientation_end = self.position + np.array([np.cos(self.angle), -np.sin(self.angle)]) * 14
        self.orientation_line = [self.position, orientation_end]

    def draw(self, screen):
        for trail_pos in self.position_history:
            pygame.draw.circle(screen, (0, 0, 255), trail_pos, 2)

        # Draw the polygon on the screen
        pygame.draw.polygon(screen, self.color, np.array(self.polygon, dtype=int), width=4)
        pygame.draw.line(screen, self.orientation_color, self.orientation_line[0], self.orientation_line[1], width=4)

    def update_state(self, state):
        # Update state from an external source, e.g., MPC output
        self.position = np.array([state['x'], state['y']], dtype=float)
        self.angle = -state['theta']
        self.update_polygon()

        self.position_history.append(self.position.copy())
