"""
Filename: app.py
Description:
    Visualization tool for the simulation.
    Assume 50 pixels = 1 meter (used for distance and velocity scaling).
"""

import numpy as np
import platform
import pygame

from Control import ControlObject
from robot import Robot

class Visualizer:
    def __init__(self, window_size=(800, 600), title="MPC Visualization", grid_size=20):
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(title)
        self.grid_color = (50, 50, 50)
        self.grid_size = grid_size

        # Info panel
        font_size = 18 if platform.system() == "Windows" else 12
        self.font = pygame.font.SysFont("jetbrainsmononerdfont", font_size) or pygame.font.Font(None, font_size)

        self.panel_color = (200, 200, 200)
        self.text_color = (0, 0, 0)
        self.panel_width = 250
        self.panel_height = 130
        self.panel_x = window_size[0] - self.panel_width - 10
        self.panel_y = 10

        self.robot = Robot()

    def draw_grid(self):
        """
        Draws background grid
        """
        for x in range(0, self.screen.get_width(), self.grid_size):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.screen.get_height()))
        for y in range(0, self.screen.get_height(), self.grid_size):
            pygame.draw.line(self.screen, self.grid_color, (0, y), (self.screen.get_width(), y))


    def draw_info_panel(self, state, distance, vel_lin, vel_r, theta):
        """
        Draws an info panel displaying robot status, distance to target, velocity, and orientation.
        """
        panel_rect = pygame.Rect(self.panel_x, self.panel_y, self.panel_width, self.panel_height)
        panel_surface = pygame.Surface((self.panel_width, self.panel_height), pygame.SRCALPHA)

        # Draw rounded rectangle
        pygame.draw.rect(self.screen, (200, 200, 200, 180), panel_rect, border_radius=15)

        # Status
        state_text = self.font.render(f"Status: {state}", True, self.text_color)
        self.screen.blit(state_text, (self.panel_x + 10, self.panel_y + 10))

        # Distance to target
        distance_text = self.font.render(f"Distance to target: {np.sqrt(distance)/50:.2f} m", True, self.text_color)
        self.screen.blit(distance_text, (self.panel_x + 10, self.panel_y + 30))

        # Lin vel
        vel_lin_text = self.font.render(f"Linear Velocity: {vel_lin / 50:.3f} m/s", True, self.text_color)
        self.screen.blit(vel_lin_text, (self.panel_x + 10, self.panel_y  + 60))

        # Angular vel
        vel_r_text = self.font.render(f"Angular Velocity: {vel_r / 50:.3f} rad/s", True, self.text_color)
        self.screen.blit(vel_r_text, (self.panel_x + 10, self.panel_y + 80))

        # Theta
        theta_text = self.font.render(f"Orientation: {theta:.3f} rad", True, self.text_color)
        self.screen.blit(theta_text, (self.panel_x + 10, self.panel_y + 100))


    def draw_fading_trail(self, trail):
        """
        Draw target's trail with position history
        """
        trail_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        fade_threshold = 40

        new_trail = []

        for i in range(len(trail) - 1):
            alpha = int(255 * (i / len(trail)))
            if alpha > fade_threshold:
                color = (0, 255, 255, alpha)
                pygame.draw.line(trail_surface, color, trail[i], trail[i+1], 3)
                new_trail.append(trail[i])

        trail[:] = new_trail
        self.screen.blit(trail_surface, (0, 0))

    def draw_arrow(self, start_pos, angle, length=20, color=(255, 0, 0), width=2):
        """
        Draw mpc predictions
        """
        end_pos = (start_pos[0] + np.cos(angle) * length, start_pos[1] + np.sin(angle) * length)
        pygame.draw.line(self.screen, color, start_pos, end_pos, width)

    def update(self, position, trail, state, mpc):
        """
        Updates the visualization by drawing the robot, predicted trajectory,
                and relevant information.
        """
        self.screen.fill((0, 0, 0))  # Clear the screen
        self.draw_grid()

        # draw path history
        self.draw_fading_trail(trail)

        # Draw the predicted trajectory from the MPC solver
        if mpc.sol is not None:
            xs = mpc.sol.value(mpc.pos_x)
            ys = mpc.sol.value(mpc.pos_y)
            thetas = mpc.sol.value(mpc.pos_theta)

            for i in range(len(xs)):
                self.draw_arrow((xs[i], ys[i]), thetas[i], length=40, color=(255, 0, 255), width=1)

        # Update the robot's state and draw it
        self.robot.update_state(state)
        self.robot.draw(self.screen)

        # Draw the target position
        pygame.draw.circle(self.screen, (255, 0, 0), position, 8)

        # Draw the info panel
        self.draw_info_panel(mpc.status, mpc.current_distance, state['vt'], state['vr'], state['theta'])
        pygame.display.flip()
