"""
Filename: app.py
Description:
    Visualization tool for the simulation.
    Loads the obstacle map unto the simulation.
    Assume 50 pixels = 1 meter (used for distance and velocity scaling).
"""

import numpy as np
import platform
import pygame
import csv

from robot import Robot

class ObstacleMap:
    def __init__(self, filename, grid_size=20):
        self.obstacles = set()
        self.grid_size = grid_size
        self.load_from_csv(filename)

    def load_from_csv(self, filename):
        """
        Load obstacle positions from a CSV file
        """
        with open(filename, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                x, y = map(int, row)
                self.obstacles.add((x, y))

    def draw(self, screen):
        """
        Draw obstacles on the screen as squares
        """
        for obstacle in self.obstacles:
            pygame.draw.rect(screen, (255, 255, 255), (*obstacle, self.grid_size, self.grid_size))

class Visualizer:
    def __init__(self, window_size=(1100, 600), grid_size=20, info_panel_width=300):
        pygame.init()

        # Expand the window to fit the info panel on the right
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("MPC Visualization")

        self.grid_color = (50, 50, 50)
        self.grid_size = grid_size
        self.info_panel_width = info_panel_width  # Width for info panel

        # Font setup (windows renders font size differently)
        font_size = 18 if platform.system() == "Windows" else 12
        self.font = pygame.font.SysFont("jetbrainsmononerdfont", font_size) or pygame.font.Font(None, font_size)

        self.panel_color = (200, 200, 200)
        self.text_color = (0, 0, 0)

        self.robot = Robot()
        self.obstacle_map = ObstacleMap("Maps/obstacle_grid_map.csv")

    def draw_grid(self):
        """
        Draws background grid
        """
        for x in range(0, self.screen.get_width() - self.info_panel_width, self.grid_size):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.screen.get_height()))
        for y in range(0, self.screen.get_height(), self.grid_size):
            pygame.draw.line(self.screen, self.grid_color, (0, y), (self.screen.get_width() - self.info_panel_width, y))

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

    def draw_info_panel(self, state, distance, vel_lin, vel_r, theta, mpc_solve_time, square_solve_time, mpc_elapsed, square_elapsed):
        """
        Draws an info panel on the right side of the window
        """
        panel_rect = pygame.Rect(self.screen.get_width() - self.info_panel_width, 0, self.info_panel_width, self.screen.get_height())
        pygame.draw.rect(self.screen, self.panel_color, panel_rect)

        text_lines = [
            f"Status: {state}",
            f"Distance to Target: {np.sqrt(distance) / 50:.2f} m",
            f"Linear Velocity: {abs(vel_lin) / 50:.3f} m/s",
            f"Angular Velocity: {abs(vel_r) / 50:.3f} rad/s",
            f"Orientation: {((theta + np.pi) % (2 * np.pi) - np.pi):.3f} rad"
        ]

        y_offset = 20
        for line in text_lines:
            text_surface = self.font.render(line, True, self.text_color)
            self.screen.blit(text_surface, (self.screen.get_width() - self.info_panel_width + 10, y_offset))
            y_offset += 30

        text_lines = [
            mpc_solve_time,
            square_solve_time
        ]

        y_offset += 40
        for line in text_lines:
            text_surface = self.font.render(line, True, self.text_color)
            self.screen.blit(text_surface, (self.screen.get_width() - self.info_panel_width + 10, y_offset))
            y_offset += 30

        text_surface = self.font.render(f"Total elapsed time: {mpc_elapsed + square_elapsed:.4f} seconds", True, (255, 0, 0))
        self.screen.blit(text_surface, (self.screen.get_width() - self.info_panel_width + 10, y_offset))

    def update(self, position, trail, state, mpc):
        """
        Updates the visualization by drawing the robot, predicted trajectory,
                and relevant information.
        """
        self.screen.fill((0, 0, 0))  # Clear the screen

        # Draw the obstacle map
        self.obstacle_map.draw(self.screen)
        self.draw_grid()

        if hasattr(mpc, 'corners') and mpc.corners is not None:
            for square in mpc.corners:
                pygame.draw.polygon(self.screen, (250, 250, 0), [(int(x), int(y)) for x, y in square], 2)

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
        self.draw_info_panel(mpc.status,
            mpc.current_distance,
            state['vt'],
            state['vr'],
            state['theta'],
            mpc.solve_time,
            mpc.stat_obj.solve_time,
            mpc.elapsed,
            mpc.stat_obj.elapsed
        )

        pygame.display.flip()
