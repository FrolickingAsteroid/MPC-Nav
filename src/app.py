import numpy as np
import pygame

from Control import ControlObject
from robot import Robot

class Visualizer:
    def __init__(self, window_size=(800, 600), title="MPC Visualization", grid_size=20):
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(title)
        self.grid_size = grid_size
        self.grid_color = (50, 50, 50)

        # Info panel
        self.font = pygame.font.Font(None, 36)
        self.panel_color = (200, 200, 200)
        self.text_color = (0, 0, 0)
        self.panel_width = 250
        self.panel_height = 80
        self.panel_x = window_size[0] - self.panel_width - 10
        self.panel_y = 10

        self.robot = Robot()

    def draw_grid(self):
        for x in range(0, self.screen.get_width(), self.grid_size):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.screen.get_height()))
        for y in range(0, self.screen.get_height(), self.grid_size):
            pygame.draw.line(self.screen, self.grid_color, (0, y), (self.screen.get_width(), y))

    def draw_info_panel(self, state, distance):
        # Draw the panel background
        pygame.draw.rect(self.screen, self.panel_color, (self.panel_x, self.panel_y, self.panel_width, self.panel_height))

        # Render and draw the state text
        state_text = self.font.render(f"{state}", True, self.text_color)
        self.screen.blit(state_text, (self.panel_x + 10, self.panel_y + 10))

        # Render and draw the distance text
        distance_text = self.font.render(f"Distance: {np.sqrt(distance):.2f}", True, self.text_color)
        self.screen.blit(distance_text, (self.panel_x + 10, self.panel_y + 50))


    def draw_arrow(self, start_pos, angle, length=20, color=(255, 0, 0), width=2):
        end_pos = (start_pos[0] + np.cos(angle) * length, start_pos[1] + np.sin(angle) * length)
        pygame.draw.line(self.screen, color, start_pos, end_pos, width)

    def update(self, position, trail, state, mpc):

        self.screen.fill((0, 0, 0))  # Clear the screen
        self.draw_grid()

        # Draw the trail of the target
        for trail_pos in trail:
            pygame.draw.circle(self.screen, (200, 200, 200), trail_pos, 2)

        # Draw the predicted trajectory from the MPC solver
        if mpc.sol is not None:
            xs = mpc.sol.value(mpc.pos_x)
            ys = mpc.sol.value(mpc.pos_y)
            thetas = mpc.sol.value(mpc.pos_theta)

            for i in range(len(xs)):
                # Draw an arrow for each predicted state
                self.draw_arrow((xs[i], ys[i]), thetas[i], length=40, color=(255, 0, 255), width=2)

        # Update the robot's state and draw it
        self.robot.update_state(state)
        self.robot.draw(self.screen)

        # Draw the target position
        pygame.draw.circle(self.screen, (255, 0, 0), position, 8)

        # Draw the info panel
        self.draw_info_panel(mpc.status, mpc.current_distance)

        # Update the display
        pygame.display.flip()
