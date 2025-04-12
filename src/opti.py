#!/usr/bin/env python3
"""
Filename: opti.py
Description:
    Runs a real-time simulation of an MPC people-following robot.
"""
import numpy as np
import pygame

from Control import ControlObject
from functions import Auxiliary
from app import Visualizer
from mpc import MPC

class Simulation:
    def __init__(self, window_size=(800, 600)):
        self.N = 10 # number of control intervals
        self.T = 1

        self.dt = self.T / self.N

        # Initialize robot's state
        # Position is given in pixels and set by the user
        self.state = {'x':0.0, 'y': 0.0, 'theta': -np.pi/2, 'vt': 0.0, 'vr': 0.0}

        self.clock = pygame.time.Clock()

        self.map = "Maps/obstacle_grid_map.csv"
        self.visualizer = Visualizer(self.map)
        self.mpc = MPC(self.T, self.N, self.state, self.map)
        self.fs = Auxiliary()

        self.path = []

        # define target and robot position
        self.user_init()
        self.running = True

    def user_init(self):
            """
            Set target and robot position
            """
            clicked = 0
            font = pygame.font.SysFont(None, 24)
            while clicked < 2:
                self.visualizer.draw_static_map()
                text = "Set robot position" if clicked == 0 else "Set target position"
                label = font.render(text, True, (255, 255, 255))
                self.visualizer.screen.blit(label, (10, 10))
                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        return
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        if clicked == 0:
                            self.state['x'], self.state['y'] = pos
                            clicked += 1
                        elif clicked == 1:
                            self.control_object = ControlObject(list(pos))
                            clicked += 1

    def run(self):
        """
         Main simulation loop
        """
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # Update target position based on current velocity and steering
            keys = pygame.key.get_pressed()
            self.control_object.handle_input(keys)
            self.control_object.update(self.dt)

            target = self.control_object.get_target()

            # Run mpc optimization and update robot state
            if self.mpc.mpc_opt(self.state, target):
                self.state = self.fs.simulate_step(
                    dt=self.mpc.dt,
                    state=self.state,
                    control=self.mpc.get_control(i=0),
                    configs=self.mpc.configs
                )
                self.path.append(self.state)  # Store trajectory

            # Update visualization
            self.visualizer.update(
                self.control_object.get_position(),
                self.control_object.get_history(),
                self.state,
                self.mpc
            )

        pygame.quit()


# Run the simulation
if __name__ == "__main__":
    sim = Simulation()
    sim.run()
