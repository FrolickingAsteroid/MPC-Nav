#!/usr/bin/env python3
"""
Filename: opti.py
Description:
    Runs a real-time simulation of an MPC people-following robot.
"""
import numpy as np
import pygame
import time
import math

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
        # Position is given in pixels
        self.state = {'x':400.0, 'y': 350.0, 'theta': -np.pi/2, 'vt': 0.0, 'vr': 0.0}

        self.clock = pygame.time.Clock()
        self.visualizer = Visualizer()
        self.mpc = MPC(self.T, self.N, self.state)
        self.fs = Auxiliary()

        self.path = []

        # Initialize target at the center of the screen
        self.control_object = ControlObject([window_size[0] // 2, window_size[1] // 2])
        self.running = True

    def run(self):
        """
         Main simulation loop
        """
        while self.running:
            #dt = self.clock.tick(60) / 1000.0  # deltaTime in seconds

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
