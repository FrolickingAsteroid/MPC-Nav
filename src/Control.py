import pygame
import numpy as np

class ControlObject:
    def __init__(self, position, speed = 0, steer_angle = -np.pi / 2):
        self.position = np.array(position, dtype=float)
        self.speed = speed
        self.steer_angle = steer_angle
        self.max_speed = 5
        self.max_steer = np.pi / 4

        self.scale_factor = 100 # 10 pixels per meter

        self.position_history = []

    def update(self, dt):
        # Update the position based on speed and steering angle
        direction = np.array([np.cos(self.steer_angle), np.sin(self.steer_angle)])
        self.position += direction * self.speed * dt

        self.position_history.append(self.position.copy())

    def handle_input(self, keys):
        if keys[pygame.K_UP]:
            self.speed = min(self.speed + 1, self.max_speed)
        elif keys[pygame.K_DOWN]:
            self.speed = max(self.speed - 1, -self.max_speed)

        if keys[pygame.K_LEFT]:
            self.steer_angle -= 0.05
        elif keys[pygame.K_RIGHT]:
            self.steer_angle += 0.05

        # Normalize the steering angle to remain within -pi to pi
        self.steer_angle = (self.steer_angle + np.pi) % (2 * np.pi) - np.pi

    def get_target(self):
        return (self.position) #add the distance

    def get_position(self):
        # Convert to integer for rendering
        return self.position.astype(int)


    def get_history(self):
        # Return the position history
        return [pos.astype(int) for pos in self.position_history]
