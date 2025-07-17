#!/usr/bin/env python3
"""
Filename: create_map.py
Description:
    Interactive map creator that stores pixel-level obstacles.
"""
import pygame
import csv

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 20
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pixel-Level Obstacle Map Creator")

# Colors
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
BLACK = (0, 0, 0)

# Store obstacles at pixel-level
obstacles = set()

def draw_grid():
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (WIDTH, y))

def draw_obstacles():
    for (x, y) in obstacles:
        screen.set_at((x, y), RED)

# Main loop
running = True
while running:
    screen.fill(WHITE)
    draw_grid()
    draw_obstacles()
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click to add an obstacle block
                snapped_x = event.pos[0] // GRID_SIZE * GRID_SIZE
                snapped_y = event.pos[1] // GRID_SIZE * GRID_SIZE
                for dx in range(GRID_SIZE):
                    for dy in range(GRID_SIZE):
                        px = snapped_x + dx
                        py = snapped_y + dy
                        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                            obstacles.add((px, py))

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                with open("empty_map.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    for (x, y) in sorted(obstacles):
                        writer.writerow([x, y])
                print("Map saved to pixel_obstacle_map.csv")
                running = False

pygame.quit()
