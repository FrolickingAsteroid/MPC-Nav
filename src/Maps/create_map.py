#!/usr/bin/env python3
"""
Filename: create_map.py
Description:
    Interactive map creator
"""
import pygame
import csv


pygame.init()
WIDTH, HEIGHT = 800, 600 # needs to be the same as the sim window size
GRID_SIZE = 20
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Obstacle Map Generator")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Store obstacles as grid-aligned squares
obstacles = set()

def snap_to_grid(pos):
    x = round(pos[0] / GRID_SIZE) * GRID_SIZE
    y = round(pos[1] / GRID_SIZE) * GRID_SIZE
    return x, y

running = True
while running:
    screen.fill(WHITE)

    # Draw grid
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (WIDTH, y))

    # Draw existing obstacles
    for obstacle in obstacles:
        pygame.draw.rect(screen, RED, (*obstacle, GRID_SIZE, GRID_SIZE))

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Left click to add an obstacle
            if event.button == 1:
                snapped_pos = snap_to_grid(event.pos)
                obstacles.add(snapped_pos)

        elif event.type == pygame.KEYDOWN:
            # Press "S" to save the map and exit
            if event.key == pygame.K_s:
                with open("obstacle_grid_map.csv", "w", newline="") as file:
                    writer = csv.writer(file)
                    for obstacle in obstacles:
                        writer.writerow(obstacle)  # Save each obstacle as (x, y)
                running = False

pygame.quit()
