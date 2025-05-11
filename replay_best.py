import pygame
import numpy as np
from snake_game import SnakeGame
import time

# Load the saved best action sequence
best_actions = np.load("best_run_actions.npy")

# Initialize the game
game = SnakeGame(width=450, height=450, block_size=10)
game.reset()

# Pygame setup for overlay text
pygame.init()
font = pygame.font.SysFont('Arial', 25)
screen = game.display

# Replay the actions
score = 0
for step, action in enumerate(best_actions):
    _, reward, done = game.step(action)
    score += reward

    # Render the game frame
    game.render()

    # Show step and score
    step_text = font.render(f'Step: {step + 1}', True, (255, 255, 255))
    score_text = font.render(f'Score: {int(score)}', True, (255, 255, 255))
    screen.blit(step_text, (10, 10))
    screen.blit(score_text, (10, 40))
    pygame.display.update()

    # Delay between frames
    time.sleep(0.05)

    if done:
        break

# Wait before quitting
time.sleep(2)
pygame.quit()
