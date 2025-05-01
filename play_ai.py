import pygame
from stable_baselines3 import DQN
from snake_env import SnakeEnv
import time

# Initialize Pygame and fonts for rendering text
pygame.init()

# Load environment
env = SnakeEnv()

# Load trained model
model = DQN.load("dqn_snake")

# Reset environment
obs = env.reset()
done = False

# Set up font for displaying text (reward score)
font = pygame.font.SysFont('Arial', 25)

# Start game loop
while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()  # Make sure SnakeGame's render() calls pygame.display.update()

    # Display the current reward on the screen
    reward_text = font.render(f'Reward: {reward}', True, (255, 255, 255))  # White text
    env.display.blit(reward_text, (10, 10))  # Place it at the top-left of the screen

    pygame.display.update()  # Update the Pygame window to show the text
    time.sleep(0.05)  # Optional: slow it down so you can see it play

    if done:
        obs = env.reset()
