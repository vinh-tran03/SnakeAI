import pygame
from stable_baselines3 import DQN
from snake_env import SnakeEnv
import time

# Initialize Pygame and fonts for rendering text
pygame.init()

# Load environment and trained model
env = SnakeEnv()
model = DQN.load("dqn_snake")

# Set up font for displaying text
font = pygame.font.SysFont('Arial', 25)

# Initialize episode counter
episode = 1

# Reset environment
obs = env.reset()
done = False

# Start game loop
while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()

    # Render reward and episode ("life") on screen
    reward_text = font.render(f'Reward: {reward}', True, (255, 255, 255))
    life_text = font.render(f'Life: {episode}', True, (255, 255, 255))

    # Blit to the display (must come after env.render() to not get erased)
    env.game.display.blit(reward_text, (10, 10))
    env.game.display.blit(life_text, (10, 40))

    pygame.display.update()
    time.sleep(0.05)

    if done:
        episode += 1
        obs = env.reset()
