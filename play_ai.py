import pygame
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from snake_env import SnakeEnv
import time

pygame.init()

# Load model and wrap env in DummyVecEnv
env = DummyVecEnv([lambda: SnakeEnv()])
model = DQN.load("dqn_snake")

font = pygame.font.SysFont('Arial', 25)
episode = 1

obs = env.reset()
done = False

while True:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    reward = reward[0]
    done = done[0]

    # Render the game
    env.envs[0].render()

    # Overlay text
    reward_text = font.render(f'Reward: {reward}', True, (255, 255, 255))
    life_text = font.render(f'Life: {episode}', True, (255, 255, 255))
    env.envs[0].game.display.blit(reward_text, (10, 10))
    env.envs[0].game.display.blit(life_text, (10, 40))
    pygame.display.update()

    time.sleep(0.05)

    if done:
        episode += 1
        obs = env.reset()
