from gym import Env
from gym.spaces import Discrete, Box
from snake_game import SnakeGame
import numpy as np

class SnakeEnv(Env):
    def __init__(self):
        super().__init__()
        self.game = SnakeGame()
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=1, shape=(48, 72, 3), dtype=np.uint8)

    def step(self, action):
        state, reward, done = self.game.step(action)
        return state, reward, done, {}

    def reset(self):
        return self.game.reset()

    def render(self, mode='human'):
        self.game.render()
