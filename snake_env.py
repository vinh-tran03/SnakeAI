from gymnasium import Env
from gymnasium.spaces import Discrete, Box
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
        truncated = False  # You can handle this if needed
        info = {}
        return state, reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        state = self.game.reset()
        info = {}
        return state, info

    def render(self):
        self.game.render()
