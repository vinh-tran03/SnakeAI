#snake_env.py
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from snake_game import SnakeGame
import numpy as np

class SnakeEnv(Env):
    def __init__(self):
        super().__init__()
        self.game = SnakeGame(width=450, height=450, block_size=10)  
        self.action_space = Discrete(4)

        # Dynamically determine observation space shape
        initial_state = self.game.reset()
        self.observation_space = Box(low=0, high=1, shape=initial_state.shape, dtype=np.uint8)

    def step(self, action):
        state, reward, done = self.game.step(action)
        truncated = False  # Modify this if needed for time-limited episodes
        info = {}
        return state, reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        state = self.game.reset()
        info = {}
        return state, info

    def render(self):
        self.game.render()
