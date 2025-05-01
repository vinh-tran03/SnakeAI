from stable_baselines3 import DQN
from snake_env import SnakeEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

# Custom callback to log training progress
class TrainLoggerCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Called at each step
        if self.locals.get('done'):
            reward = self.locals.get('rewards')
            self.episode_rewards.append(reward)

        # Print progress every check_freq steps
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            print(f"Step: {self.n_calls} | Mean Reward (last 100): {mean_reward:.2f}")
        return True


# Create environment
env = SnakeEnv()

# Instantiate the model
model = DQN("MlpPolicy", env, verbose=0, tensorboard_log="./tensorboard_snake/")

# Train the model with the logger callback
model.learn(total_timesteps=500_000, callback=TrainLoggerCallback(check_freq=5000))

# Save the trained model
model.save("dqn_snake")
