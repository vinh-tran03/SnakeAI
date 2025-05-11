#train_dqn.py
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from snake_env import SnakeEnv

# Custom callback to log training progress and save best run
class TrainLoggerCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.best_score = -np.inf
        self.best_actions = []
        self.current_episode_actions = []
        self.episode_reward = 0

    def _on_step(self) -> bool:
        action = self.locals["actions"][0]
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self.current_episode_actions.append(action)
        self.episode_reward += reward

        if done:
            self.episode_rewards.append(self.episode_reward)

            if self.episode_reward > self.best_score:
                self.best_score = self.episode_reward
                self.best_actions = self.current_episode_actions.copy()
                np.save("best_run_actions.npy", np.array(self.best_actions))  # Save actions
                print(f"\n🚀 New best score: {self.best_score:.2f} — saved best_run_actions.npy")

            self.current_episode_actions = []
            self.episode_reward = 0

        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            print(f"Step: {self.n_calls} | Mean Reward (last 100): {mean_reward:.2f}")
        return True

# Create the environment and wrap it in a vectorized environment
env = DummyVecEnv([lambda: Monitor(SnakeEnv())])

# Set policy network architecture
policy_kwargs = dict(net_arch=[128, 128])

# Define the DQN model with optimized parameters
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-5,
    buffer_size=50_000,
    learning_starts=10_000,
    batch_size=64,
    tau=0.005,
    gamma=0.95,
    train_freq=1,
    target_update_interval=100,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./tensorboard_snake/",
    device="cuda",  # Use "cpu" if CUDA is not available
)

# Train the model with the logger callback
model.learn(total_timesteps=100_000, callback=TrainLoggerCallback(check_freq=5000))

# Save the trained model
model.save("dqn_snake")

print("Training completed! Model saved as 'dqn_snake'.")
