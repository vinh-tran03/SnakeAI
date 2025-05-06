import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from snake_env import SnakeEnv


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


# Create the environment and wrap it in a vectorized environment
env = DummyVecEnv([lambda: Monitor(SnakeEnv())])

# Set policy network architecture
policy_kwargs = dict(net_arch=[128, 128])

# Define the DQN model with optimized parameters
model = DQN(
    "MlpPolicy",  # Use MLP policy
    env,
    learning_rate=5e-5,           # Try lower learning rate
    buffer_size=50_000,           # Larger replay buffer
    learning_starts=10_000,        # Start training after # steps
    batch_size=64,                # Use larger batch size for training
    tau=0.005,                    # Soft update rate of the target network
    gamma=0.99,                   # Discount factor for future rewards
    train_freq=1,                 # Frequency of training updates
    target_update_interval=100,   # Update target network every 500 steps
    exploration_fraction=0.9,     # Fraction of timesteps using random actions
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,   # Final epsilon for epsilon-greedy strategy
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./tensorboard_snake/",  # TensorBoard logging for monitoring
    #device="cuda",
)

# Train the model with the logger callback
model.learn(total_timesteps=100_000, callback=TrainLoggerCallback(check_freq=5000))

# Save the trained model
model.save("dqn_snake")

print("Training completed! Model saved as 'dqn_snake'.")
