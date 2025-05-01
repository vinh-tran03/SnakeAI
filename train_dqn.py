import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from snake_env import SnakeEnv  # make sure snake_env.py and snake_game.py are in the same folder

# Create and check the environment
env = SnakeEnv()
check_env(env, warn=True)  # Optional but useful for debugging

# Train the model
model = DQN(
    policy="CnnPolicy",   # Because observation is a grid (image-like)
    env=env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    tensorboard_log="./snake_tensorboard/"
)

# Start training
model.learn(total_timesteps=500_000)

# Save the model
model.save("dqn_snake")

# Optional: evaluate the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
