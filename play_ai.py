from stable_baselines3 import DQN
from snake_env import SnakeEnv
import time

# Load environment
env = SnakeEnv()

# Load trained model
model = DQN.load("dqn_snake")

# Reset environment
obs = env.reset()
done = False

# Play the game using the trained AI
while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()  # Make sure SnakeGame's render() calls pygame.display.update()
    time.sleep(0.05)  # Optional: slow it down so you can see it play

    if done:
        obs = env.reset()
