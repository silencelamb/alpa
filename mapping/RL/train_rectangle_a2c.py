import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from rectangle_cutting_env import RectangleGridEnv

# Vectorize the environment
vec_env = DummyVecEnv([lambda: RectangleGridEnv(render_mode='human', use_tuple=False)])

# Initialize the agent
model = A2C("MlpPolicy", vec_env, verbose=1)

# Train the agent
model.learn(total_timesteps=50000)

# Save the trained model
model.save("rectangle_grid_agent_a2c")

# Load the model for testing
model = PPO.load("rectangle_grid_agent_a2c")

# Test the trained agent
obs = vec_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    if done:
        obs = vec_env.reset()
