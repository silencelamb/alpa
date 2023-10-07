import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rectangle_cutting_env import RectangleGridEnv
from cnn_extractor import MeshCNN

# Vectorize the environment
vec_env = DummyVecEnv([lambda: RectangleGridEnv(render_mode='human', use_image=True)])

# Initialize the agent
model = PPO("CnnPolicy", vec_env, verbose=1, 
            policy_kwargs=dict(
                features_extractor_class=MeshCNN,
                features_extractor_kwargs=dict(
                    normalized_image=True
                    )
                )
        )
            

# # Train the agent
# model.learn(total_timesteps=50000)

# # # Save the trained model
# model.save("rectangle_grid_agent_cnn")

# Load the model for testing
model = PPO.load("rectangle_grid_agent_cnn")

# Test the trained agent
obs = vec_env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # if done:
    #     obs = vec_env.reset()
