import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from mapping_env_direction_basepoint import MappingEnv
import sys
sys.path.insert(0, "..")
from cnn_extractor import MeshCNN

# Vectorize the environment
vec_env = DummyVecEnv([lambda: MappingEnv(render_mode='human', use_image=True)])

# Initialize the agent
model = PPO("CnnPolicy", vec_env, verbose=1, 
            policy_kwargs=dict(
                features_extractor_class=MeshCNN,
                features_extractor_kwargs=dict(
                    normalized_image=True
                    )
                )
        )
            

# Train the agent
model.learn(total_timesteps=50000)

# Save the trained model
model.save("mapping_simple_mask_all_ave_agent_cnn")

# Load the model for testing
model = PPO.load("mapping_simple_mask_all_ave_agent_cnn")

# Test the trained agent
reward_list = []
while len(reward_list) < 10:
    done = False
    obs = vec_env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        print(f"Step {vec_env.envs[0].current_step}: {action}")
        obs, reward, done, info = vec_env.step(action)
        print(f"Obs: {obs}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
    if reward:
        print(f"Reward: {reward}")
        vec_env.render()
        reward_list.append(reward[0])
print(reward_list)
