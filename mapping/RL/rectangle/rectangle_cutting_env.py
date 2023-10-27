import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import statistics

class RectangleGridEnv(gym.Env):
    def __init__(self, render_mode='human', use_tuple=False, use_image=False):
        super(RectangleGridEnv, self).__init__()
        
        self.rows, self.cols = (5, 5)
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.render_mode = render_mode
        
        self.use_tuple = use_tuple
        # Action space: (col_start, row_start, col_end, row_end)
        if use_tuple:
            self.action_space = spaces.Tuple(
                (spaces.Discrete(self.cols), 
                spaces.Discrete(self.rows), 
                spaces.Discrete(self.cols), 
                spaces.Discrete(self.rows))
            )
        else:
            self.action_space = spaces.Discrete(self.cols * self.rows * self.cols * self.rows)
        
        self.use_image = use_image
        # Observation space
        if self.use_image:
            self.observation_space = spaces.Box(low=0, high=self.rows * self.cols, shape=(1, self.rows, self.cols), dtype=np.int8)
        else:
            self.observation_space = spaces.Box(low=0, high=self.rows * self.cols, shape=(self.rows, self.cols), dtype=int)
        
        self.current_step = 0
        self.max_steps = 128
        self.rect_list = []
        self.rect_id = 0
        self.total_reward = 0
        self.max_ave_episode = 50
        self.reward_record = deque(maxlen=self.max_ave_episode)
        self.average_reward = 0
        self.mae_reward = 0
        self.mae_param = 0.9
        self.max_ave_episode = 0
    
    def reset(self, seed=None, options=None):
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.current_step = 0
        self.rect_list = []
        self.rect_id = 0
        self.total_reward = 0
        
        return self.grid, {}

    def step(self, action):
        if self.use_tuple:
            col_start, row_start, col_end, row_end  = action
        else:
            col_start, row_start, col_end, row_end  = self.decode_action(action)
        
        # Check for valid rectangle
        is_valid = self._is_valid_rectangle(col_start, row_start, col_end, row_end)
        
        reward = 0
        if is_valid:
            self.rect_id += 1
            self.grid[row_start:row_end+1, col_start:col_end+1] = self.rect_id
            reward = 1
            self.rect_list.append([col_start, row_start, col_end, row_end])
            
            
            # Check if grid is full
            if np.sum(self.grid == 0) == 0:
                reward = len(self.rect_list)
        else:
            reward = -1
            
        self.current_step += 1
        truncted = (self.current_step >= self.max_steps)
        done = (np.sum(self.grid == 0) == 0) or truncted
        self.total_reward += reward
        if done:
            self.reward_record.append(self.total_reward)
            self.average_reward = statistics.mean(self.reward_record)
            self.mae_reward = self.mae_reward * self.mae_param + self.total_reward * (1 - self.mae_param)
            print(f'average reward: {self.average_reward}, mae reward: {self.mae_reward}, total reward: {self.total_reward}')
        return self.grid, reward, done, truncted, {}


    def decode_action(self, action):
        a, b, c, d = np.unravel_index(action, (self.cols, self.rows, self.cols, self.rows))
        return a, b, c, d


    def _is_valid_rectangle(self, col_start, row_start, col_end, row_end):
        if row_end < row_start or col_end < col_start:
            return False
        if row_end >= self.rows or col_end >= self.cols:
            return False
        if np.sum(self.grid[row_start:row_end+1, col_start:col_end+1]) > 0:
            # equivalent to iou
            return False
        return True
    
    def get_action_mask(self):
        """get action mask
        Args:
            None
        Return: 
        action_mask: 0 for invalid action, 1 for valid action；
        """
        action_mask = np.zeros(self.action_space.n)
        for left in range(self.cols):
            for top in range(self.rows):
                for right in range(left, self.cols):
                    for bottom in range(top, self.rows):
                        if np.sum(self.grid[top:bottom+1, left:right+1]) == 0:
                            index = np.ravel_multi_index((left, top, right, bottom), 
                                                         (self.cols, self.rows, self.cols, self.rows)
                                                         )
                            action_mask[index] = 1
        return action_mask

    def render(self, mode=None):
        # This can be improved, but for simplicity, just print the grid
        mode = mode or self.render_mode
        print(self.grid)
        
    def close(self):
        pass


if __name__ == '__main__':
    env = RectangleGridEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # You can use a smarter sampling method or your trained agent here
        action_mask = env.get_action_mask()
        obs, reward, done, truncted, _ = env.step(action)
        env.render()
    env.render()
    print(env.rect_list)
