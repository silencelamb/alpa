import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RectangleGridEnv(gym.Env):
    def __init__(self, render_mode='human'):
        super(RectangleGridEnv, self).__init__()
        
        self.rows, self.cols = (5, 5)
        self.grid = np.zeros((self.rows, self.cols))
        self.render_mode = render_mode
        
        # Action space: (col_start, row_start, col_end, row_end)
        # self.action_space = spaces.Tuple(
        #     (spaces.Discrete(self.cols), 
        #      spaces.Discrete(self.rows), 
        #      spaces.Discrete(self.cols), 
        #      spaces.Discrete(self.rows))
        # )
        self.action_space = spaces.Discrete(self.cols * self.rows * self.cols * self.rows)
        
        # Observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.rows, self.cols), dtype=int)
        
        self.current_step = 0
        self.max_steps = 2000
        self.rect_list = []
        self.rect_id = 0

    def reset(self, seed=None):
        self.grid = np.zeros((self.rows, self.cols))
        self.current_step = 0
        self.rect_list = []
        self.rect_id = 0
        
        return self.grid, {}

    def step(self, action):
        # col_start, row_start, col_end, row_end  = action
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
                reward = 10
        else:
            reward = -1
            
        self.current_step += 1
        truncted = (self.current_step >= self.max_steps)
        done = (np.sum(self.grid == 0) == 0) or truncted
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
    
    def compute_iou(self, row_start, col_start, row_end, col_end):
        #TODO: Implement this IOU function
        pass

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
        obs, reward, done, truncted, _ = env.step(action)
        env.render()
    env.render()
    print(env.rect_list)
