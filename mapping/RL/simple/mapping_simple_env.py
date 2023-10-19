import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MappingSimpleEnv(gym.Env):
    def __init__(self, render_mode='human', use_image=False):
        super(MappingSimpleEnv, self).__init__()
        
        # total compute in a microbatch
        self.compute_of_a_microbatch= 100
        self.num_microbatch = 1024
        self.rows, self.cols = (5, 5)
        self.grid = np.zeros((self.rows, self.cols))
        self.render_mode = render_mode
        
        self.use_image = use_image
        # Observation space
        if self.use_image:
            self.observation_space = spaces.Box(low=0, high=self.rows * self.cols, shape=(1, self.rows, self.cols), dtype=np.int8)
        else:
            self.observation_space = spaces.Box(low=0, high=self.rows * self.cols, shape=(self.rows, self.cols), dtype=int)
                    
        self.action_space = spaces.MultiDiscrete([self.compute_of_a_microbatch, 
                                                  self.cols, self.rows, self.cols, self.rows])
        
        self.constant = self.compute_of_a_microbatch * self.num_microbatch
        
        self.current_step = 0
        self.max_steps = self.rows * self.cols
        self.compute_list = []
        self.rect_list = []
        self.rect_id = 0
        
    def reward(self):
        """Reward function
        """
        rect_num = self.rect_id
        bubble_ratio = rect_num / self.num_microbatch
        stage_latency_list = []
        for compute, submesh in zip(self.compute_list, self.rect_list):
            left, top, right, bottom = submesh
            die_num = (right - left + 1) * (bottom - top + 1)
            
            communication_ratio = die_num/25
            compute_time = compute / die_num
            stage_time = compute_time * (1+communication_ratio)  # condider communication ratio
            
            stage_latency_list.append(stage_time)
        max_stage_lantecy = max(stage_latency_list)
        # reshard_comm = 
        total_latency = sum(stage_latency_list) + max_stage_lantecy*(self.num_microbatch-1)
        return self.constant-total_latency

    def reset(self, seed=None):
        self.grid = np.zeros((self.rows, self.cols))
        self.current_step = 0
        self.rect_list = []
        self.rect_id = 0
        
        return self.grid, {}

    def step(self, action):
        cur_compute, col_start, row_start, col_end, row_end  = action
        self.current_step += 1
        # Check for valid rectangle
        is_valid = self._is_valid_rectangle(col_start, row_start, col_end, row_end)
        
        reward = -1000
        if is_valid:
            self.rect_id += 1
            self.grid[row_start:row_end+1, col_start:col_end+1] = self.rect_id
            self.rect_list.append([col_start, row_start, col_end, row_end])
            
            # check if compute is valid, if not then termimated is true
            if cur_compute > self.compute_of_a_microbatch:
                cur_compute = self.compute_of_a_microbatch
                self.compute_list.append(cur_compute)
                reward = self.reward()
                return self.grid, reward, True, False, {}
                
            # Check if grid is full, if true then compute is the compute left
            elif np.sum(self.grid == 0) == 0:
                cur_compute = self.compute_of_a_microbatch
                self.compute_list.append(cur_compute)
                reward = self.reward()
                return self.grid, reward, True, False, {}
            else:
                # internal state, reward is none
                reward = 0
                return self.grid, reward, False, False, {}
                
        else:
            # invalid action, terminated is true, and the reward is very negative
            reward = -1000
            return self.grid, reward, True, False, {}


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
    env = MappingSimpleEnv()
    
    best_reward = -1000
    reward_list = []
    while len(reward_list) < 10:
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # You can use a smarter sampling method or your trained agent here
            obs, reward, done, truncted, _ = env.step(action)
            # print(f"Step {env.current_step}: {action}")
            # print(f"Obs: {obs}")
            # print(f"Reward: {reward}")
            # env.render()
        if reward > -1000:
            print(f"Reward: {reward}")
            env.render()
            reward_list.append(reward)
        
    env.render()
    print(reward_list)
