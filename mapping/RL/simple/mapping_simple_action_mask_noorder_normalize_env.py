import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import statistics

class MappingSimpleActionMaskNoOderNormalizeEnv(gym.Env):
    
    def __init__(self, render_mode='human', use_image=False):
        super(MappingSimpleActionMaskNoOderNormalizeEnv, self).__init__()
        
        # total compute in a microbatch
        self.compute_of_a_microbatch= 100
        self.left_compute = self.compute_of_a_microbatch
        self.num_microbatch = 1024
        self.rows, self.cols = (5, 5)
        self.grid = np.zeros((self.rows, self.cols), int)
        self.render_mode = render_mode
        
        self.use_image = use_image
        # Observation space
        if self.use_image:
            self.observation_space = spaces.Box(low=0, high=self.rows * self.cols, shape=(1, self.rows, self.cols), dtype=np.int8)
        else:
            self.observation_space = spaces.Box(low=0, high=self.rows * self.cols, shape=(self.rows, self.cols), dtype=int)
                    
        self.action_space = spaces.MultiDiscrete([50,   # ratio of ave compute, 
                                                  self.cols * self.rows * self.cols * self.rows  # rect action
                                                  ])
        # latest position of the rectangle, [left, top, right, bottom]
        self.latest_position = []
        
        self.constant = self.compute_of_a_microbatch * self.num_microbatch
        
        self.current_step = 0
        self.max_steps = 100
        self.compute_list = []
        self.rect_list = []
        self.rect_id = 0
        self.action_mask = None

        self.total_reward = 0
        self.max_ave_episode = 50
        self.reward_record = deque(maxlen=self.max_ave_episode)
        self.average_reward = 0
        self.mae_reward = 0
        self.mae_param = 0.9
        self.max_ave_episode = 0
        
    def reward(self):
        """Reward function
        """
        compute_sum = sum(self.compute_list)
        compute_ratio_list = [x/compute_sum for x in self.compute_list]
        self.compute_list = [x * self.compute_of_a_microbatch for x in compute_ratio_list]
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
        return -total_latency

    def reset(self, seed=None, options=None):
        self.grid = np.zeros((self.rows, self.cols), int)
        self.current_step = 0
        self.left_compute = self.compute_of_a_microbatch
        self.rect_list = []
        self.rect_id = 0
        self.latest_position = []
        self.compute_list = []
        
        self.total_reward = 0
        
        if self.use_image:
            return self.grid[np.newaxis, :], {}
        else:
            return self.grid, {}

    def step(self, action):
        self.current_step += 1
        ratio, rect_action  = action
        col_start, row_start, col_end, row_end  = self.decode_action(rect_action)
        ratio = (ratio+1-100)/100
        ave_compute = self.compute_of_a_microbatch / (self.cols*self.rows)
        
        # Check for valid rectangle
        is_valid = self._is_valid_rectangle(col_start, row_start, col_end, row_end)
        
        cur_compute =  (col_end-col_start+1) * (row_end-row_start+1) * ave_compute * (1+ratio)
        reward = -self.constant
        truncted = False
        done = False
        if is_valid:
            self.rect_id += 1
            self.grid[row_start:row_end+1, col_start:col_end+1] = self.rect_id
            self.rect_list.append([col_start, row_start, col_end, row_end])
            self.latest_position = [col_start, row_start, col_end, row_end]
            
            # check if compute is valid, if not then termimated is true
                
            # Check if grid is full, if true then compute is the compute left
            elif np.sum(self.grid == 0) == 0:
                cur_compute = self.left_compute
                self.compute_list.append(cur_compute)
                self.left_compute = 0
                reward = self.reward()
                done = True
            else:
                # internal state, reward is none
                reward = 0
                self.compute_list.append(cur_compute)
                self.left_compute = self.left_compute - cur_compute
                
        else:
            # invalid action

            # option 1: terminated is true, and fix the mapping, compute reward
            self.compute_list[-1] += self.left_compute
            reward = self.reward()
            done = True
            
            # # option 2: continue the game, and the reward is negative
            # reward = -self.constant/10
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"Step {self.current_step}: invalid action, {col_start, row_start, col_end, row_end}")
            # print(self.grid)
            # print(np.sum(self.action_mask))
            
        if self.current_step >= self.max_steps:
            done, truncted = True, True
        self.total_reward += reward
        if done:
            self.reward_record.append(self.total_reward)
            self.average_reward = statistics.mean(self.reward_record)
            self.mae_reward = self.mae_reward * self.mae_param + self.total_reward * (1 - self.mae_param)
            print(f'average reward: {self.average_reward}, mae reward: {self.mae_reward}, total reward: {self.total_reward}')
            print(self.grid)
            print(self.compute_list)
            
        if self.use_image:
            return self.grid[np.newaxis, :], reward, done, truncted, {}
        else:
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
        action_mask = np.zeros(self.action_space.nvec[1], dtype=np.int8)
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
        print("env rending start=======================")
        print(self.grid)
        print(self.compute_list)
        print(self.rect_list)
        print("env rending end=========================")
        
    def close(self):
        pass


if __name__ == '__main__':
    env = MappingSimpleActionMaskNoOderNormalizeEnv()
    
    best_reward = -1000
    reward_list = []
    while len(reward_list) < 10:
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # You can use a smarter sampling method or your trained agent here
            obs, reward, done, truncted, _ = env.step(action)
            print(f"Step {env.current_step}: action {action}")
            print(f"Obs: \n{obs}")
            print(f"Reward: {reward}")
        if reward > -102400:
            print(f"Reward: {reward}")
            env.render()
            reward_list.append(reward)
    
    # env.render()
    print(reward_list)
    
    """
    test_cae 1
    """
    # env.grid = np.array([[1., 1., 1., 1., 0.],
    #                      [2., 2., 2., 0., 0.],
    #                      [2., 2., 2., 0., 0.],
    #                      [2., 2., 2., 0., 0.],
    #                      [0., 0., 0., 0., 0.]])
    # action = [52, 0, 3, 4]
    # env.latest_position = [0, 1, 2, 3]
    # env.rect_id = 2
    # env.rect_list = [
    #     [0, 0, 3, 0],
    #     [0, 1, 2, 3],
    # ]
    # obs, reward, done, truncted, _ = env.step(action)
    # print(obs)
    # print(env.grid)
