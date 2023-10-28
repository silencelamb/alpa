import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import statistics

class RectangleGridOrderEnv(gym.Env):
    
    def __init__(self, render_mode='human', use_image=False):
        super(RectangleGridOrderEnv, self).__init__()
        

        self.rows, self.cols = (5, 5)
        self.grid = np.zeros((self.rows, self.cols), int)
        self.render_mode = render_mode
        
        self.use_image = use_image
        # Observation space
        if self.use_image:
            self.observation_space = spaces.Box(low=0, high=self.rows * self.cols, shape=(1, self.rows, self.cols), dtype=np.int8)
        else:
            self.observation_space = spaces.Box(low=0, high=self.rows * self.cols, shape=(self.rows, self.cols), dtype=int)
                    
        self.action_space = spaces.Discrete(self.cols * self.rows * self.cols * self.rows)
        
        # latest position of the rectangle, [left, top, right, bottom]
        self.latest_position = []
        
        self.current_step = 0
        self.max_steps = 100
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
        

    def reset(self, seed=None, options=None):
        self.grid = np.zeros((self.rows, self.cols), int)
        self.current_step = 0
        self.rect_list = []
        self.rect_id = 0
        self.latest_position = []
        
        self.total_reward = 0
        
        return self.grid, {}

    def step(self, action):
        self.current_step += 1
        col_start, row_start, col_end, row_end  = self.decode_action(action)
       
        # Check for valid rectangle
        is_valid = self._is_valid_rectangle(col_start, row_start, col_end, row_end)

        reward = -1
        truncted = False
        done = False
        if is_valid:
            self.rect_id += 1
            self.grid[row_start:row_end+1, col_start:col_end+1] = self.rect_id
            self.rect_list.append([col_start, row_start, col_end, row_end])
            self.latest_position = [col_start, row_start, col_end, row_end]
            
            # Check if grid is full
            if np.sum(self.grid == 0) == 0:
                reward = len(self.rect_list)
                done = True
            else:
                # internal state, reward is none
                reward = 0
                
        else:
            # invalid action
            reward = -25 + len(self.rect_list)
            done = True

        if self.current_step >= self.max_steps:
            done, truncted = True, True
        self.total_reward += reward
        if done:
            print(self.grid)
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
        if self.latest_position != []  and \
            not self.is_adjacent_no_overlap(
                (col_start, row_start, col_end, row_end), 
                self.latest_position):
            # not adjacent to the latest position
            return False
        return True

    def is_adjacent_no_overlap(self, rect1, rect2):
        col_start1, row_start1, col_end1, row_end1 = rect1
        col_start2, row_start2, col_end2, row_end2 = rect2

        # 检查是否在水平方向相邻
        horizontal_adjacent = (
            (col_end1+1 == col_start2 and not (row_start1 > row_end2 or row_end1 < row_start2)) or
            (col_start1 == col_end2+1 and not (row_start1 > row_end2 or row_end1 < row_start2))
        )

        # 检查是否在垂直方向相邻
        vertical_adjacent = (
            (row_end1+1 == row_start2 and not (col_start1 > col_end2 or col_end1 < col_start2)) or
            (row_start1 == row_end2+1 and not (col_start1 > col_end2 or col_end1 < col_start2))
        )

        return horizontal_adjacent or vertical_adjacent
    
    def get_action_mask(self):
        """get action mask
        Args:
            None
        Return: 
        action_mask: 0 for invalid action, 1 for valid action；
        """
        action_mask = np.zeros(self.action_space.n, dtype=np.int8)
        if self.latest_position == []:
            # actions start from (0, 0)
            left, top = 0, 0
            for right in range(left, self.cols):
                for bottom in range(top, self.rows):
                    if np.sum(self.grid[top:bottom+1, left:right+1]) == 0:
                        index = np.ravel_multi_index((left, top, right, bottom), 
                                                        (self.cols, self.rows, self.cols, self.rows)
                                                        )
                        action_mask[index] = 1
        else:
            # actions should be adjacent to the latest position
            latest_left, latest_top, latest_right, latest_bottom = self.latest_position
            # case 1: position -> bottom adjacent -> left side
            left, top = latest_left, latest_bottom+1
            for right in range(left, self.cols):
                for bottom in range(top, self.rows):
                    if np.sum(self.grid[top:bottom+1, left:right+1]) == 0:
                        index = np.ravel_multi_index((left, top, right, bottom), 
                                                        (self.cols, self.rows, self.cols, self.rows)
                                                        )
                        action_mask[index] = 1
            # case 2: position -> bottom adjacent -> right side
            right = latest_right
            top = latest_bottom + 1
            for left in range(right+1):
                for bottom in range(top, self.rows):
                    if np.sum(self.grid[top:bottom+1, left:right+1]) == 0:
                        index = np.ravel_multi_index((left, top, right, bottom), 
                                                        (self.cols, self.rows, self.cols, self.rows)
                                                        )
                        action_mask[index] = 1
            # case 3: position -> right adjacent-> top side
            left, top = latest_right+1, latest_top
            for right in range(left, self.cols):
                for bottom in range(top, self.rows):
                    if np.sum(self.grid[top:bottom+1, left:right+1]) == 0:
                        index = np.ravel_multi_index((left, top, right, bottom), 
                                                        (self.cols, self.rows, self.cols, self.rows)
                                                        )
                        action_mask[index] = 1
            # case 4: position -> right adjacent-> bottom side
            left, bottom = latest_right+1, latest_bottom
            for right in range(left, self.cols):
                for top in range(bottom+1):
                    if np.sum(self.grid[top:bottom+1, left:right+1]) == 0:
                        index = np.ravel_multi_index((left, top, right, bottom), 
                                                        (self.cols, self.rows, self.cols, self.rows)
                                                        )
                        action_mask[index] = 1
            # case 5: position -> top adjacent-> left side
            left, bottom = latest_left, latest_top-1
            for right in range(left, self.cols):
                for top in range(bottom+1):
                    if np.sum(self.grid[top:bottom+1, left:right+1]) == 0:
                        index = np.ravel_multi_index((left, top, right, bottom), 
                                                        (self.cols, self.rows, self.cols, self.rows)
                                                        )
                        action_mask[index] = 1
            # case 6: position -> top adjacent-> right side
            right, bottom = latest_right, latest_top-1
            for left in range(right+1):
                for top in range(bottom+1):
                    if np.sum(self.grid[top:bottom+1, left:right+1]) == 0:
                        index = np.ravel_multi_index((left, top, right, bottom), 
                                                        (self.cols, self.rows, self.cols, self.rows)
                                                        )
                        action_mask[index] = 1
            # case 7: position -> left adjacent-> top side
            right, top = latest_left-1, latest_top
            for left in range(right+1):
                for bottom in range(top, self.rows):
                    if np.sum(self.grid[top:bottom+1, left:right+1]) == 0:
                        index = np.ravel_multi_index((left, top, right, bottom), 
                                                        (self.cols, self.rows, self.cols, self.rows)
                                                        )
                        action_mask[index] = 1
            # case 8: position -> left adjacent-> bottom side
            right, bottom = latest_left-1, latest_bottom
            for left in range(right+1):
                for top in range(bottom+1):
                    if np.sum(self.grid[top:bottom+1, left:right+1]) == 0:
                        index = np.ravel_multi_index((left, top, right, bottom), 
                                                        (self.cols, self.rows, self.cols, self.rows)
                                                        )
                        action_mask[index] = 1
        self.action_mask = action_mask
        return action_mask        

    def render(self, mode=None):
        # This can be improved, but for simplicity, just print the grid
        mode = mode or self.render_mode
        print("env rending start=======================")
        print(self.grid)
        print(self.rect_list)
        print("env rending end=========================")
        
    def close(self):
        pass


if __name__ == '__main__':
    env = RectangleGridOrderEnv()
    
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
