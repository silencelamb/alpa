import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MappingSimpleMaskALLEnv(gym.Env):
    
    def __init__(self, render_mode='human', use_image=False):
        super(MappingSimpleMaskALLEnv, self).__init__()
        
        # total compute in a microbatch
        self.compute_of_a_microbatch= 100
        self.left_compute = self.compute_of_a_microbatch
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
                                                  4,     # direction
                                                  2,     # basepoint
                                                  self.cols, self.rows])
        # latest position of the rectangle, [left, top, right, bottom]
        self.latest_position = []
        
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
        return -total_latency

    def reset(self, seed=None):
        self.grid = np.zeros((self.rows, self.cols))
        self.current_step = 0
        self.left_compute = self.compute_of_a_microbatch
        self.rect_list = []
        self.rect_id = 0
        self.latest_position = []
        self.compute_list = []
        
        return self.grid, {}

    def step(self, action):
        # import pdb; pdb.set_trace()
        cur_compute, direction, base_point, width, height  = action
        assert base_point in [0, 1]
        cur_compute += 1
        width += 1
        height += 1
        self.current_step += 1
        if self.latest_position == []:
            col_start, row_start, col_end, row_end = 0, 0, width-1, height-1 
            base_col, base_row = col_start, row_start
        elif direction == 0:
            # up
            if base_point == 0:
                col_start = self.latest_position[0]
                col_end = col_start + width - 1
                row_end = self.latest_position[1] - 1
                row_start = row_end - height + 1
                base_col, base_row = col_start, row_end
            else:
                col_end = self.latest_position[2]
                col_start = col_end - width + 1
                row_end = self.latest_position[1] - 1
                row_start = row_end - height + 1
                base_col, base_row = col_end, row_end
        elif direction == 1:
            # right
            if base_point == 0:
                col_start = self.latest_position[2] + 1
                col_end = col_start + width - 1
                row_start = self.latest_position[1]
                row_end = row_start + height - 1
                base_col, base_row = col_start, row_start
            else:
                col_start = self.latest_position[2] + 1
                col_end = col_start + width - 1
                row_end = self.latest_position[3]
                row_start = row_end - height + 1
                base_col, base_row = col_start, row_end
        elif direction == 2:
            # down 
            if base_point == 0:
                col_start = self.latest_position[0]
                col_end = col_start + width - 1
                row_start = self.latest_position[3] + 1
                row_end = row_start + height - 1
                base_col, base_row = col_start, row_start
            else:
                col_end = self.latest_position[2]
                col_start = col_end - width + 1
                row_start = self.latest_position[3] + 1
                row_end = row_start + height - 1
                base_col, base_row = col_end, row_start
        elif direction == 3:
            # left
            if base_point == 0:
                col_end = self.latest_position[0] - 1
                col_start = col_end - width + 1
                row_start = self.latest_position[1]
                row_end = row_start + height - 1
                base_col, base_row = col_end, row_start
            else:
                col_end = self.latest_position[0] - 1
                col_start = col_end - width + 1
                row_end = self.latest_position[3]
                row_start = row_end - height + 1
                base_col, base_row = col_end, row_end
        
        # Check for valid rectangle
        is_valid, col_start, row_start, col_end, row_end = self._is_valid_rectangle(
                        direction, base_point, base_col, base_row, col_start, row_start, col_end, row_end)
        
        reward = -self.constant
        if is_valid:
            self.rect_id += 1
            self.grid[row_start:row_end+1, col_start:col_end+1] = self.rect_id
            self.rect_list.append([col_start, row_start, col_end, row_end])
            self.latest_position = [col_start, row_start, col_end, row_end]
            
            # check if compute is valid, if not then termimated is true
            if cur_compute > self.left_compute:
                cur_compute = self.left_compute
                self.compute_list.append(cur_compute)
                self.left_compute = 0
                reward = self.reward()
                return self.grid, reward, True, False, {}
                
            # Check if grid is full, if true then compute is the compute left
            elif np.sum(self.grid == 0) == 0:
                cur_compute = self.left_compute
                self.compute_list.append(cur_compute)
                self.left_compute = 0
                reward = self.reward()
                return self.grid, reward, True, False, {}
            else:
                # internal state, reward is none
                reward = 0
                self.compute_list.append(cur_compute)
                self.left_compute = self.left_compute - cur_compute
                return self.grid, reward, False, False, {}
                
        else:
            # invalid action
            """
            # option 1: terminated is true, and fix the mapping, compute reward
            
            if len(self.compute_list) > 0:
                self.compute_list[-1] += self.left_compute
                reward = self.reward()
            else:
                reward = -self.constant/2
            return self.grid, reward, True, False, {}
            """
            
            # option 2: continue the game, and the reward is negative
            reward = -self.constant/2
            return self.grid, reward, False, False, {}

    def _is_valid_rectangle(self, direction, base_point, base_col, base_row, col_start, row_start, col_end, row_end):
        rect_action = [col_start, row_start, col_end, row_end]
        rect_grid = [0, 0, self.cols-1, self.rows-1]
        # process the boundary
        area, new_left, new_top, new_right, new_bottom = self.common_area(rect_action, rect_grid)
        if area == -1:
            return False, new_left, new_top, new_right, new_bottom
        rect_new = [new_left, new_top, new_right, new_bottom]
        # import pdb; pdb.set_trace()
        for rect in self.rect_list:
            valid, new_left, new_top, new_right, new_bottom = self.bypass(direction, base_point, base_col, base_row, rect_new, rect)
            if not valid:
                return False, new_left, new_top, new_right, new_bottom
            else:
                rect_new = [new_left, new_top, new_right, new_bottom]
        
        return True, new_left, new_top, new_right, new_bottom
    
    def common_area(self, rect1, rect2):
        # rect1: left, top, right, bottom
        # rect2: left, top, right, bottom
        left_1, top_1, right_1, bottom_1 = rect1
        left_2, top_2, right_2, bottom_2 = rect2
        left = max(left_1, left_2)
        right = min(right_1, right_2)
        top = max(top_1, top_2)
        bottom = min(bottom_1, bottom_2)
        if left > right or top > bottom:
            area = -1
        else:
            area = (right - left + 1) * (bottom - top + 1)
        return area, left, top, right, bottom
    
    def bypass(self, direction, base_point, base_col, base_row, rect_new, rect):
        # avoid the overlap of the new rect and the old rect, must contain the point (base_col, base_row)
        # the new rect also must be contained in the rect_new
        # rect_new: left, top, right, bottom
        # rect: left, top, right, bottom
        # direction: 0, 1, 2, 3 up right down left
        left_new, top_new, right_new, bottom_new = rect_new
        left, top, right, bottom = rect
        common_area, _, _, _, _ = self.common_area(rect_new, rect)
        if common_area == -1:
            # no overlap, return the rect_new
            return True, left_new, top_new, right_new, bottom_new
        # overlap exists, bypass the overlap region
        if direction == 0:
            if base_point == 0:
                assert base_col == left_new, base_row == bottom_new
                left_bypass = left_new
                top_bypass = max(top_new, bottom+1)
                right_bypass = right_new
                bottom_bypass = bottom_new
            else:
                assert base_col == right_new, base_row == bottom_new
                left_bypass = left_new
                top_bypass = max(top_new, bottom+1)
                right_bypass = right_new
                bottom_bypass = bottom_new
        elif direction == 1:
            if base_point == 0:
                assert base_col == left_new, base_row == top_new
                left_bypass = left_new
                top_bypass = top_new
                right_bypass = min(right_new, left-1)
                bottom_bypass = bottom_new
            else:
                assert base_col == left_new, base_row == bottom_new
                left_bypass = left_new
                top_bypass = top_new
                right_bypass = min(right_new, left-1)
                bottom_bypass = bottom_new
        elif direction == 2:
            if base_point == 0:
                assert base_col == left_new, base_row == top_new
                left_bypass = left_new
                top_bypass = top_new
                right_bypass = right_new
                bottom_bypass = min(bottom_new, top-1)
            else:
                assert base_col == right_new, base_row == top_new
                left_bypass = left_new
                top_bypass = top_new
                right_bypass = right_new
                bottom_bypass = min(bottom_new, top-1)
                
        elif direction == 3:
            if base_point == 0:
                assert base_col == right_new, base_row == top_new
                left_bypass = max(left_new, right+1)
                top_bypass = top_new
                right_bypass = right_new
                bottom_bypass = bottom_new
            else:
                assert base_col == right_new, base_row == bottom_new
                left_bypass = max(left_new, right+1)
                top_bypass = top_new
                right_bypass = right_new
                bottom_bypass = bottom_new
        if left_bypass > right_bypass or top_bypass > bottom_bypass:
            return False, left_bypass, top_bypass, right_bypass, bottom_bypass
        else:
            return True, left_bypass, top_bypass, right_bypass, bottom_bypass


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
    env = MappingSimpleMaskALLEnv()
    
    best_reward = -1000
    reward_list = []
    while len(reward_list) < 1:
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
