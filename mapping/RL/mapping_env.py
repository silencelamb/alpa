import numpy as np
from collections import deque
import statistics
import gymnasium as gym
from gymnasium import spaces
import sys
sys.path.insert(0, "/code/alpa/benchmark/alpa/")
from benchmark_one_case import benchmark_one_case
from benchmark_parallel_utils import BenchmarkCase
from alpa.global_env import get_global_config, set_global_config, get_collective_cost_dict
from suite_manual_gpt import gpt_specs
from alpa import ManualStageOption, WSCManualStageOption
from suite_auto_gpt import get_config_cases_idx, max_global_batch_size
from alpa.util import to_str_round, GB
from benchmark_parallel_utils import BenchmarkCase, ConfigParallelArgs

class WSCMappingEnv(gym.Env):
    
    def __init__(self, render_mode='human', use_image=False):
        super(WSCMappingEnv, self).__init__()
        
        # set global env
        global_env_new = get_global_config()
        global_env_new.hardware = 'wsc'
        global_env_new.only_mapping = True
        global_env_new.use_analytical_perf_model = True
        get_collective_cost_dict()

        set_global_config(global_env_new)
        global_env = get_global_config()
        
        # total compute in a microbatch
        self.compute_of_a_microbatch= 100
        self.left_compute = self.compute_of_a_microbatch
        self.num_microbatch = 1024
        self.rows, self.cols = (5, 4)
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
        model_type = "gpt"
        total_compute = sum(self.compute_list)
        normalized_compute_list = [compute / total_compute for compute in self.compute_list]
        partition_index = []
        cur_sum = 0.0
        for x in normalized_compute_list:
            partition_index.append(cur_sum)
            cur_sum = cur_sum + x
        global max_global_batch_size
        max_global_batch_size = 1000
        suite = get_config_cases_idx(gpt_specs["350M"], [100],
                        # partition_index="uniform",
                        partition_index=partition_index,
                        stage_option=WSCManualStageOption(forward_stage_layer_ids=[[x] for x in range(len(self.rect_list)) ],
                                                          submeshes=self.rect_list,
                                                          submesh_physical_shapes=None,
                                                          submesh_logical_shapes=None,
                                                          submesh_autosharding_option_dicts=[{} for x in range(len(self.rect_list))])
                )

        # Run all cases
        for benchmark_case in suite:
            benchmark_case: BenchmarkCase
            print(benchmark_case.batch_size)
            model_config = benchmark_case.model_config
            num_micro_batches = benchmark_case.num_micro_batches
            try:
                auto_layers = benchmark_case.parallel_args.num_auto_layers
            except AttributeError:
                auto_layers = 'auto'

            parallel_args = benchmark_case.parallel_args

            # Run one case
            print("Working on case: {}".format(str(benchmark_case)))
            result = benchmark_one_case(model_type,
                                    benchmark_case,
                                    niter=3,
                                    num_hosts=self.rows,
                                    num_devices_per_host=self.cols,
                                    shard_only=False,
                                    local=False,
                                    profile_driver_time=False,
                                    disable_tqdm=True,
                                    use_separate_process=False)
            (parameter_count, peak_mem, latencies, tflops, metadata) = result  
            heads = [
                "Type", "Model Config", "#Microbatch", "#GPU", "Parallel Config",
                "Mean Time (s)", "Std Time (s)", "#Params (Billion)", "TFLOPs",
                "Peak Mem (GB)", "Metadata"
            ]
            if isinstance(parallel_args, ConfigParallelArgs):
                parallel_args = parallel_args._replace(input_placement_specs=[])
                
            values = [
                model_type, model_config, num_micro_batches, f"{self.rows}x{self.cols}",
                parallel_args, f"{np.mean(latencies):.3f}",
                f"{np.std(latencies):.3f}", f"{parameter_count/1e9:.3f}B",
                f"{tflops:.2f}", f"{peak_mem/GB:.3f}",
                to_str_round(metadata, 6)
            ]
            values = [str(x) for x in values]
            result_dict = dict(zip(heads, values)) 
            print('One result: ' + str(result_dict))
            total_latency = metadata['estimated_total_time']
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
            if cur_compute > self.left_compute:
                cur_compute = self.left_compute
                self.compute_list.append(cur_compute)
                self.left_compute = 0
                reward = self.reward()
                done = True
                
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
            
            # print(f"Step {self.current_step}: invalid action, {col_start, row_start, col_end, row_end}, {self.latest_position}")
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
        action_mask = np.zeros(self.action_space.nvec[1], dtype=np.int8)
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
        print(self.compute_list)
        print(self.rect_list)
        print("env rending end=========================")
        
    def close(self):
        pass


if __name__ == '__main__':
    env = WSCMappingEnv()
    
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
