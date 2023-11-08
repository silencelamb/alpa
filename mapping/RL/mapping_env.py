import numpy as np
from collections import deque
import random
import statistics
from collections import namedtuple
import gymnasium as gym
from gymnasium import spaces
import torch
from torch_geometric.data import Data
import sys
sys.path.insert(0, "/code/alpa/benchmark/alpa/")
from benchmark_one_case import benchmark_one_case
from benchmark_parallel_utils import BenchmarkCase
from alpa.global_env import get_global_config, set_global_config, get_collective_cost_dict
from suite_manual_gpt import gpt_specs
from alpa import ManualStageOption, WSCManualStageOption
from suite_auto_gpt import get_one_config_case_idx, max_global_batch_size
from alpa.util import to_str_round, GB
from benchmark_parallel_utils import BenchmarkCase, ConfigParallelArgs

GraphData = namedtuple('GraphData', ['model_type', 'model_size', 'graph_data'])

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
        
        self.rows, self.cols = (5, 4)
        self.grid = np.zeros((self.rows, self.cols), int)
        self.render_mode = render_mode
        
        self.use_image = use_image
        # Observation space
        if self.use_image:
            self.observation_space = spaces.Box(low=0, high=self.rows * self.cols, shape=(1, self.rows, self.cols), dtype=np.int8)
        else:
            self.observation_space = spaces.Box(low=0, high=self.rows * self.cols, shape=(self.rows, self.cols), dtype=int)
                    
        self.action_space = spaces.MultiDiscrete([10,   # ratio of ave compute, 
                                                  self.cols * self.rows * self.cols * self.rows  # rect action
                                                  ])
        # latest position of the rectangle, [left, top, right, bottom]
        self.latest_position = []
        
        self.current_step = 0
        self.max_steps = 25
        self.compute_list = []
        self.rect_list = []
        self.rect_id = 0
        self.action_mask = None
        
        # 构建GraphDataSet
        self.GraphDataSet = self.construct_graph_dataset()
        self.model_type = None
        self.model_size = None
        
        # for reward statistics
        self.total_reward = 0
        self.max_ave_episode = 50
        self.reward_record = deque(maxlen=self.max_ave_episode)
        self.average_reward = 0
        self.mae_reward = 0
        self.mae_param = 0.9
        self.max_ave_episode = 0
        
    def construct_graph_dataset(self):
        # load graph data, temporarily use mock data
        node_features, edge_index = self.gen_mock_graph_data()
        # 创建图数据
        data = Data(x=node_features, edge_index=edge_index)
        GraphDataSet = [
            GraphData("gpt", "350M", data),
            GraphData("gpt", "760M", data),
            GraphData("gpt", "1.3B", data),
            GraphData("gpt", "2.6B", data),
            GraphData("gpt", "6.7B", data),
            GraphData("gpt", "15B", data),
            GraphData("bert", "Tiny", data),
            GraphData("bert", "Small", data),
            GraphData("bert", "Medium", data),
            GraphData("bert", "Base", data),
            GraphData("bert", "Large", data),
            GraphData("wresnet", "25.56M", data),
            GraphData("wresnet", "44.55M", data),
            GraphData("wresnet", "60.19M", data),
            GraphData("wresnet", "68.88M", data),
            GraphData("wresnet", "126.88M", data)
        ]
        return GraphDataSet
    
    def reward(self):
        """Reward function
        """
        # return -np.random.randint(1, 1000)*1.0     # mock reward, for fast debug
        # common code
        total_compute = sum(self.compute_list)
        normalized_compute_list = [compute / total_compute for compute in self.compute_list]
        partition_index = []
        cur_sum = 0.0
        for x in normalized_compute_list:
            partition_index.append(cur_sum)
            cur_sum = cur_sum + x
        stage_option = WSCManualStageOption(
            forward_stage_layer_ids=[[x] for x in range(len(self.rect_list)) ],
            submeshes=self.rect_list,
            submesh_physical_shapes=None,
            submesh_logical_shapes=None,
            submesh_autosharding_option_dicts=[{} for x in range(len(self.rect_list))]
        )
        model_type = self.model_type
        if model_type == 'gpt':
            suite = get_one_config_case_idx(
                gpt_specs[self.model_size], 
                [100],
                partition_index=partition_index,
                stage_option=stage_option       
            )
        elif model_type == "bert":
            # TODO: add bert specs
            pass
        elif model_type == "wresnet":
            # TODO: add wresnet specs
            pass

        # Run all cases
        for benchmark_case in suite:
            benchmark_case: BenchmarkCase
            print(benchmark_case.batch_size)
            print(self.grid)
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
                "Mean Time (s)", "Std Time (s)", "#Params (Billion)", "Actual TFLOPs(Per Device)",
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
        cur_compute, rect_action  = action
        cur_compute += 1
        col_start, row_start, col_end, row_end  = self.decode_action(rect_action)
        
        # Check for valid rectangle
        is_valid = self._is_valid_rectangle(col_start, row_start, col_end, row_end)
        
        truncted = False
        done = False
        if is_valid:
            self.rect_id += 1
            self.grid[row_start:row_end+1, col_start:col_end+1] = self.rect_id
            self.rect_list.append([col_start, row_start, col_end, row_end])
            self.latest_position = [col_start, row_start, col_end, row_end]
            self.compute_list.append(cur_compute)
                
            # Check if grid is full, if true then compute is the compute left
            if np.sum(self.grid == 0) == 0:
                reward = self.reward()
                done = True
            else:
                # internal state, reward is none
                reward = 0
                
        else:
            # invalid action

            # option 1: terminated is true, and fix the mapping, compute reward
            penalty = np.sum(self.grid == 0) / (self.cols * self.rows)
            self.compute_list.append(cur_compute)
            reward = self.reward()
            reward = reward * (1+penalty)
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

    @property
    def num_graphs(self):
        return len(self.GraphDataSet)

    def get_graph_data_by_index(self, index=0):
        return self.GraphDataSet[index].graph_data
    
    def get_graph_data_by_type_size(self, model_type, model_size):
        for graph_data in self.GraphDataSet:
            if graph_data.model_type == model_type and graph_data.model_size == model_size:
                return graph_data.graph_data
    
    def set_model_type_size(self, model_type, model_size):
        self.model_type = model_type
        self.model_size = model_size
    
    def get_graph_feature_dim(self):
        return self.GraphDataSet[0].graph_data.x.shape[1]
    
    def shuffle_graphs(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.GraphDataSet)
           
    def gen_mock_graph_data(self):
        # 假设我们有5个节点，每个节点有3个特征
        num_nodes = 1000
        num_node_features = 128

        # 创建一个有随机特征的节点特征矩阵
        node_features = torch.randn(num_nodes, num_node_features)

        # 假设图是无向的，并且我们随机创建了10条边
        num_edges = 500

        # 对于无向图，每一条边都需要两个方向，因此实际上有num_edges * 2个边的索引
        edge_indices = torch.randint(0, num_nodes, (2, num_edges * 2), dtype=torch.long)

        # 由于我们的图是无向的，我们需要确保对于每一条边(u, v)，都存在相对的边(v, u)
        # 下面的代码将创建一个镜像边缘索引并将其与原边缘索引连接
        edge_indices = torch.cat([edge_indices, edge_indices.flip([0])], dim=1)

        # 除去自循环和重复的边
        edge_indices = edge_indices[:, edge_indices[0] != edge_indices[1]]  # 去掉自环
        edge_indices = torch.unique(edge_indices, dim=1)  # 去掉重复的边
        return node_features, edge_indices
    
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
