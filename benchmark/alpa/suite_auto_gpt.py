"""Benchmark suites for gpt with auto parallelization."""
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # add to sys.path.

from suite_manual_gpt import *
from benchmark_parallel_utils import (BenchmarkCase, SearchParallelArgs,
                                      LoadSolutionParallelArgs, ConfigParallelArgs)
from alpa import ManualStageOption, WSCManualStageOption

# NOTE: match for wsc 5 size, micro batch = 100, global bactch = 1000
# max_global_batch_size = 1000
# NOTE: normal is 1024
max_global_batch_size = 1024
# NOTE: For auto search option
auto_stage_option = {
    "submesh_physical_shape_space": "small_power_of_two",
    "submesh_logical_shape_space": "all",
    "stage_imbalance_tolerance": 1.0,
    "use_hlo_cost_model": True,
    "profiling_database_filename": "prof_database.pkl",
}

prefer_reduce_scatter = True
use_remat = True

# get auto search Benchmark Case -- BenchmarkCase parallel args is search
def get_search_cases(model_spec, num_micro_batches_list, num_auto_layers_list):
    return [
        BenchmarkCase(
            max_global_batch_size, model_spec, num_micro_batches, "search",
            SearchParallelArgs(prefer_reduce_scatter, use_remat,
                               num_auto_layers, auto_stage_option))
        for num_micro_batches in num_micro_batches_list
        for num_auto_layers in num_auto_layers_list
    ]


def get_config_cases(model_spec, num_micro_batches_list, input_placement_specs_pkl, stage_option):
    import pickle
    with open(input_placement_specs_pkl, 'rb') as f:
        input_placement_specs = pickle.load(f)
    stage_num = len(stage_option.forward_stage_layer_ids)

    return [
        BenchmarkCase(
            max_global_batch_size, model_spec, num_micro_batches, "config",
            ConfigParallelArgs(stage_num, input_placement_specs, None,'1f1b', stage_option, use_remat))
        for num_micro_batches in num_micro_batches_list
    ]

def get_config_cases_idx(model_specs, num_micro_batches_list, partition_index, stage_option):
    stage_num = len(stage_option.forward_stage_layer_ids)
    return [
        BenchmarkCase(
            max_global_batch_size, model_spec, num_micro_batches, "config",
            ConfigParallelArgs(stage_num, None, partition_index,'1f1b', stage_option, use_remat))
        for num_micro_batches in num_micro_batches_list
        for model_spec in model_specs
    ]


def get_solution_case(model_spec, num_micro_batches, num_auto_layers,
                      forward_stage_layer_ids, submesh_physical_shapes,
                      submesh_logical_shapes,
                      submesh_autosharding_option_dicts):
    return [
        BenchmarkCase(
            max_global_batch_size, model_spec, num_micro_batches,
            "load_solution",
            LoadSolutionParallelArgs(prefer_reduce_scatter, use_remat,
                                     num_auto_layers, forward_stage_layer_ids,
                                     submesh_physical_shapes,
                                     submesh_logical_shapes,
                                     submesh_autosharding_option_dicts))
    ]


force_dp_dict = {"force_batch_dim_to_mesh_dim": 0}


# NOTE: research how to construct different suite
wsc_config_test_suite = { 
        # tx8
    20: get_config_cases_idx(gpt_wsc_specs.values(), [128],
                        partition_index="uniform",
                        stage_option=WSCManualStageOption(forward_stage_layer_ids=[[0]],
                                                          submeshes=[[0, 0, 4, 3]],
                                                          submesh_physical_shapes=None,
                                                          submesh_logical_shapes=None,
                                                          submesh_autosharding_option_dicts=[{}])),
    25: get_config_cases_idx(gpt_wsc_specs.values(), [128],
                        # partition_index="uniform",
                        partition_index=[0.013333333333333334, 0.08, 0.10666666666666667, 0.2, 0.32, 0.41333333333333333, 0.52, 0.5733333333333334, 0.6933333333333334, 0.76, 0.88, 0.9733333333333334],
                        stage_option=WSCManualStageOption(forward_stage_layer_ids=[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]],
                                                          submeshes=[[0, 0, 0, 2], [0, 3, 0, 5], [0, 6, 0, 7], [0, 8, 0, 10], [0, 11, 0, 11], [0, 12, 0, 13], [0, 14, 0, 14], [0, 15, 0, 15], [0, 16, 0, 16], [0, 17, 0, 17], [0, 18, 0, 19], [0, 20, 0, 21], [0, 22, 0, 24]],
        submesh_physical_shapes=None,
        submesh_logical_shapes=None,
        submesh_autosharding_option_dicts=[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}])
    ),

    40: get_config_cases_idx(gpt_wsc_specs.values(), [128],
                        partition_index="uniform",
                        stage_option=WSCManualStageOption(forward_stage_layer_ids=[[0]],
                                                          submeshes=[[0, 0, 4, 4]],
                                                          submesh_physical_shapes=None,
                                                          submesh_logical_shapes=None,
                                                          submesh_autosharding_option_dicts=[{}])
    ),

    # 2: get_config_cases_idx(gpt_specs.values(), [128],
    #                     # partition_index="uniform",
    #                     # partition_index=[0, 1210, 2419],
    #                     # partition_index=[0, 1210],
    #                     partition_index = [1210],
    #                     # partition_index = [800],
    #                     stage_option=WSCManualStageOption(forward_stage_layer_ids=[[0], [1]],
    #                                                       submeshes=[
    #                         [0, 0, 0, 0],
    #                         [0, 1, 0, 1]
    #                     ],
    #     submesh_physical_shapes=None,
    #     submesh_logical_shapes=None,
    #     submesh_autosharding_option_dicts=[{}, {}])
    # ),   
    # 4: get_config_cases_idx(gpt_specs.values(), [128],
    #                     partition_index=[0.0, 0.7142857142857143],
    #                     stage_option=WSCManualStageOption(forward_stage_layer_ids=[[0], [1]], 
    #                                                       submeshes=[[0, 0, 0, 2], [0, 3, 0, 3]],
    #                                                     submesh_physical_shapes=None,
    #                                                     submesh_logical_shapes=None,
    #                                                     submesh_autosharding_option_dicts=[{}, {}])
    # ),  
    # 8: get_config_cases_idx(gpt_specs.values(), [128],
    #                     # partition_index="uniform",
    #                     partition_index=[0, 1000, 2000, 3203],
    #                     stage_option=WSCManualStageOption(forward_stage_layer_ids=[[0], [1], [2]],
    #                                                       submeshes=[
    #                         [0, 0, 0, 1],
    #                         [1, 0, 1, 1],
    #                         [2, 0, 3, 1],
    #                     ],
    #     submesh_physical_shapes=None,
    #     submesh_logical_shapes=None,
    #     submesh_autosharding_option_dicts=[{}, {}, {}])
    # ),
    # 16: get_config_cases_idx(gpt_specs.values(), [128],
    #                     partition_index="uniform",
    #                     stage_option=WSCManualStageOption(forward_stage_layer_ids=[[0]],
    #                                                       submeshes=[[0, 0, 3, 3]],
    #                                                       submesh_physical_shapes=None,
    #                                                       submesh_logical_shapes=None,
    #                                                       submesh_autosharding_option_dicts=[{}])
    # ),



}


# Temporary debug suite
tmp_suite = {}

model_type_size_dict = {
    "gpt":{
        1: "350M",
        2: "760M",
        4: "1.3B",
        8: "2.6B",
        16: "6.7B",
        32: "15B",
        64: "39B"
    },
    "moe":{
        1: "380M",
        2: "690M",
        4: "1.3B",
        8: "2.4B",
        16: "10B",
        32: "27B",
        64: "70B"
    },
    "wresnet":{
        1: "250M",
        2: "500M",
        4: "1B",
        8: "2B",
        16: "4B",
        32: "6.8B",
        64: "13B"
    }
}
# Performance test with search solutions found for p3.16xlarge
perf_test_suite = {
    1:
        get_solution_case(gpt_specs["350M"], 512, 1, [[0]], [(1, 1)], [(1, 1)],
                          [{}]),
    2:
        get_solution_case(gpt_specs["760M"], 128, 6, [[0, 1, 2], [3, 4, 5]],
                          [(1, 1)] * 2, [(1, 1)] * 2, [force_dp_dict] * 2),
    4:
        get_solution_case(gpt_specs["1.3B"], 128, 6, [[0, 1, 2], [3, 4, 5]],
                          [(1, 2)] * 2, [(2, 1)] * 2, [force_dp_dict] * 2),
    8:
        get_solution_case(gpt_specs["2.6B"], 128,
                          8, [[0, 1], [2, 3], [4, 5, 6, 7]], [(1, 2), (1, 2),
                                                              (1, 4)], [(2, 1),
                                                                        (2, 1),
                                                                        (4, 1)],
                          [force_dp_dict, {}, {}]),
    16:
        get_solution_case(gpt_specs["6.7B"], 64, 8,
                          [[0, 1, 2, 3], [4, 5, 6, 7]], [(1, 8)] * 2,
                          [(2, 4)] * 2, [force_dp_dict] * 2),
    32:
        get_solution_case(
            gpt_specs["15B"], 128, 16,
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            [(1, 8)] * 4, [(2, 4)] * 4, [force_dp_dict] * 4),
    64:
        get_solution_case(gpt_specs["39B"], 1024,
                          16, [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9],
                               [10], [11], [12], [13], [14], [15]],
                          [(1, 4)] * 16, [(1, 4)] * 16, [force_dp_dict] * 16),
}

# Grid search on hyperparameters
# grid_search_suite = {
#     2: (get_search_cases(gpt_specs["760M"], [32, 64, 128, 256], [6]) +
#         get_search_cases(gpt_specs["760M"], [32, 64], [12])),
#     4: (get_search_cases(gpt_specs["1.3B"], [32, 64, 128], [6]) +
#         get_search_cases(gpt_specs["1.3B"], [32, 64], [12])),
#     8: (get_search_cases(gpt_specs["2.6B"], [64, 128, 256], [8]) +
#         get_search_cases(gpt_specs["2.6B"], [64, 128], [16])),
#     16: get_search_cases(gpt_specs["6.7B"], [32, 64, 128, 256], [8]),
#     32: get_search_cases(gpt_specs["15B"], [64, 128, 256, 512], [16]),
#     64: get_search_cases(gpt_specs["39B"], [128, 256, 512, 1024], [8]),
# }
# suite: key represent num of GPUs
grid_search_suite = {
    1: get_search_cases(gpt_specs["350M"], [512], [1]),
    2: get_search_cases(gpt_specs["760M"], [128], [6]),
    4: get_search_cases(gpt_specs["1.3B"], [128], [6]),
    8: get_search_cases(gpt_specs["2.6B"], [128], [8]),
    # 16: get_search_cases(gpt_specs["6.7B"], [64], [8]),
    16: get_search_cases(gpt_specs["6.7B"], [128], [8]),
    # 16: get_search_cases(gpt_specs["15B"], [128], [8]),
    32: get_search_cases(gpt_specs["15B"], [128], [16]),
    64: get_search_cases(gpt_specs["39B"], [1024], [16]),
    1024: get_search_cases(gpt_specs["39B"], [32], [64]),    
}

# Small test cases for correctness test
correctness_test_suite = {
    8: get_search_cases(gpt_specs["2.6B"], [128], [8]),
}

config_test_suite = {
    # 2: get_config_cases(gpt_specs["760M"], [128], 
    #                     'tmp_a100_gpu_real/gpt.grid_search_auto-2X1-actualA100-2023-03-01-02-57-12/Batchsize_1024-num_b_128-auto_layers_6/input_placement_specs.pkl', 
    #                     stage_option=ManualStageOption(forward_stage_layer_ids=[[0], [1]], 
    #                                                    submesh_physical_shapes=[[1, 1], [1, 1]], 
    #                                                    submesh_logical_shapes=[[1, 1], [1, 1]], 
    #                                                    submesh_autosharding_option_dicts=[{}, {}])
    #                     ),
    2: get_config_cases_idx(gpt_specs["760M"], [128], 
                        # partition_index="uniform", 
                        # partition_index=[0, 1210, 2419],
                        partition_index=[0, 1210],
                        # partition_index = [1210],
                        stage_option=ManualStageOption(forward_stage_layer_ids=[[0], [1]], 
                                                       submesh_physical_shapes=[[1, 1], [1, 1]], 
                                                       submesh_logical_shapes=[[1, 1], [1, 1]], 
                                                       submesh_autosharding_option_dicts=[{}, {}])
                        ),
    # 8: get_config_cases(gpt_specs["2.6B"], [128], 
    #                     'tmp_wsc_perf_15GB_fp16/gpt.grid_search_auto-8X1-perf@gpu-2023-03-07-09-02-58/Batchsize_1024-num_b_128-auto_layers_8/input_placement_specs.pkl', 
    #                     stage_option=ManualStageOption(forward_stage_layer_ids=[[0], [1], [2]], 
    #                                                    submesh_physical_shapes=[[1, 2], [1, 2], [1, 4]], 
    #                                                    submesh_logical_shapes=[[2, 1], [2, 1], [4, 1]], 
    #                                                    submesh_autosharding_option_dicts=[{}, {}, {}])
    #                     ),
    8: get_config_cases_idx(gpt_specs["2.6B"], [128], 
                        # partition_index="uniform",
                        partition_index=[0, 1000, 2000, 3203],
                        stage_option=ManualStageOption(forward_stage_layer_ids=[[0], [1], [2]], 
                                                       submesh_physical_shapes=[[1, 2], [1, 2], [1, 4]], 
                                                       submesh_logical_shapes=[[2, 1], [2, 1], [4, 1]], 
                                                       submesh_autosharding_option_dicts=[{}, {}, {}])
                        )
}

# 'tmp_wsc_perf_15GB_fp16/gpt.grid_search_auto-8X1-perf@gpu-2023-03-07-09-02-58/Batchsize_1024-num_b_128-auto_layers_8/input_placement_specs.pkl'
# 'tmp_a100_gpu_onlymapping/gpt.grid_search_auto-8X1-costmodel-2023-03-13-10-05-53/Batchsize_1024-num_b_128-auto_layers_8/input_placement_specs.pkl'