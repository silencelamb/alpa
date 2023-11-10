from suite_manual_gpt import GPTModelConfig
from collections import namedtuple
from benchmark_parallel_utils import (BenchmarkCase, SearchParallelArgs,
                                      LoadSolutionParallelArgs, ConfigParallelArgs,
                                      UniformParallelArgs)

# NOTE: only for Wafer-Scale GPU
gpt_models = {
    #                      S，   H,   L,  head,   V,
    "125M": GPTModelConfig(1024, 768, 12, 12, 51200),
    "350M": GPTModelConfig(1024, 1024, 24, 16, 51200),
    "760M": GPTModelConfig(1024, 1536, 24, 16, 51200),
    "1.3B": GPTModelConfig(1024, 2048, 24, 32, 51200),
    "2.6B": GPTModelConfig(1024, 2560, 32, 32, 51200),
    "6.7B": GPTModelConfig(1024, 4096, 32, 32, 51200),
    "15B": GPTModelConfig(1024, 5120, 48, 40, 51200),
    "39B": GPTModelConfig(1024, 8192, 48, 64, 51200),
    "76B": GPTModelConfig(1024, 10240, 60, 80, 51200),
}


gpt_params = {
    # GPT models
    tuple(GPTModelConfig(1024, 768, 12, 12, 51200)): (12, 64),
    tuple(GPTModelConfig(1024, 1024, 24, 16, 51200)): (12, 64),  # (4, 64)
    tuple(GPTModelConfig(1024, 1536, 24, 16, 51200)): (12, 64),
    tuple(GPTModelConfig(1024, 2048, 24, 32, 51200)): (24, 64),
    tuple(GPTModelConfig(1024, 2560, 32, 32, 51200)): (24, 64),
    tuple(GPTModelConfig(1024, 4096, 32, 32, 51200)): (24, 64),
    tuple(GPTModelConfig(1024, 5120, 48, 40, 51200)): (24, 64),
    tuple(GPTModelConfig(1024, 8192, 48, 64, 51200)): (24, 64),
    tuple(GPTModelConfig(1024, 10240, 60, 80, 51200)): (24, 64),
}

# NOTE: only for DOJO
gpt_dojo_models = {

    "125M": GPTModelConfig(1024, 768, 12, 12, 51200),
    "350M": GPTModelConfig(1024, 1024, 24, 16, 51200),
    "760M": GPTModelConfig(1024, 1536, 24, 16, 51200),
    "1.3B": GPTModelConfig(1024, 2048, 24, 32, 51200),
    "2.6B": GPTModelConfig(1024, 2560, 32, 32, 51200),
    "6.7B": GPTModelConfig(1024, 4096, 32, 32, 51200),
    "15B": GPTModelConfig(1024, 5120, 48, 40, 51200),
    "39B": GPTModelConfig(1024, 8192, 48, 64, 51200),
    "76B": GPTModelConfig(1024, 10240, 60, 80, 51200),
}

gpt_dojo_params = {

    tuple(GPTModelConfig(1024, 768, 12, 12, 51200)): (10, 40),
    tuple(GPTModelConfig(1024, 1024, 24, 16, 51200)): (10, 40),  # (4, 64)
    tuple(GPTModelConfig(1024, 1536, 24, 16, 51200)): (10, 40),
    tuple(GPTModelConfig(1024, 2048, 24, 32, 51200)): (20, 40),
    tuple(GPTModelConfig(1024, 2560, 32, 32, 51200)): (20, 40),
    tuple(GPTModelConfig(1024, 4096, 32, 32, 51200)): (20, 40),
    tuple(GPTModelConfig(1024, 5120, 48, 40, 51200)): (20, 40),
    tuple(GPTModelConfig(1024, 8192, 48, 64, 51200)): (20, 40),
    tuple(GPTModelConfig(1024, 10240, 60, 80, 51200)): (20, 40),
}


prefer_reduce_scatter = True
use_remat = True
force_dp_dict = {"force_batch_dim_to_mesh_dim": 0}


def get_solution_cases(model_spec, num_micro_batches, num_auto_layers,
                       forward_stage_layer_ids, submesh_physical_shapes,
                       submesh_logical_shapes,
                       submesh_autosharding_option_dicts, batch_size):
    return [
        BenchmarkCase(
            batch_size, model_spec, num_micro_batches,
            "load_solution",
            LoadSolutionParallelArgs(prefer_reduce_scatter, use_remat,
                                     num_auto_layers, forward_stage_layer_ids,
                                     submesh_physical_shapes,
                                     submesh_logical_shapes,
                                     submesh_autosharding_option_dicts))
    ]


def flatten_list(nested_list):
    return [v for sublist in nested_list for item in sublist for v in item]


def get_list(layers, size):
    # 定义原始列表
    original_list = list(range(layers))

    # 定义切分大小
    split_size = size

    # 使用列表切片进行切分
    split_list = [original_list[i:i+split_size]
                  for i in range(0, layers, split_size)]
    return split_list


wsc_perf_suite = {
    #             [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]],
    25:  flatten_list([
        # dp=25, tp=1, pp=1
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=1000,
                               model_spec=mod, num_micro_batches=params[1],
                               num_auto_layers=params[0], forward_stage_layer_ids=get_list(
                                   int(params[0]), int(params[0]/pp)),
                               submesh_physical_shapes=[(5, 5)]*1, submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp_dict] * pp)
            for mod, params in zip(gpt_dojo_models.values(), gpt_dojo_params.values())
            for dp, tp, pp in zip([25], [1], [1])
        ],


        # # dp=1, tp=25, pp=1
        # [
        #     # NOTE: fit for small models with layer=12
        #     get_solution_cases(batch_size=1000,
        #                        model_spec=mod, num_micro_batches=params[1],
        #                        num_auto_layers=params[0], forward_stage_layer_ids=get_list(
        #                            int(params[0]), int(params[0]/pp)),
        #                        submesh_physical_shapes=[(5, 5)]*1, submesh_logical_shapes=[(dp, tp)]*pp,
        #                        submesh_autosharding_option_dicts=[force_dp_dict] * pp)
        #     for mod, params in zip(gpt_dojo_models.values(), gpt_dojo_params.values())
        #     for dp, tp, pp in zip([1], [25], [1])
        # ],


        # # dp=5, tp=5, pp=1
        # [
        #     # NOTE: fit for small models with layer=12
        #     get_solution_cases(batch_size=1000,
        #                        model_spec=mod, num_micro_batches=params[1],
        #                        num_auto_layers=params[0], forward_stage_layer_ids=get_list(
        #                            int(params[0]), int(params[0]/pp)),
        #                        submesh_physical_shapes=[(5, 5)]*1, submesh_logical_shapes=[(dp, tp)]*pp,
        #                        submesh_autosharding_option_dicts=[force_dp_dict] * pp)
        #     for mod, params in zip(gpt_dojo_models.values(), gpt_dojo_params.values())
        #     for dp, tp, pp in zip([5], [5], [1])
        # ],

        # # dp=5, tp=1, pp=5
        # [
        #     # NOTE: fit for small models with layer=12
        #     get_solution_cases(batch_size=1000,
        #                        model_spec=mod, num_micro_batches=params[1],
        #                        num_auto_layers=params[0], forward_stage_layer_ids=get_list(
        #                            int(params[0]), int(params[0]/pp)),
        #                        submesh_physical_shapes=[(5, 1)]*5, submesh_logical_shapes=[(dp, tp)]*pp,
        #                        submesh_autosharding_option_dicts=[force_dp_dict] * pp)
        #     for mod, params in zip(gpt_dojo_models.values(), gpt_dojo_params.values())
        #     for dp, tp, pp in zip([5], [1], [5])
        # ],


        # # dp=1, tp=5, pp=5
        # [
        #     # NOTE: fit for small models with layer=12
        #     get_solution_cases(batch_size=1000,
        #                        model_spec=mod, num_micro_batches=params[1],
        #                        num_auto_layers=params[0], forward_stage_layer_ids=get_list(
        #                            int(params[0]), int(params[0]/pp)),
        #                        submesh_physical_shapes=[(5, 1)]*5, submesh_logical_shapes=[(dp, tp)]*pp,
        #                        submesh_autosharding_option_dicts=[force_dp_dict] * pp)
        #     for mod, params in zip(gpt_dojo_models.values(), gpt_dojo_params.values())
        #     for dp, tp, pp in zip([1], [5], [5])
        # ],

    ]),


    24: flatten_list([
        # dp=24, tp=1, pp=1
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=1536,
                               model_spec=mod, num_micro_batches=params[1],
                               num_auto_layers=params[0], forward_stage_layer_ids=get_list(
                                   int(params[0]), int(params[0]/pp)),
                               submesh_physical_shapes=[(6, 4)]*1, submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp_dict] * pp)
            for mod, params in zip(gpt_models.values(), gpt_params.values())
            for dp, tp, pp in zip([24], [1], [1])
        ],


        # # dp=1, tp=24, pp=1
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=1536,
                               model_spec=mod, num_micro_batches=params[1],
                               num_auto_layers=params[0], forward_stage_layer_ids=get_list(
                                   int(params[0]), int(params[0]/pp)),
                               submesh_physical_shapes=[(6, 4)]*1, submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp_dict] * pp)
            for mod, params in zip(gpt_models.values(), gpt_params.values())
            for dp, tp, pp in zip([1], [24], [1])
        ],


        # # # dp=1, tp=1, pp=24 -- 由于小模型的num_layers最大为18，无法实现pp=24, layers=24
        # [
        #     # NOTE: fit for small models with layer=12
        #     get_solution_cases(batch_size=1536,
        #                        model_spec=mod, num_micro_batches=params[1],
        #                        num_auto_layers=params[0], forward_stage_layer_ids=get_list(
        #                            int(params[0]), int(params[0]/pp)),
        #                        submesh_physical_shapes=[(1, 1)]*24,
        #                        submesh_logical_shapes=[(dp, tp)]*pp,
        #                        submesh_autosharding_option_dicts=[force_dp_dict] * pp)
        #     for mod, params in zip(gpt_models.values(), gpt_params.values())
        #     for dp, tp, pp in zip([1], [1], [24])
        # ],



        # # dp=6, tp=4, pp=1
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=1536,
                               model_spec=mod, num_micro_batches=params[1],
                               num_auto_layers=params[0], forward_stage_layer_ids=get_list(
                                   int(params[0]), int(params[0]/pp)),
                               submesh_physical_shapes=[(6, 4)]*1,
                               submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp_dict] * pp)
            for mod, params in zip(gpt_models.values(), gpt_params.values())
            for dp, tp, pp in zip([6], [4], [1])
        ],

        # # dp=6, tp=1, pp=4
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=1536,
                               model_spec=mod, num_micro_batches=params[1],
                               num_auto_layers=params[0], forward_stage_layer_ids=get_list(
                                   int(params[0]), int(params[0]/pp)),
                               submesh_physical_shapes=[(6, 1)]*4,
                               submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp_dict] * pp)
            for mod, params in zip(gpt_models.values(), gpt_params.values())
            for dp, tp, pp in zip([6], [1], [4])
        ],

        # # dp=1, tp=6, pp=4
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=1536,
                               model_spec=mod, num_micro_batches=params[1],
                               num_auto_layers=params[0], forward_stage_layer_ids=get_list(
                                   int(params[0]), int(params[0]/pp)),
                               submesh_physical_shapes=[(6, 1)]*4,
                               submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp_dict] * pp)
            for mod, params in zip(gpt_models.values(), gpt_params.values())
            for dp, tp, pp in zip([1], [6], [4])
        ],


        # # dp=6, tp=2, pp=2
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=1536,
                               model_spec=mod, num_micro_batches=params[1],
                               num_auto_layers=params[0], forward_stage_layer_ids=get_list(
                                   int(params[0]), int(params[0]/pp)),
                               submesh_physical_shapes=[(6, 2)]*2,
                               submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp_dict] * pp)
            for mod, params in zip(gpt_models.values(), gpt_params.values())
            for dp, tp, pp in zip([6], [2], [2])
        ],

    ]),
}
