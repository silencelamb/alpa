from suite_manual_bert import BERTModelConfig
from collections import namedtuple
from benchmark_parallel_utils import (BenchmarkCase, SearchParallelArgs,
                                      LoadSolutionParallelArgs, ConfigParallelArgs,
                                      UniformParallelArgs)


bert_models = {
    # #                    Seq_len, Hidden, Layer,  heads, vocab
    # "Tiny": BERTModelConfig(512, 128, 2, 8, 30522),
    # "Mini": BERTModelConfig(512, 256, 4, 8, 30522),
    # "Small": BERTModelConfig(512, 512, 4, 8, 30522),
    # "Medium": BERTModelConfig(512, 512, 8, 8, 30522),
    "Base": BERTModelConfig(512, 768, 12, 12, 30522),
    "Large": BERTModelConfig(512, 1024, 24, 16, 30522),
    "LL": BERTModelConfig(512, 1536, 24, 24, 30522),
    "LLL": BERTModelConfig(512, 2048, 32, 32, 30522),
    "LLLL": BERTModelConfig(512, 2560, 32, 32, 30522),
    "LLLLL": BERTModelConfig(512, 4096, 48, 32, 30522),
    "LLLLLL": BERTModelConfig(512, 5120, 48, 40, 30522),
    # "LLLLLLL": BERTModelConfig(512, 8192, 64, 64, 30522),
    # "LLLLLLLL": BERTModelConfig(512, 10240, 80, 64, 30522),
}

auto_layers = 12

bert_num_batchs = [64, 128, 256, 1536]


# NOTE: only for DOJO
bert_dojo_models = {
    # #                    Seq_len, Hidden, Layer,  heads, vocab
    # "Tiny": BERTModelConfig(512, 128, 2, 8, 30522),
    # "Mini": BERTModelConfig(512, 256, 4, 8, 30522),
    # "Small": BERTModelConfig(512, 512, 4, 8, 30522),
    # "Medium": BERTModelConfig(512, 512, 8, 8, 30522),
    "Base": BERTModelConfig(512, 768, 12, 12, 30522),
    "Large": BERTModelConfig(512, 1024, 24, 16, 30522),
    "LL": BERTModelConfig(512, 1536, 24, 24, 30522),
    "LLL": BERTModelConfig(512, 2048, 32, 32, 30522),
    "LLLL": BERTModelConfig(512, 2560, 32, 32, 30522),
    "LLLLL": BERTModelConfig(512, 4096, 48, 32, 30522),
    "LLLLLL": BERTModelConfig(512, 5120, 48, 40, 30522),
    "LLLLLLL": BERTModelConfig(512, 8192, 64, 64, 30522),
    "LLLLLLLL": BERTModelConfig(512, 10240, 80, 64, 30522),

}

dojo_auto_layers = 10
bert_dojo_num_batchs = [40, 200, 1000]


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


dojo_global_batch = 1000
wsgpu_global_batch = 1536

prefer_reduce_scatter = True
use_remat = True
# Force DP
force_dp = {"force_batch_dim_to_mesh_dim": 0}
# below is force data parallel --
force_dp1 = {"force_data_parallel": True}

auto_sharding_dict = {"force_simple_heuristic": "shard-first"}

force_zero_stage_3 = {"force_zero_stage_3": True}

wsc_perf_suite = {
    #             [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]],
    25:  flatten_list([

        # dp=25, tp=1, pp=1
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=dojo_global_batch,
                               model_spec=mod, num_micro_batches=num_batchs,
                               num_auto_layers=dojo_auto_layers, forward_stage_layer_ids=get_list(
                                   int(dojo_auto_layers), int(dojo_auto_layers/pp)),
                               submesh_physical_shapes=[(5, 5)]*1, submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp] * pp)
            for mod in bert_dojo_models.values()
            for num_batchs in bert_dojo_num_batchs
            for dp, tp, pp in zip([25], [1], [1])
            if (dojo_global_batch/num_batchs) % dp == 0
        ],


        # dp=1, tp=25, pp=1
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=dojo_global_batch,
                               model_spec=mod, num_micro_batches=num_batchs,
                               num_auto_layers=dojo_auto_layers, forward_stage_layer_ids=get_list(
                                   int(dojo_auto_layers), int(dojo_auto_layers/pp)),
                               submesh_physical_shapes=[(5, 5)]*1, submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[auto_sharding_dict] * pp)
            for mod in bert_dojo_models.values()
            for num_batchs in bert_dojo_num_batchs
            for dp, tp, pp in zip([1], [25], [1])
            if (dojo_global_batch/num_batchs) % dp == 0
        ],


        # dp=5, tp=5, pp=1
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=dojo_global_batch,
                               model_spec=mod, num_micro_batches=num_batchs,
                               num_auto_layers=dojo_auto_layers, forward_stage_layer_ids=get_list(
                                   int(dojo_auto_layers), int(dojo_auto_layers/pp)),
                               submesh_physical_shapes=[(5, 5)]*1, submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp] * pp)
            for mod in bert_dojo_models.values()
            for num_batchs in bert_dojo_num_batchs
            for dp, tp, pp in zip([5], [5], [1])
            if (dojo_global_batch/num_batchs) % dp == 0
        ],

        # dp=5, tp=1, pp=5
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=dojo_global_batch,
                               model_spec=mod, num_micro_batches=num_batchs,
                               num_auto_layers=dojo_auto_layers, forward_stage_layer_ids=get_list(
                                   int(dojo_auto_layers), int(dojo_auto_layers/pp)),
                               submesh_physical_shapes=[(5, 1)]*5, submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp] * pp)
            for mod in bert_dojo_models.values()
            for num_batchs in bert_dojo_num_batchs
            for dp, tp, pp in zip([5], [1], [5])
            if (dojo_global_batch/num_batchs) % dp == 0
        ],



    ]),


    24: flatten_list([


        # dp=24, tp=1, pp=1
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=wsgpu_global_batch,
                               model_spec=mod, num_micro_batches=num_batchs,
                               num_auto_layers=auto_layers, forward_stage_layer_ids=get_list(
                                   int(auto_layers), int(auto_layers/pp)),
                               submesh_physical_shapes=[(6, 4)]*1, submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp] * pp)
            for mod in bert_models.values()
            for num_batchs in bert_num_batchs
            for dp, tp, pp in zip([24], [1], [1])
            if (wsgpu_global_batch/num_batchs) % dp == 0
        ],


        # # dp=1, tp=24, pp=1
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=wsgpu_global_batch,
                               model_spec=mod, num_micro_batches=num_batchs,
                               num_auto_layers=auto_layers, forward_stage_layer_ids=get_list(
                                   int(auto_layers), int(auto_layers/pp)),
                               submesh_physical_shapes=[(6, 4)]*1, submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[auto_sharding_dict] * pp)
            for mod in bert_models.values()
            for num_batchs in bert_num_batchs
            for dp, tp, pp in zip([1], [24], [1])
            if (wsgpu_global_batch/num_batchs) % dp == 0
        ],


        # # dp=6, tp=4, pp=1
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=wsgpu_global_batch,
                               model_spec=mod, num_micro_batches=num_batchs,
                               num_auto_layers=auto_layers, forward_stage_layer_ids=get_list(
                                   int(auto_layers), int(auto_layers/pp)),
                               submesh_physical_shapes=[(6, 4)]*1,
                               submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp] * pp)
            for mod in bert_models.values()
            for num_batchs in bert_num_batchs
            for dp, tp, pp in zip([6], [4], [1])
            if (wsgpu_global_batch/num_batchs) % dp == 0
        ],

        # # dp=6, tp=1, pp=4
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=wsgpu_global_batch,
                               model_spec=mod, num_micro_batches=num_batchs,
                               num_auto_layers=auto_layers, forward_stage_layer_ids=get_list(
                                   int(auto_layers), int(auto_layers/pp)),
                               submesh_physical_shapes=[(6, 1)]*4,
                               submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp] * pp)
            for mod in bert_models.values()
            for num_batchs in bert_num_batchs
            for dp, tp, pp in zip([6], [1], [4])
            if (wsgpu_global_batch/num_batchs) % dp == 0
        ],

        # # dp=1, tp=6, pp=4
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=wsgpu_global_batch,
                               model_spec=mod, num_micro_batches=num_batchs,
                               num_auto_layers=auto_layers, forward_stage_layer_ids=get_list(
                                   int(auto_layers), int(auto_layers/pp)),
                               submesh_physical_shapes=[(6, 1)]*4,
                               submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[auto_sharding_dict] * pp)
            for mod in bert_models.values()
            for num_batchs in bert_num_batchs
            for dp, tp, pp in zip([1], [6], [4])
            if (wsgpu_global_batch/num_batchs) % dp == 0
        ],


        # # dp=6, tp=2, pp=2
        [
            # NOTE: fit for small models with layer=12
            get_solution_cases(batch_size=wsgpu_global_batch,
                               model_spec=mod, num_micro_batches=num_batchs,
                               num_auto_layers=auto_layers, forward_stage_layer_ids=get_list(
                                   int(auto_layers), int(auto_layers/pp)),
                               submesh_physical_shapes=[(6, 2)]*2,
                               submesh_logical_shapes=[(dp, tp)]*pp,
                               submesh_autosharding_option_dicts=[force_dp] * pp)
            for mod in bert_models.values()
            for num_batchs in bert_num_batchs
            for dp, tp, pp in zip([6], [2], [2])
            if (wsgpu_global_batch/num_batchs) % dp == 0
        ],

    ]),
}
