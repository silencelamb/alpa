
from suite_manual_gpt import mlp_specs
from benchmark_parallel_utils import (BenchmarkCase, SearchParallelArgs,
                                      LoadSolutionParallelArgs, ConfigParallelArgs)
from alpa import WSCManualStageOption
from suite_auto_gpt import get_config_cases, get_search_cases

grid_search_suite_mlp = {
    4: get_search_cases(mlp_specs["4layers"], [128], [1]),
}

wsc_config_test_suite_mlp = {
    2: get_config_cases(mlp_specs["4layers"], [128],
                        'data/tmp_a100_gpu_real/gpt.grid_search_auto-2X1-actualA100-2023-03-01-02-57-12/Batchsize_1024-num_b_128-auto_layers_6/input_placement_specs.pkl',
                        stage_option=WSCManualStageOption(forward_stage_layer_ids=[[0], [1]],
                                                          submeshes=[
                            [0, 0, 0, 0],
                            [0, 1, 0, 1]
                        ],
        submesh_physical_shapes=None,
        submesh_logical_shapes=None,
        submesh_autosharding_option_dicts=[{}, {}])
    ),
    4: get_config_cases(mlp_specs["4layers"], [128],
                        # 'data/tmp_22_gpu_analytical/mlp.grid_search_auto-4X1-perf@gpu-2023-06-14-05-45-36/Batchsize_1024-num_b_128-auto_layers_1/input_placement_specs.pkl',
                        'zhc_test/input_placement_specs_mlp.pkl',
                        stage_option=WSCManualStageOption(forward_stage_layer_ids=[[0], [1], [2]],
                                                          submeshes=[
                            [0, 0, 0, 0],
                            [0, 1, 0, 1],
                            [0, 2, 0, 3],
                        ],
        submesh_physical_shapes=None,
        submesh_logical_shapes=None,
        submesh_autosharding_option_dicts=[{}, {}, {}])
    ),
    8: get_config_cases(mlp_specs["4layers"], [128],
                        'data/tmp_wsc_perf_15GB_fp16/gpt.grid_search_auto-8X1-perf@gpu-2023-03-07-09-02-58/Batchsize_1024-num_b_128-auto_layers_8/input_placement_specs.pkl',
                        stage_option=WSCManualStageOption(forward_stage_layer_ids=[[0], [1], [2]],
                                                       submeshes=[
                                                                    [0, 0, 0, 1],
                                                                    [1, 0, 1, 1],
                                                                    [2, 0, 3, 1],
                                                                ],
                                                       submesh_physical_shapes=None,
                                                       submesh_logical_shapes=None,
                                                       submesh_autosharding_option_dicts=[{}, {}, {}])
                        ),
}
