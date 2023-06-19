
from suite_manual_gpt import mlp_specs
from benchmark_parallel_utils import (BenchmarkCase, SearchParallelArgs,
                                      LoadSolutionParallelArgs, ConfigParallelArgs)
from alpa import WSCManualStageOption
from suite_auto_gpt import get_config_cases, get_search_cases

grid_search_suite_mlp = {
    4: get_search_cases(mlp_specs["4layers"], [128], [1]),
}

wsc_config_test_suite_mlp = {
    
    4: get_config_cases(mlp_specs["4layers"], [128],
                        'data/tmp_22_gpu_analytical/mlp.grid_search_auto-4X1-perf@gpu-2023-06-16-03-10-49/Batchsize_1024-num_b_128-auto_layers_1/input_placement_specs.pkl',                        
                        stage_option=WSCManualStageOption(forward_stage_layer_ids=[[0]],
                                                          submeshes=[
                            [0, 0, 0, 3],                            
                        ], 
        submesh_physical_shapes=None,
        submesh_logical_shapes=None,
        submesh_autosharding_option_dicts=[{}])
    ),    
}
