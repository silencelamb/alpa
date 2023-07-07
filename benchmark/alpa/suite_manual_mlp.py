
from suite_manual_gpt import mlp_specs
from benchmark_parallel_utils import (BenchmarkCase, SearchParallelArgs, UniformParallelArgs,
                                      LoadSolutionParallelArgs, ConfigParallelArgs)
from alpa import WSCManualStageOption
from suite_auto_gpt import get_config_cases, get_search_cases, get_config_cases_idx

grid_search_suite_mlp = {
    4: get_search_cases(mlp_specs["4layers"], [128], [1]),
}

# Performance test on normal models
# B, model, NB, PM, (RS, Remat, 3D Config, FM)
mlp_perf_test_suite = {
    1: [
        BenchmarkCase(16, mlp_specs["4layers"], 1, "uniform",
                      UniformParallelArgs(True, True, 1, 1, 1, True))
    ],
    8: [
        BenchmarkCase(32, mlp_specs["4layers"], 4, "uniform",
                      UniformParallelArgs(True, True, 2, 2, 2, True))
    ],
    64: [
        BenchmarkCase(1024, mlp_specs["4layers"], 1024, "uniform",
                      UniformParallelArgs(True, True, 1, 4, 16, True))
    ],
}

