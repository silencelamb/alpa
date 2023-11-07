"""Benchmark suites for gpt with manual specifications."""
from collections import namedtuple
from benchmark_parallel_utils import BenchmarkCase, UniformParallelArgs

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# head = num_heads,
# NB = num_micro_batches, PM = parallel_mode
# 3D config = 3D parallel config (Data, Operator, Pipeline)
# RS = prefer_reduce_scatter, Remat = use_rematerialization,
# FM = force_batch_dim_mapping,

GPTModelConfig = namedtuple(
    "GPTModelConfig",
    ["seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size"])

MLPModelConfig = namedtuple(
    "MLPModelConfig",
    ["num_layers", "hidden_size", "use_bias"])

mlp_specs = {
    "4layers":MLPModelConfig(16,1024,True),
}

gpt_specs = {
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

gpt_wsc_specs = {
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
    #                      S，   H,   L,  head,   V,
    tuple(GPTModelConfig(1024, 768, 12, 12, 51200)): ["125M", ""],
    tuple(GPTModelConfig(1024, 1024, 24, 16, 51200)): ["350M", ""],
    tuple(GPTModelConfig(1024, 1536, 24, 16, 51200)): ["760M", ""],
    tuple(GPTModelConfig(1024, 2048, 24, 32, 51200)): ["1.3B", ""],
    tuple(GPTModelConfig(1024, 2560, 32, 32, 51200)): ["2.6B", ""],
    tuple(GPTModelConfig(1024, 4096, 32, 32, 51200)): ["6.7B", ""],
    tuple(GPTModelConfig(1024, 5120, 48, 40, 51200)): ["15B", ""],
    tuple(GPTModelConfig(1024, 8192, 48, 64, 51200)): ["39B", ""],
    tuple(GPTModelConfig(1024, 10240, 60, 80, 51200)): ["76B", ""],
}

_ = None

# Temporary debug suite
# key = the number of gpus, value = a list of cases
# BenchmarCase(B, model, NB, PM, (RS, Remat, 3D Config, FM))
tmp_suite = {
    1: [
        BenchmarkCase(16, gpt_specs["350M"], 1, "uniform",
                      UniformParallelArgs(True, True, 1, 1, 1, True))
    ],
    8: [
        BenchmarkCase(128, GPTModelConfig(1024, 4096, 4, 32, 51200),
                      4, "uniform",
                      UniformParallelArgs(True, True, 4, 1, 2, True)),
    ],
}

# Fast performance test on models with fewer layers
# B, model, NB, PM, (RS, Remat, 3D Config, FM)
perf_test_fast_2d_suite = {
    1: [
        BenchmarkCase(8, GPTModelConfig(1024, 1024, 4, 32, 51200), 1, "uniform",
                      UniformParallelArgs(False, True, 1, 1, 1, True))
    ],
    8: [
        BenchmarkCase(32, GPTModelConfig(1024, 4096, 4, 32, 51200),
                      1, "uniform",
                      UniformParallelArgs(True, True, 8, 1, 1, True)),
        BenchmarkCase(128, GPTModelConfig(1024, 4096, 4, 32, 51200),
                      4, "uniform",
                      UniformParallelArgs(True, True, 8, 1, 1, True)),
    ],
}

# Performance test on normal models
# B, model, NB, PM, (RS, Remat, 3D Config, FM)
# NOTE: for manual test -- #gpu must be equal to prod(3D config)
perf_test_suite = {
    # 1: [
    #     BenchmarkCase(16, gpt_specs["125M"], 1, "uniform",
    #                   UniformParallelArgs(True, True, 1, 1, 1, True))
    # ],
    1: [
        BenchmarkCase(16, gpt_specs["350M"], 1, "uniform",
                      UniformParallelArgs(True, True, 1, 1, 1, True))
    ],
    # add for test by daixu
    2: [
        BenchmarkCase(16, gpt_specs["760M"], 4, "uniform",
                      UniformParallelArgs(True, True, 1, 2, 1, True))
    ],
    4: [
        BenchmarkCase(16, gpt_specs["1.3B"], 4, "uniform",
                      UniformParallelArgs(True, True, 2, 2, 1, True))
    ],
    8: [
        BenchmarkCase(32, gpt_specs["2.6B"], 8, "uniform",
                      UniformParallelArgs(True, True, 2, 4, 1, True))
    ],
    16: [
        BenchmarkCase(128, gpt_specs["6.7B"], 16, "uniform",
                      UniformParallelArgs(True, True, 2, 4, 2, True))
    ],

    32: [
        BenchmarkCase(256, gpt_specs["15B"], 32, "uniform",
                      UniformParallelArgs(True, True, 2, 4, 4, True))
    ],
    64: [
        BenchmarkCase(1024, gpt_specs["39B"], 128, "uniform",
                      UniformParallelArgs(True, True, 1, 4, 16, True))
    ],
    # NOTE: 76B layers = 60, max factor=15, PP(pipeline parallel)=15
    120: [
        BenchmarkCase(2048, gpt_specs["76B"], 256, "uniform",
                      UniformParallelArgs(True, True, 2, 4, 15, True))
    ],
}
