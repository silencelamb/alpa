"""Benchmark suites for bert with manual specifications."""
from collections import namedtuple
from benchmark_parallel_utils import BenchmarkCase, UniformParallelArgs

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# head = num_heads,
# NB = num_micro_batches, PM = parallel_mode
# 3D config = 3D parallel config (Data, Operator, Pipeline)
# RS = prefer_reduce_scatter, Remat = use_rematerialization,
# FM = force_batch_dim_mapping,

BERTModelConfig = namedtuple(
    "BERTModelConfig",
    ["seq_len", "hidden_size", "num_layers", "num_heads", "vocab_size"])

MLPModelConfig = namedtuple(
    "MLPModelConfig",
    ["num_layers", "hidden_size", "use_bias"])

mlp_specs = {           
    "4layers":MLPModelConfig(16,1024,True),
}

bert_specs = {
    # NOTE: BERT-Large: 340M, BERT-Base: 110M
    #                    Seq_len, Hidden, Layer,  heads, vocab
    "Tiny": BERTModelConfig(512, 128, 2, 8, 30522),
    "Mini": BERTModelConfig(512, 256, 4, 8, 30522),
    "Small": BERTModelConfig(512, 512, 4, 8, 30522),
    "Medium": BERTModelConfig(512, 512, 8, 8, 30522),
    "Base": BERTModelConfig(512, 768, 12, 12, 30522),
    "Large": BERTModelConfig(512, 1024, 24, 16, 30522),
    "LL": BERTModelConfig(512, 1536, 24, 16, 30522),
    "LLL": BERTModelConfig(512, 2048, 24, 32, 30522),
    "LLLL": BERTModelConfig(512, 2560, 32, 32, 30522),
    "LLLLL": BERTModelConfig(512, 4096, 32, 32, 30522),
    "LLLLLL": BERTModelConfig(512, 5120, 48, 40, 30522),
    "LLLLLLL": BERTModelConfig(512, 8192, 48, 64, 30522),
    "LLLLLLLL": BERTModelConfig(512, 10240, 60, 80, 30522),
}

# bert_params = {
#     #                      S，   H,   L,  head,   V,
#     tuple(BERTModelConfig(512, 128, 2, 8, 30522)): ["125M", ""],
#     tuple(BERTModelConfig(512, 256, 4, 8, 30522)): ["350M", ""],
#     tuple(BERTModelConfig(512, 512, 4, 8, 30522)): ["760M", ""],
#     tuple(BERTModelConfig(512, 512, 8, 8, 30522)): ["1.3B", ""],
#     tuple(BERTModelConfig(512, 768, 12, 12, 30522)): ["110M", ""],
#     tuple(BERTModelConfig(512, 1024, 24, 16, 30522)): ["340M", ""],

#     tuple(BERTModelConfig(512, 5120, 48, 40, 30522)): ["15B", ""],
#     tuple(BERTModelConfig(512, 8192, 48, 64, 30522)): ["39B", ""],
#     tuple(BERTModelConfig(512, 10240, 60, 80, 30522)): ["76B", ""],
# }


_ = None

# Temporary debug suite
# key = the number of gpus, value = a list of cases
# B, model, NB, PM, (RS, Remat, 3D Config, FM)
tmp_suite = {
    1: [
        BenchmarkCase(16, bert_specs["Tiny"], 1, "uniform",
                      UniformParallelArgs(True, True, 1, 1, 1, True))
    ],
    8: [
        BenchmarkCase(128, bert_specs["Medium"],
                      4, "uniform",
                      UniformParallelArgs(True, True, 4, 1, 2, True)),
    ],
}

# Fast performance test on models with fewer layers
# B, model, NB, PM, (RS, Remat, 3D Config, FM)
perf_test_fast_2d_suite = {
    1: [
        BenchmarkCase(8, bert_specs["Tiny"], 1, "uniform",
                      UniformParallelArgs(False, True, 1, 1, 1, True))
    ],
    8: [
        BenchmarkCase(32, bert_specs["Mini"],
                      1, "uniform",
                      UniformParallelArgs(True, True, 8, 1, 1, True)),
        BenchmarkCase(128, bert_specs["Medium"],
                      4, "uniform",
                      UniformParallelArgs(True, True, 8, 1, 1, True)),
    ],
}

# Performance test on normal models
# B, model, NB, PM, (RS, Remat, 3D Config, FM)
perf_test_suite = {
    # 1: [
    #     BenchmarkCase(16, bert_specs["Tiny"], 1, "uniform",
    #                   UniformParallelArgs(True, True, 1, 1, 1, True))
    # ],
    1: [
        BenchmarkCase(16, bert_specs["Large"], 1, "uniform",
                      UniformParallelArgs(True, True, 1, 1, 1, True))
    ],
    # add for test by daixu
    2: [
        BenchmarkCase(16, bert_specs["LL"], 4, "uniform",
                      UniformParallelArgs(True, True, 1, 2, 1, True))
    ],
    4: [
        BenchmarkCase(16, bert_specs["LLL"], 4, "uniform",
                      UniformParallelArgs(True, True, 2, 2, 1, True))
    ],
    8: [
        BenchmarkCase(32, bert_specs["LLLL"], 8, "uniform",
                      UniformParallelArgs(True, True, 2, 4, 1, True))
    ],
    16: [
        BenchmarkCase(128, bert_specs["LLLLL"], 16, "uniform",
                      UniformParallelArgs(True, True, 2, 4, 2, True))
    ],

    32: [
        BenchmarkCase(256, bert_specs["LLLLLL"], 32, "uniform",
                      UniformParallelArgs(True, True, 2, 4, 4, True))
    ],
    64: [
        BenchmarkCase(1024, bert_specs["LLLLLLL"], 128, "uniform",
                      UniformParallelArgs(True, True, 1, 4, 16, True))
    ],
    120: [
        BenchmarkCase(2048, bert_specs["LLLLLLLL"], 256, "uniform",
                      UniformParallelArgs(True, True, 2, 4, 15, True))
    ],
}
