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

}

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
    1: [
        BenchmarkCase(16, bert_specs["Tiny"], 1, "uniform",
                      UniformParallelArgs(True, True, 1, 1, 1, True))
    ],
    # add for test by daixu
    2: [
        BenchmarkCase(16, bert_specs["Mini"], 4, "uniform",
                      UniformParallelArgs(True, True, 1, 2, 1, True))
    ],
    8: [
        BenchmarkCase(32, bert_specs["Medium"], 4, "uniform",
                      UniformParallelArgs(True, True, 2, 2, 2, True))
    ],
    64: [
        BenchmarkCase(1024, bert_specs["Large"], 1024, "uniform",
                      UniformParallelArgs(True, True, 1, 4, 16, True))
    ],
}
