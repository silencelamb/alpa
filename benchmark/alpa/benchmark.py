"""The entry point of intra-op + inter-op parallelism benchmark."""
from ast import arg
import os
import argparse
from datetime import datetime
import time
import json

import numpy as np

from alpa.global_env import get_global_config, set_global_config
from alpa.util import (write_tsv, get_num_hosts_and_num_devices, to_str_round,
                       GB)
from gen_mapping_vis_result import gen_mapping_vis_result
from benchmark_parallel_utils import BenchmarkCase, ConfigParallelArgs

from benchmark_one_case import benchmark_one_case
import suite_auto_gpt
import suite_auto_moe
import suite_manual_gpt
import suite_manual_moe
import suite_wresnet
import suite_inference_gpt

benchmark_suites = {
    "gpt.tmp": suite_manual_gpt.tmp_suite,
    "gpt.tmp_auto": suite_auto_gpt.tmp_suite,
    "gpt.perf_test_fast_2d": suite_manual_gpt.perf_test_fast_2d_suite,
    "gpt.perf_test_manual": suite_manual_gpt.perf_test_suite,
    "gpt.perf_test_auto": suite_auto_gpt.perf_test_suite,
    "gpt.grid_search_auto": suite_auto_gpt.grid_search_suite,
    "gpt.correctness_test_auto": suite_auto_gpt.correctness_test_suite,
    "gpt_inference.profile": suite_inference_gpt.profile_suite,
    "gpt_no_embedding_inference.profile": suite_inference_gpt.profile_suite,
    "gpt.config_test": suite_auto_gpt.config_test_suite,
    "gpt.wsc_config_test": suite_auto_gpt.wsc_config_test_suite, 
    "moe.tmp": suite_manual_moe.tmp_suite,
    "moe.tmp_auto": suite_auto_moe.tmp_suite,
    "moe.perf_test_fast_2d": suite_manual_moe.perf_test_fast_2d_suite,
    "moe.perf_test_auto": suite_auto_moe.perf_test_suite,
    "moe.grid_search_auto": suite_auto_moe.grid_search_suite,
    "wresnet.perf_test_2d": suite_wresnet.perf_test_2d_suite,
    "wresnet.perf_test_auto": suite_wresnet.perf_test_auto_suite,
    "wresnet.grid_search_auto": suite_wresnet.grid_search_auto_suite,
}


def benchmark_suite(suite_name,
                    num_hosts,
                    num_devices_per_host,
                    exp_name="default",
                    niter=3,
                    shard_only=False,
                    local=False,
                    profile_driver_time=False,
                    disable_tqdm=False,
                    use_separate_process=True):
    num_gpus = num_hosts * num_devices_per_host

    if local:
        assert shard_only, ("Only shard-only mode is supported for execution "
                            "on local GPUs.")

    assert num_gpus in benchmark_suites[suite_name], (
        f"No available benchmark suite for {suite_name} on {num_gpus} GPUs")
    suite = benchmark_suites[suite_name][num_gpus]

    os.makedirs("tmp", exist_ok=True)

    model_type = suite_name.split(".")[0]
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    global_config = get_global_config()
    output_name = f"{global_config.rst_folder}/{model_type}_alpa_{exp_name}_{date_str}.tsv"

    # Run all cases
    for benchmark_case in suite:
        benchmark_case: BenchmarkCase
        totol_batch_size = benchmark_case.batch_size
        model_config = benchmark_case.model_config
        num_micro_batches = benchmark_case.num_micro_batches
        try:
            auto_layers = benchmark_case.parallel_args.num_auto_layers
        except AttributeError:
            auto_layers = 'auto'

        global_config.maping_rst_dir = f"{global_config.rst_folder}/Batchsize_{totol_batch_size}" + \
            f"-num_b_{num_micro_batches}-auto_layers_{auto_layers}"
        os.makedirs(global_config.maping_rst_dir, exist_ok=True)
        set_global_config(global_config)
        parallel_args = benchmark_case.parallel_args

        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))
        result = benchmark_one_case(model_type,
                                    benchmark_case,
                                    niter,
                                    num_hosts,
                                    num_devices_per_host,
                                    shard_only=shard_only,
                                    local=local,
                                    profile_driver_time=profile_driver_time,
                                    disable_tqdm=disable_tqdm,
                                    use_separate_process=use_separate_process)

        (parameter_count, peak_mem, latencies, tflops, metadata) = result

        heads = [
            "Type", "Model Config", "#Microbatch", "#GPU", "Parallel Config",
            "Mean Time (s)", "Std Time (s)", "#Params (Billion)", "TFLOPs",
            "Peak Mem (GB)", "Metadata"
        ]
        if isinstance(parallel_args, ConfigParallelArgs):
            parallel_args = parallel_args._replace(input_placement_specs=[])
            
        values = [
            model_type, model_config, num_micro_batches, num_gpus,
            parallel_args, f"{np.mean(latencies):.3f}",
            f"{np.std(latencies):.3f}", f"{parameter_count/1e9:.3f}B",
            f"{tflops:.2f}", f"{peak_mem/GB:.3f}",
            to_str_round(metadata, 6)
        ]
        write_tsv(heads, values, output_name)
        values = [str(x) for x in values]
        result_dict = dict(zip(heads, values))
        with open(global_config.maping_rst_dir+"/over_all_perf.json", "w") as f:
            json.dump(result_dict, f, indent=4)
        gen_mapping_vis_result(global_config.maping_rst_dir)
        time.sleep(0.1)  # for ctrl+c to work


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",
                        choices=list(benchmark_suites.keys()),
                        type=str,
                        required=True)
    parser.add_argument("--niter",
                        type=int,
                        default=3,
                        help="The number of benchmark iterations")
    parser.add_argument("--num-hosts", type=int, default=None)
    parser.add_argument("--num-devices-per-host", type=int, default=None)
    parser.add_argument("--shard-only",
                        action="store_true",
                        help="Only profile the 2D case. No pipeline "
                        "parallelism.")
    parser.add_argument("--local",
                        action="store_true",
                        help="Run on local GPUs. Do not use ray actors.")
    parser.add_argument("--profile-driver-time",
                        action="store_true",
                        help="Profile the execution time on the driver instead "
                        "of the workers.")
    parser.add_argument("--no-separate-process",
                        action="store_false",
                        help="Do not launch separate processes for benchmark. "
                        "Errors in a single case will terminate this "
                        "script.",
                        dest="use_separate_process")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--only-mapping", action="store_true", dest="only_mapping")
    parser.add_argument("--use-analytical-perf-model", action="store_true", dest="use_analytical_perf_model")
    parser.add_argument("--rst_folder", type=str, default="")
    parser.add_argument("--hardware", type=str, default="gpu")
    parser.add_argument("--force_use_fp16", action="store_true")
    args = parser.parse_args()
    num_hosts, num_devices_per_host = get_num_hosts_and_num_devices(args)

    # set global_config, only_mapping
    global_config = get_global_config()
    global_config.only_mapping = args.only_mapping

    # set whether use analytical model
    global_config.use_analytical_perf_model = args.use_analytical_perf_model

    # set mapping result save dir
    if args.rst_folder == "":
        args.rst_folder = "tmp"
    
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if args.only_mapping:
        if global_config.use_analytical_perf_model:
            actual_or_virtual = f"perf@{global_config.hardware}"
        else:
            actual_or_virtual = "costmodel"
    else:
        actual_or_virtual =  "actualA100"
    args.rst_folder = f"{args.rst_folder}/{args.suite}-{num_devices_per_host}X{num_hosts}-{actual_or_virtual}-{date_str}"
    print(args.rst_folder)
    os.makedirs(args.rst_folder, exist_ok=True)

    global_config.rst_folder = args.rst_folder
    global_config.hardware = args.hardware
    global_config.force_use_fp16 = args.force_use_fp16

    set_global_config(global_config)
    global_config = get_global_config()
    print(global_config.use_analytical_perf_model)


    benchmark_suite(args.suite, num_hosts, num_devices_per_host, args.exp_name,
                    args.niter, args.shard_only, args.local,
                    args.profile_driver_time, args.disable_tqdm,
                    args.use_separate_process)
