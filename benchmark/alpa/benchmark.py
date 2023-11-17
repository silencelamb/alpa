"""The entry point of intra-op + inter-op parallelism benchmark."""
from ast import arg
import os
import argparse
from datetime import datetime
import time
import json

import numpy as np

from alpa.global_env import get_global_config, set_global_config, get_collective_cost_dict
from alpa.global_env import PrimitiveType
from alpa.util import (write_tsv, to_str_round, GB, TOPS)
from gen_mapping_vis_result import gen_mapping_vis_result
from benchmark_parallel_utils import BenchmarkCase, ConfigParallelArgs

from benchmark_one_case import benchmark_one_case
import suite_auto_gpt
import suite_auto_bert
import suite_manual_gpt
import suite_manual_bert
import suite_wresnet
import suite_inference_gpt
import suite_inference_bert

from suite_auto_gpt import model_type_size_dict

from suite_all import wsc_perf_suite, all_models

import suite_gpt
import suite_bert
import suite_wresnet0
benchmark_suites = {
    "wsc_perf": wsc_perf_suite,
    "gpt.perf_test_fast_2d": suite_manual_gpt.perf_test_fast_2d_suite,
    "gpt.perf_test_manual": suite_manual_gpt.perf_test_suite,
    "gpt.perf_test_auto": suite_auto_gpt.perf_test_suite,

    "gpt.grid_search_auto": suite_auto_gpt.grid_search_suite,

    "gpt.correctness_test_auto": suite_auto_gpt.correctness_test_suite,
    "gpt_inference.profile": suite_inference_gpt.profile_suite,
    "gpt_no_embedding_inference.profile": suite_inference_gpt.profile_suite,
    "gpt.config_test": suite_auto_gpt.config_test_suite,
    "gpt.wsc_config_test": suite_auto_gpt.wsc_config_test_suite,
    "gpt.wsc_perf": suite_gpt.wsc_perf_suite,
    "gpt.wsc_perf_debug": suite_auto_gpt.wsc_perf_debug_suite,


    "bert.tmp": suite_manual_bert.tmp_suite,
    "bert.tmp_auto": suite_auto_bert.tmp_suite,
    "bert.perf_test_fast_2d": suite_manual_bert.perf_test_fast_2d_suite,
    "bert.perf_test_manual": suite_manual_bert.perf_test_suite,
    "bert.perf_test_auto": suite_auto_bert.perf_test_suite,

    "bert.grid_search_auto": suite_auto_bert.grid_search_suite,

    "bert.correctness_test_auto": suite_auto_bert.correctness_test_suite,
    "bert_inference.profile": suite_inference_bert.profile_suite,
    "bert_no_embedding_inference.profile": suite_inference_bert.profile_suite,
    "bert.config_test": suite_auto_bert.config_test_suite,
    "bert.wsc_config_test": suite_auto_bert.wsc_config_test_suite,

    "bert.wsc_perf": suite_bert.wsc_perf_suite,
    "bert.wsc_perf0": suite_auto_bert.wsc_perf_suite0,
    "bert.wsc_perf1": suite_auto_bert.wsc_perf_suite1,
    "bert.wsc_perf2": suite_auto_bert.wsc_perf_suite2,

    # "wresnet.perf_test_2d": suite_wresnet.perf_test_2d_suite,
    "wresnet.perf_test_auto": suite_wresnet.perf_test_auto_suite,
    "wresnet.grid_search_auto": suite_wresnet.grid_search_auto_suite,
    "wresnet.wsc_config_test": suite_wresnet.wsc_config_test_suite,

    "wresnet.wsc_perf": suite_wresnet0.wsc_perf_suite,
    "wresnet.wsc_perf0": suite_wresnet.wsc_perf_suite0,
    "wresnet.wsc_perf1": suite_wresnet.wsc_perf_suite1,
    "wresnet.wsc_perf2": suite_wresnet.wsc_perf_suite2,

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
                    use_separate_process=True,
                    heads=None, table=None,
                    offload=False):

    num_gpus = num_hosts * num_devices_per_host
    if local:
        assert shard_only, ("Only shard-only mode is supported for execution "
                            "on local GPUs.")

    assert num_gpus in benchmark_suites[suite_name], (
        f"No available benchmark suite for {suite_name} on {num_gpus} GPUs")
    # NOTE: First select suite_name, then select num_gpus

    suite = benchmark_suites[suite_name][num_gpus]

    for i in range(len(suite)):
        benchmark_case = suite[i]
        os.makedirs("tmp", exist_ok=True)

        model_type = suite_name.split(".")[0]
        # model_type = list(all_models.keys())[i]
        # if model_type[0] == "G":
        #     model_type = "gpt"
        # elif model_type[0] == "B":
        #     model_type = "bert"
        # elif model_type[0] in ("W", "R"):
        #     model_type = "wresnet"

        date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        global_config = get_global_config()
        output_name = f"{global_config.rst_folder}/{model_type}_alpa_{exp_name}_{date_str}.tsv"
        # Run all cases

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
        if global_config.save_jaxpr_json:
            micro_batch_size = totol_batch_size // num_micro_batches
            model_size = model_type_size_dict[model_type][num_gpus]
            use_remat = benchmark_case.parallel_args.use_remat
            global_config.save_jaxpr_json_file = os.path.join(global_config.save_jaxpr_dir,
                                                              f"{model_type}{model_size}_bsize{micro_batch_size}_useremat_{use_remat}.json")
        set_global_config(global_config)
        parallel_args = benchmark_case.parallel_args

        logical_shape = parallel_args.submesh_logical_shapes
        parallel = (logical_shape[0][0],
                    logical_shape[0][1], len(logical_shape))
        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))
        print(f"***************3D Parallel = {parallel}")

        result = benchmark_one_case(model_type,
                                    benchmark_case,
                                    niter,
                                    num_hosts,
                                    num_devices_per_host,
                                    shard_only=shard_only,
                                    local=local,
                                    profile_driver_time=profile_driver_time,
                                    disable_tqdm=disable_tqdm,
                                    use_separate_process=use_separate_process,
                                    offload=offload)

        (parameter_count, peak_mem, latencies, tflops, metadata) = result
        # # NOTE: only WResNet is static tuple, GPT&BERT computed by func
        # if model_type == "wresnet":
        #     params_list = suite_wresnet.wresnet_params[tuple(model_config)]
        #     parameter_count, tflops = params_list
        # elif model_type == "gpt":
        #     params_list = suite_manual_gpt.gpt_params[tuple(model_config)]

        if isinstance(parallel_args, ConfigParallelArgs):
            parallel_args = parallel_args._replace(input_placement_specs=[])
        tile_r_num = global_config.wsc_config["analytical_perf_wsc::tile_r_num"]
        tile_c_num = global_config.wsc_config["analytical_perf_wsc::tile_c_num"]
        tile_num = tile_c_num * tile_r_num
        tile_FP16 = global_config.wsc_config["analytical_perf::compute_dict"][PrimitiveType.F16.value]
        tile_FP32 = global_config.wsc_config["analytical_perf::compute_dict"][PrimitiveType.F32.value]
        Peak_FP16 = tile_FP16 * tile_num / TOPS
        Peak_FP32 = tile_FP32 * tile_num / TOPS
        DDR_MEM = global_config.wsc_config["analytical_perf_wsc::ddr_mem"]

        values = [
            model_type, f"{parameter_count/1e9:.3f}B",
            f"{tflops}", f"{tflops/Peak_FP16*100}",
            f"{np.mean(latencies)}", f"{np.std(latencies)}",
            f"{peak_mem/GB > DDR_MEM/GB}", f"{peak_mem/GB:.5f}", f"{DDR_MEM/GB}",
            str(num_micro_batches), str(
                parallel), offload, num_gpus, str(model_config),
            parallel_args, to_str_round(metadata, 6)
        ]

        write_tsv(heads, values, output_name)
        values = [str(x) for x in values]
        result_dict = dict(zip(heads, values))
        # NOTE: generate final result performance
        with open(global_config.maping_rst_dir+"/over_all_perf.json", "w") as f:
            json.dump(result_dict, f, indent=4)
        if not global_config.full_on_hlo_analysis:
            gen_mapping_vis_result(global_config.maping_rst_dir)
        time.sleep(0.1)  # for ctrl+c to work

        table.add_data(*values)


# 自定义类型转换函数，将逗号分隔的字符串转换为列表
def str_list(string):
    if ',' in string:
        return string.split(',')
    else:
        return [string]


def int_list(string):
    if ',' in string:
        return [int(s) for s in string.split(',')]
    else:
        return [int(string)]


def get_num_hosts_and_num_devices(args, i):
    import ray
    import subprocess

    def list_gpu_info():
        """List all gpu information by calling nvidia-sim."""
        ret = subprocess.getoutput("nvidia-smi -L")
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices:
            ids = [int(x) for x in visible_devices.split(",")]
            lines = ret.split("\n")
            lines = [lines[i] for i in ids]
            ret = "\n".join(lines)
        return ret
    """Get the number of hosts and the number of devices per host for benchmark
    scripts."""
    if args.num_hosts[i] is not None or args.num_devices_per_host[i] is not None:
        assert (args.num_hosts[i] is not None and
                args.num_devices_per_host[i] is not None)
        num_hosts, num_devices_per_host = (args.num_hosts[i],
                                           args.num_devices_per_host[i])
    else:
        if hasattr(args, "local") and args.local:
            num_hosts = 1
            num_devices_per_host = list_gpu_info().count("UUID")
        else:
            ray.init(address="auto")
            num_hosts = len(ray.nodes())
            num_devices_per_host = int(
                ray.cluster_resources()["GPU"]) // num_hosts
    return num_hosts, num_devices_per_host


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite",
                        # choices=list(benchmark_suites.keys()),
                        type=str_list,
                        required=True, default="")
    parser.add_argument("--niter",
                        type=int,
                        default=3,
                        help="The number of benchmark iterations")
    parser.add_argument("--num-hosts", type=int_list, default=4)
    parser.add_argument("--num-devices-per-host", type=int_list, default=5)
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
                        dest="use_separate_process",
                        default=False)
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--only-mapping", action="store_true",
                        dest="only_mapping", default=True)
    parser.add_argument("--use-analytical-perf-model", action="store_true",
                        dest="use_analytical_perf_model", default=True)
    parser.add_argument("--rst_folder", type=str, default="")
    parser.add_argument("--hardware", type=str_list, default="wsc")
    parser.add_argument("--force_use_fp16", action="store_true")
    parser.add_argument("--use-greedy-collective-cost", action="store_true")
    parser.add_argument("--offload", action="store_true")
    args = parser.parse_args()

    heads = [
        "Type", "#Params (Billion)", "Actual TFLOPs(Per Device)", "Utilization (%)", "Mean Time (s)",
        "Std Time (s)", "Out of DDR", "Peak Mem (GB)", "DDR Mem (GB)", "#Microbatch", "3D Parallel(DP, TP, PP)",
        "Offload", "#GPU", "Model Config", "Parallel Config", "Metadata"
    ]

    import wandb
    import pandas as pd

    # 初始化W&B
    wandb.init(project='Benckmark')

    # 创建一个空的表格
    table = wandb.Table(columns=heads)

    for i in range(len(args.suite)):

        num_hosts, num_devices_per_host = get_num_hosts_and_num_devices(
            args, i)

        # set global_config, only_mapping
        global_config = get_global_config()
        # global_config.hardware = args.hardware
        # NOTE: support dojo & wsgpu config in args -- direct assign to wsc_config
        if args.hardware[i] == "dojo":
            global_config.wsc_config = global_config.dojo_config
            global_config.hardware = "wsc"
            print(f"Set DOJO config = {global_config.wsc_config}")
        elif args.hardware[i] == "wsgpu":
            global_config.wsc_config = global_config.wsgpu_config
            global_config.hardware = "wsc"
            print(f"Set SW-GPU config = {global_config.wsc_config}")
        else:
            # NOTE: origin support for GPU & TX8 WSC
            global_config.hardware = args.hardware[i]

        global_config.only_mapping = args.only_mapping

        # set whether use analytical model
        global_config.use_analytical_perf_model = args.use_analytical_perf_model
        # set mapping result save dir
        if args.rst_folder == "":
            args.rst_folder = "tmp"

        date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if args.only_mapping:
            if global_config.use_analytical_perf_model:
                actual_or_virtual = f"perf@{global_config.hardware[i]}"
            else:
                actual_or_virtual = "costmodel"
        else:
            actual_or_virtual = "actualA100"
        args.rst_folder = f"{args.rst_folder}/{args.suite[i]}-{num_devices_per_host}X{num_hosts}-{actual_or_virtual}-{date_str}"
        print(args.rst_folder)
        os.makedirs(args.rst_folder, exist_ok=True)

        global_config.rst_folder = args.rst_folder

        # set whether use the greedy-collective cost in analytical model
        global_config.wsc_config["analytical_perf::use_greedy_coll_cost"] = args.use_greedy_collective_cost
        if args.use_greedy_collective_cost:
            get_collective_cost_dict()

        global_config.force_use_fp16 = args.force_use_fp16

        set_global_config(global_config)
        global_config = get_global_config()
        print(global_config.use_analytical_perf_model)

        benchmark_suite(args.suite[i], num_hosts, num_devices_per_host, args.exp_name,
                        args.niter, args.shard_only, args.local,
                        args.profile_driver_time, args.disable_tqdm,
                        args.use_separate_process, heads, table, args.offload)

    # 将表格保存到W&B
    wandb.log({"Wafer": table})
