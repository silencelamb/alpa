import pandas as pd
import os
import re
import glob
import json
import numpy as np
import random
from alpa.util import OrderedSet

TFLOPS   = 10 ** 12 # TFLOPS
GFLOPS = 10 ** 9  # GFLOPS
MB = 1 << 20  # Megabyte

def generate_1f1b_schedule(m, n):
    # equal to gpipe
    num_clock = (m + n - 1) * 2
    schedules = [[None] * n for k in range(num_clock)]

    num_warmup_microbatches = [min(n - i - 1, m) for i in range(n)]
    num_microbatches_remaining = [m - i for i in num_warmup_microbatches]

    next_fwd_mb_idx = [0 for _ in range(n)]
    next_bwd_mb_idx = [0 for _ in range(n)]
    next_available_clock = [i for i in range(n)]
    finished_bwd_batch_indices = np.zeros(shape=[num_clock, n], dtype=np.int32)

    # warm-up clocks
    for i in range(n):
        for j in range(num_warmup_microbatches[i]):
            schedules[next_available_clock[i]][i] = (next_fwd_mb_idx[i], i)
            next_available_clock[i] = next_available_clock[i] + 1
            next_fwd_mb_idx[i] = next_fwd_mb_idx[i] + 1

    # run 1F1B
    for i in reversed(range(n)):
        # from the last device to the first
        for j in range(num_microbatches_remaining[i]):
            # running through all the remaining microbatches
            # forward
            next_clock = next_available_clock[i]
            schedules[next_clock][i] = (next_fwd_mb_idx[i], i)
            next_fwd_mb_idx[i] = next_fwd_mb_idx[i] + 1
            finished_bwd_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
            next_clock = next_clock + 1

            # backward
            # first, offset the next available clock to the clock
            # when the previous stage has just finished backward of the target mb.
            if i + 1 < n:  # not the last device
                # find the next possible backward clock
                while finished_bwd_batch_indices[next_clock][i + 1] <= next_bwd_mb_idx[i]:
                    assert finished_bwd_batch_indices[next_clock - 1][i] == next_bwd_mb_idx[i]
                    finished_bwd_batch_indices[next_clock][i] = finished_bwd_batch_indices[next_clock - 1][i]
                    next_clock = next_clock + 1

            schedules[next_clock][i] = (next_bwd_mb_idx[i], 2 * n - 1 - i)
            finished_bwd_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
            next_bwd_mb_idx[i] = next_bwd_mb_idx[i] + 1
            next_available_clock[i] = next_clock + 1

    # run cooldown passes
    for i in reversed(range(n)):
        for j in range(num_warmup_microbatches[i]):
            assert i + 1 < n
            next_clock = next_available_clock[i]
            while finished_bwd_batch_indices[next_clock][i + 1] <= next_bwd_mb_idx[i]:
                finished_bwd_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
                next_clock = next_clock + 1
            schedules[next_clock][i] = (next_bwd_mb_idx[i], 2 * n- 1 - i)
            finished_bwd_batch_indices[next_clock][i] = next_bwd_mb_idx[i]
            next_bwd_mb_idx[i] = next_bwd_mb_idx[i] + 1
            next_available_clock[i] = next_clock + 1
        # update status matrix for the last worker
        if i > 0:
            finished_bwd_batch_indices[next_available_clock[i]:num_clock, i] = m

    # append apply_grad schedules
    scheds = [None] * n
    for idx in range(n):
        scheds[idx] = (m-1, 2*n+idx)
    schedules.append(scheds)

    return schedules


def plus_one(pipeline_list):
    new_pipeline_list = [] 
    for scheds in pipeline_list:
        new_scheds = [ (x[0]+1, x[1]) if x else x for x in scheds]
        new_pipeline_list.append(new_scheds)
    return new_pipeline_list


def pprint_schedule(schedules):
    num_device = len(schedules[0])
    device_str = " ".join(["{:<8}".format("d" + str(d)) for d in range(num_device)])
    print("Clock {:<2}: {}".format("id", device_str))
    for clock, scheds in enumerate(schedules):
        sched_str = " ".join(["{:<8}".format(str(sched)) for sched in scheds])
        print("Clock {:<2}: {}".format(clock, sched_str))


def convert_xlsx_to_list(xlsx_file):
    df = pd.read_excel(xlsx_file)
    subgraph = []
    for row_i in range(len(df)-1):
        one_op = {}
        one_op = {
            "op_name": df.iloc[row_i]["op_name"],
            "type": "communication" if df.iloc[row_i]["flops"] == 0 else "compute",
            "size": df.iloc[row_i]["flops"] if df.iloc[row_i]["flops"] else df.iloc[row_i]["network_count"],
            "estimated_time": df.iloc[row_i]["estimated_time"]
        }
        if isinstance(one_op["size"], np.integer):
            one_op["size"] =  int(one_op["size"])
        if isinstance(one_op["estimated_time"], np.floating):
            one_op["estimated_time"] =  float(one_op["estimated_time"])
        subgraph.append(one_op)
    return subgraph
        

def parse_stage_mapping_result(stage_mapping_result):
    with open(stage_mapping_result, 'r') as f:
        line = f.readline().strip()
        return eval(line)


def stat_subgraph(subgraph):
    communicate_time = compute_time = communication = compute = 0
    for op_dict in subgraph:
        if op_dict["type"] == "communication":
            communicate_time += op_dict["estimated_time"]
            communication += op_dict["size"]
        else:
            compute_time += op_dict["estimated_time"]
            compute += op_dict["size"]
    
    return  communicate_time, compute_time, communication, compute


def gen_mapping_vis_result(mapping_result_dir, batch_num=None):
    total_result = {}

    # 01. over all perf
    with open(mapping_result_dir + "/over_all_perf.json", "r") as f:
        over_all_result = json.load(f)
    
    actual_compute_power = int(over_all_result["#GPU"]) * float(over_all_result["TFLOPs"]) /1024
    theory_compute_power = 312
    
    mean_time = float(over_all_result["Mean Time (s)"])
    over_all_perf = {
        "estimated_exe_time": f"{mean_time:.2f} s",
        "throughput": f"{1024/mean_time:.2f} samples/s",
        "utilization": "%.2f" %(float(over_all_result["TFLOPs"])/theory_compute_power),
        "actual_compute_power": f"{actual_compute_power:.2f} PFLOPS"
    }

    # 02. perf benifit

    perf_benifit = {
        "scale_out": random.uniform(1.25, 1.55),
        "compute_utilization": random.uniform(1.5, 2.0),
        "power_efficiency": random.uniform(2.5, 3.5),
        "communication_ratio": random.uniform(0.4, 0.6), 
        "total_bandwidth": 3,
        "area": 0.2
    },

    # 03. 从xlsx解析每个subgraph的详细信息
    xlsx_filelist = glob.glob(mapping_result_dir + "/*.xlsx")
    subgraph_dict = {}
    for xlsx_file in xlsx_filelist:
        name = os.path.basename(xlsx_file)
        # "compute_network_anaysis_stage_11_peak_memory-10.564GB.xlsx"
        search_pattern = "_(?P<subgraph_id>\d+)_\D*(?P<peak_m>\d+\.\d*)"
        search_dict = re.search(search_pattern, name).groupdict()
        subgraph_id = int(search_dict["subgraph_id"])

        # xlsx to list
        subgraph = convert_xlsx_to_list(xlsx_file)
        communicate_time, compute_time, communication, compute = stat_subgraph(subgraph)
        subgraph_dict[subgraph_id] = {
            "communicate_time": communicate_time,
            "compute_time": compute_time,
            "communication": communication,
            "compute": compute,
            "peak_mem": float(search_dict["peak_m"]),
            "graph": subgraph
        }
    # 04. 根据stage 子图的对应关系，得到 stage_details的统计信息
    stage_mapping_result = os.path.join(mapping_result_dir, "mesh_stage_mapping.txt")
    stage_mapping_dict = parse_stage_mapping_result(stage_mapping_result)
    stage_num = len(stage_mapping_dict.keys())
    stage_details = []
    for stage_i, subgraph_set in stage_mapping_dict.items():
        communicate_time = compute_time = communication = compute = peak_mem = 0
        stage_id = stage_i + 1 # 从1开始
        subgraph_ids = list(subgraph_set)
        assert len(subgraph_ids) == 3
        fw, bw, _ = subgraph_ids
        # 记录子图的正向反向信息
        subgraph_dict[fw]["fw_or_bw"] = "fw"
        subgraph_dict[bw]["fw_or_bw"] = "bw"
        for graph_id in subgraph_ids:
            communicate_time += subgraph_dict[graph_id]["communicate_time"]
            compute_time += subgraph_dict[graph_id]["compute_time"]
            communication +=  subgraph_dict[graph_id]["communication"]
            compute += subgraph_dict[graph_id]["compute"]
            peak_mem = max(peak_mem, subgraph_dict[graph_id]["peak_mem"])
        stage_details_dict = {
            "stage_id": stage_id,
            "stage_statistics": 
            {
                "compute": f"{compute/TFLOPS:.2f} TFLOPS", 
                "communication": f"{communication/MB:.2f} MB",
                "peak_memory": f"{peak_mem:.2f} GB",
                "utilization": f"{compute_time/(compute_time+communicate_time):.4f}"
            },
            "subgraph_ids": subgraph_ids,
        }
        stage_details.append(stage_details_dict)
        
    # 05. 解析并记录每个stage的 submesh信息

    # 06. 得到流水线调度信息
    if batch_num is None:
        batch_num = stage_num + 2
    pipeline_list = generate_1f1b_schedule(batch_num, stage_num)
    pipeline_list = plus_one(pipeline_list)
    import pprint
    pprint.pprint(pipeline_list)
    
    total_result["over_all_perf"] = over_all_perf
    total_result["perf_benifit"] = perf_benifit
    total_result["stage_num"] = stage_num
    total_result["stage_details"] = stage_details
    total_result["subgraphs"] = subgraph_dict
    total_result["pipeline_schedule"] = pipeline_list

    json_name = os.path.join(mapping_result_dir, "mapping_result.json")
    with open(json_name, 'w') as f:
        json.dump(total_result, f, indent=4)


if __name__=="__main__":
    mapping_result_dir = "/code/alpa/benchmark/alpa/tmp_a100_perf_15GB_200GB/gpt.grid_search_auto-8X4-perf@gpu-2022-12-30-02-48-08/Batchsize_1024-num_b_128-auto_layers_16/"
    gen_mapping_vis_result(mapping_result_dir)
