"""The entry point of intra-op + inter-op parallelism benchmark."""
from ast import arg
import os
import argparse
from datetime import datetime
import time
import json
import logging

import numpy as np

from alpa.global_env import get_global_config, set_global_config, get_collective_cost_dict
from alpa.util import (write_tsv, get_num_hosts_and_num_devices, to_str_round,
                       GB)
from alpa.global_env import PrimitiveType
from gen_mapping_vis_result import gen_mapping_vis_result
from benchmark_parallel_utils import BenchmarkCase, ConfigParallelArgs

from benchmark_one_case import benchmark_one_case
import suite_auto_gpt
import suite_auto_moe
import suite_manual_gpt
import suite_manual_moe
import suite_wresnet
import suite_inference_gpt
import suite_auto_mlp
from suite_manual_gpt import gpt_specs
from alpa import ManualStageOption, WSCManualStageOption
from suite_auto_gpt import get_one_config_case_idx


from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.model.problem import Problem
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.model.individual import Individual
from pymoo.model.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.misc import find_duplicates, has_feasible
from pymoo.operators.selection.tournament_selection import compare, TournamentSelection
from pymoo.util.dominator import Dominator
from pymoo.optimize import minimize
from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.util.display import MultiObjectiveDisplay, SingleObjectiveDisplay
from pymoo.visualization.scatter import Scatter
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
import math
import numpy as np
import autograd.numpy as anp

from jax.interpreters.pxla import ShardingSpec, NoSharding, Replicated, Chunked, ShardedAxis

import logging
from logging import handlers

#外部传入 num_host (1) x num_devices_per_host
device_num = None   # for dojo  = 25； for wafer-scale GPU = 24;

# 最多切分device_num个stage
max_stage_num = device_num

model_type = None   # "gpt", "bert", "wresnet"
model_size = None   # e.g. for gpt: "350M"; for bert: "Tiny", "Large"; for wresnet: "M"

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(
            filename=filename, when=when, backupCount=backCount, encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


# =========================================================================================================
#  The problem is code for testing in GA 
# =========================================================================================================

class Hy_opt_problem_G1(Problem):
    def __init__(self):
        self.n_var = 13
        self.n_constr = 9
        self.n_obj = 1
        self.xl = anp.zeros(self.n_var)
        self.xu = anp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1])
        super(Hy_opt_problem_G1, self).__init__(n_var=self.n_var, n_obj=self.n_obj, n_constr=self.n_constr, xl=self.xl, xu=self.xu,
                                                type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x1 = x[:, 0: 4]
        x2 = x[:, 4: 13]
        f = 5 * anp.sum(x1, axis=1) - 5 * \
            anp.sum(anp.multiply(x1, x1), axis=1) - anp.sum(x2, axis=1)

        # Constraints
        g1 = 2 * x[:, 0] + 2 * x[:, 1] + x[:, 9] + x[:, 10] - 10
        g2 = 2 * x[:, 0] + 2 * x[:, 2] + x[:, 9] + x[:, 11] - 10
        g3 = 2 * x[:, 1] + 2 * x[:, 2] + x[:, 10] + x[:, 11] - 10
        g4 = -8 * x[:, 0] + x[:, 9]
        g5 = -8 * x[:, 1] + x[:, 10]
        g6 = -8 * x[:, 2] + x[:, 11]
        g7 = -2 * x[:, 3] - x[:, 4] + x[:, 9]
        g8 = -2 * x[:, 5] - x[:, 6] + x[:, 10]
        g9 = -2 * x[:, 7] - x[:, 8] + x[:, 11]

        out["F"] = f
        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9])

    def _calc_pareto_front(self):
        return -15

    def _calc_pareto_set(self):
        return [1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1]


# =========================================================================================================
# Survival
# =========================================================================================================


class FitnessSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(True)

    def _do(self, problem, pop, n_survive, out=None, **kwargs):
        F = pop.get("F")

        if F.shape[1] != 1:
            raise ValueError(
                "FitnessSurvival can only used for single objective single!")

        return pop[np.argsort(F[:, 0])[:n_survive]]


# =========================================================================================================
# Implementation
# =========================================================================================================


def comp_by_cv_and_fitness(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV,
                           method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[i] = compare(a, pop[a].F, b, pop[b].F,
                           method='smaller_is_better', return_random_if_equal=True)

    return S[:, None].astype(np.int)


class GA(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(
                     func_comp=comp_by_cv_and_fitness),
                 crossover=SimulatedBinaryCrossover(prob=0.9, eta=3),
                 mutation=PolynomialMutation(prob=None, eta=5),
                 survival=FitnessSurvival(),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 display=SingleObjectiveDisplay(),
                 **kwargs):
        """

        Parameters
        ----------
        pop_size : {pop_size}
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         display=display,
                         **kwargs)

        self.default_termination = SingleObjectiveDefaultTermination()
        

def run_in_GA(args_=None, num_hosts=None, num_devices_per_host=None, log = None):
    
    ub = []
    lb = []
    mask = []
    
    n_gen = 20 
    pop_size = 25
    
    '''
    total encoding:
        [0, 1, 0, ..., 0]   +  [int, int, ..., int]
    len:  max_stage_num-1   +  max_stage_num-1    
    '''

    # encode hardware split, binary encode, 1 for split, 0 not split
    for i in range(max_stage_num-1):
        lb.append(0)
        ub.append(1)
        mask.append("int")
    
    for i in range(max_stage_num-1):
        # partition_index for every stage
        lb.append(1)
        ub.append(10)
        mask.append("int")
    

    sampling = MixedVariableSampling(mask, {
        "real": get_sampling("real_random"),
        "int": get_sampling("int_random")
    })

    crossover = MixedVariableCrossover(mask, {
        "real": get_crossover("real_sbx", prob=0.9, eta=3.0),
        "int": get_crossover("int_sbx", prob=0.9, eta=3.0)
    })

    mutation = MixedVariableMutation(mask, {
        "real": get_mutation("real_pm", eta=3.0),
        "int": get_mutation("int_pm", eta=3.0)
    })   
    
    method = GA(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )
    log.logger.info('GA n_gen ' + str(n_gen))
    log.logger.info('GA pop_size ' + str(pop_size))
    problem_GA = GA_problem_alpa(n_var=len(lb), n_obj=1, lb=lb, ub=ub, args_=args_,
                                 num_hosts=1, num_devices_per_host=device_num, log=log)

    res = minimize(problem_GA,
                   method,
                   termination=('n_gen', 5),
                   seed=1,
                   verbose=False)
    
    print(res.X)
    print(res.F)
    
    log.logger.info('GA result : F' + str(res.F))
    log.logger.info('GA result : X' + str(res.X))


class GA_problem_alpa(Problem):
    def __init__(self, n_var=6, n_obj=1, n_constr=0, lb=None, ub=None, save_dir=None, args_=None, num_hosts=None, num_devices_per_host=None, log = None):
        self.xl = lb
        self.xu = ub        
        self._save_dir = save_dir
        self._n_evaluated = 0
        self.args_ = args_
        self.num_hosts = num_hosts
        self.num_devices_per_host = num_devices_per_host
        self.log = log
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=lb, xu=ub)

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        for i in range(x.shape[0]):
            cls_train = get_alpa_value(self.args_, self.num_hosts, self.num_devices_per_host, paras_list=x[i],log= self.log)
            objs[i, 0] = cls_train            
        out["F"] = objs
        self._n_evaluated += 1


def get_alpa_value(args_, num_hosts, num_devices_per_host, paras_list=None, log = None):
        
    try:
        log.logger.info('paras_list: ' + str(paras_list))
        result_ = benchmark_suite(args_.suite, 
                                  num_hosts, 
                                  num_devices_per_host, 
                                  args_.exp_name,
                                  args_.niter, 
                                  args_.shard_only, 
                                  args_.local,
                                  args_.profile_driver_time, 
                                  args_.disable_tqdm,
                                  args_.use_separate_process, 
                                  parameters_list=paras_list,
                                  use_offload=args_.use_offload,
                                  constrain_mem=args_.constrain_mem,
                                  log =log
                                  )     
    except Exception as e:
        log.logger.warning(f"Wrong !!!!===========================================================")
        print(e)
        result_ = 10e10
    print("result_ : "+str(result_))
    log.logger.info(str(paras_list) + ' result : ' + str(result_))
    log.logger.info('One Mid Result : ' + str(result_))
    return result_ 

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
                    parameters_list = None,
                    use_offload = False,
                    constrain_mem = False,
                    log = None):
    
    num_gpus = num_hosts * num_devices_per_host

    # get stage num and device_num of per stage
    device_cur_stage = []
    device_per_stage = []
    for i in range(max_stage_num-1):
        device_cur_stage.append(i)
        # 1 for split
        if parameters_list[i] == 1:
            device_per_stage.append(device_cur_stage)
            device_cur_stage = []
        # the last one device (max_stage_num - 1)
        if i == max_stage_num - 1 - 1:
            device_cur_stage.append(i+1)
            device_per_stage.append(device_cur_stage)
        
    stage_num = len(device_per_stage)
    
    # construct stage_option
    forward_stage_layer_ids = []
    submesh_autosharding_option_dicts =[]
    submeshes = []
    device_start = 0
    for i in range(stage_num):
        forward_stage_layer_ids.append([i])
        submesh_autosharding_option_dicts.append({})
        device_end = device_start + len(device_per_stage[i]) -1   
        submeshes.append([device_start, 0, device_end, 0])
        device_start = device_end + 1
    stage_option=WSCManualStageOption(
        forward_stage_layer_ids=forward_stage_layer_ids,
        submeshes=submeshes,
        submesh_physical_shapes=None,
        submesh_logical_shapes=None,
        submesh_autosharding_option_dicts=submesh_autosharding_option_dicts
    )

    # graph partition by ratio
    partition_index = parameters_list[max_stage_num-1: max_stage_num-1+stage_num]
    partition_index_sum = sum(partition_index)
    # convert to ratio
    partition_index = [sum(partition_index[:i])/partition_index_sum for i in range(len(partition_index))]
        
    if model_type == "gpt":
        suite = get_one_config_case_idx(
            gpt_specs[model_size], 
            [100],
            partition_index=partition_index,
            stage_option=stage_option
        )
    elif model_type == "bert":
        # add corresponding bert suite
        pass
    elif model_type == "wresnet":
        # add corresponding wresnet suite
        pass                        
                            
    log.logger.info('one pop: ' + str(parameters_list) +
                    'suite: ' + str(suite))
    result_latency = 10e10
    
    # Run all cases
    for benchmark_case in suite:
        benchmark_case: BenchmarkCase
        totol_batch_size = benchmark_case.batch_size
        model_config = benchmark_case.model_config
        num_micro_batches = benchmark_case.num_micro_batches
        parallel_args = benchmark_case.parallel_args
        try:
            auto_layers = benchmark_case.parallel_args.num_auto_layers
        except AttributeError:
            auto_layers = 'auto'

        # Run one case
        print("Working on case: {}".format(str(benchmark_case)))
        try:
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
                                        offload = use_offload
                                        )
            (parameter_count, peak_mem, latencies, tflops, metadata) = result

            heads = [
                "Type", "#Params (Billion)", "Actual TFLOPs(Per Device)", "Utilization (%)","Mean Time (s)", 
                "Std Time (s)", "Out of DDR", "Peak Mem (GB)", "DDR Mem (GB)", "#Microbatch", 
                "#GPU", "Model Config", "Parallel Config", "Metadata"
            ]
            if isinstance(parallel_args, ConfigParallelArgs):
                parallel_args = parallel_args._replace(input_placement_specs=[])
                
            Peak_FP16 = global_config.wsc_config["analytical_perf::compute_dict"][PrimitiveType.F16.value]
            Peak_FP32 = global_config.wsc_config["analytical_perf::compute_dict"][PrimitiveType.F32.value]
            DDR_MEM = global_config.wsc_config["analytical_perf_wsc::ddr_mem"]
            out_of_mem = peak_mem/GB > DDR_MEM/GB
            values = [
                model_type, f"{parameter_count/1e9:.3f}B",
                f"{tflops:.4f}", f"{tflops/Peak_FP16*100}",
                f"{np.mean(latencies):.5f}", f"{np.std(latencies):.5f}",
                f"{out_of_mem}", f"{peak_mem/GB:.5f}", f"{DDR_MEM/GB}",
                str(num_micro_batches), num_gpus, str(model_config),
                parallel_args, to_str_round(metadata, 6)
            ]
            values = [str(x) for x in values]
            result_dict = dict(zip(heads, values)) 
            log.logger.info('One result: ' + str(result_dict))
            if out_of_mem and constrain_mem:
                result_latency = 10e10
            else:
                result_latency = metadata['estimated_total_time'] 
        except RuntimeError:
            log.logger.error("alpa runtime error !!!")
            result_latency = 10e10
            
        
        
    return result_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter",
                        type=int,
                        default=3,
                        help="The number of benchmark iterations")
    parser.add_argument("--suite", type=str, default="GAmapping")
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
    parser.add_argument(
        "--only-mapping", action="store_true", dest="only_mapping")
    parser.add_argument("--use-analytical-perf-model",
                        action="store_true", dest="use_analytical_perf_model")
    parser.add_argument("--rst_folder", type=str, default="")
    parser.add_argument("--hardware", type=str, choices=['gpu', 'wsc', 'dojo', 'wsgpu'], 
                        default="gpu")
    parser.add_argument("--force_use_fp16", action="store_true")
    parser.add_argument("--use-greedy-collective-cost", action="store_true")
    parser.add_argument("--model-type", type=str, choices=['gpt', 'bert', 'wresnet'],
                        help="set model type, only support gpt, bert, wresnet")
    parser.add_argument("--model-size", type=str, default=None,
                        help="set model size")
    parser.add_argument("--constrain-mem", action="store_true",
                        help="if the peak mem exceeds the device mem, it is invalid, then set the latency to 10e10")
    parser.add_argument("--use-offload", action="store_true",
                        help="use offload strategy (zero offoad -like)")
    
    args = parser.parse_args()
    # when use use offload strategy, the memory contraint is off
    if args.use_offload:
        args.constrain_mem = False
    num_hosts, num_devices_per_host = get_num_hosts_and_num_devices(args)
    model_type = args.model_type
    model_size = args.model_size

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
            actual_or_virtual = f"perf@{args.hardware}"
        else:
            actual_or_virtual = "costmodel"
    else:
        actual_or_virtual = "actualA100"

    if args.hardware == "dojo":
        global_config.wsc_config = global_config.dojo_config
        global_config.hardware = "wsc"
        print(f"Set DOJO config = {global_config.wsc_config}")
    elif args.hardware == "wsgpu":
        global_config.wsc_config = global_config.wsgpu_config
        global_config.hardware = "wsc"
        print(f"Set SW-GPU config = {global_config.wsc_config}")
    else:
        # NOTE: origin support for GPU & TX8 WSC
        global_config.hardware = args.hardware
        
    # set device num
    assert num_hosts == 1, ("Only support 1 host now.")
    device_num = num_devices_per_host *num_hosts
    max_stage_num = device_num
    
    args.rst_folder = f"{args.rst_folder}/{args.suite}-{actual_or_virtual}-{num_devices_per_host}X{num_hosts}-{model_type}-{model_size}-{date_str}"
    print(args.rst_folder)
    os.makedirs(args.rst_folder, exist_ok=True)

    global_config.rst_folder = args.rst_folder
    global_config.hardware = args.hardware
    global_config.force_use_fp16 = args.force_use_fp16

    global_config.wsc_config["analytical_perf::use_greedy_coll_cost"] = args.use_greedy_collective_cost
    if args.use_greedy_collective_cost:
        get_collective_cost_dict()
        
    set_global_config(global_config)
    global_config = get_global_config()
    print(global_config.use_analytical_perf_model)
    save_path = args.rst_folder
    log_format = '%(asctime)s %(filename)s %(levelname)s %(message)s'
    log = Logger(os.path.join(save_path, 'log.log'), level='info')
    log.logger.info('begin search by GA')
        
    run_in_GA(args, num_hosts, num_devices_per_host, log)
