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
import suite_auto_mlp


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
        

def run_in_GA(args_=None, num_hosts=None, num_devices_per_host=None):
    
    ub = []
    lb = []

    # initchannels
    lb.append(8)
    ub.append(80)

    # normal1
    lb.append(1)
    ub.append(6)

    # normal2
    lb.append(1)
    ub.append(6)

    # normal3
    lb.append(1)
    ub.append(6)

    # channels 1
    lb.append(1)
    ub.append(3)

    # channels 2
    lb.append(1)
    ub.append(3)
    mask = ["int", "int", "int", "int", "real", "real"]

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
        pop_size=100,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    problem_GA = GA_problem_alpa(n_var=7,n_obj=1, args_=args_, num_hosts=num_hosts,num_devices_per_host=num_devices_per_host)

    res = minimize(problem_GA,
                   method,
                   termination=('n_gen', 100),
                   seed=1,
                   verbose=False)
    
    print(res.X)
    print(res.F)


class GA_problem_alpa(Problem):
    def __init__(self, n_var=7, n_obj=1, n_constr=0, lb=None, ub=None, save_dir=None, args_=None, num_hosts=None, num_devices_per_host=None):
        self.xl = lb
        self.xu = ub        
        self._save_dir = save_dir
        self._n_evaluated = 0
        self.args_ = args_
        self.num_hosts = num_hosts
        self.num_devices_per_host = num_devices_per_host
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=lb, xu=ub)

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        for i in range(x.shape[0]):
            cls_train = get_alpa_value(self.args_, self.num_hosts, self.num_devices_per_host, paras_list=x[i])
            objs[i, 0] = cls_train            
        out["F"] = objs
        self._n_evaluated += 1
        

def get_alpa_value(args_, num_hosts, num_devices_per_host, paras_list=None):
    import pickle
    result_ = 10e10
            
    path__pkl = "/zhanghaichao/lab2/alpa/benchmark/alpa/data/tmp_22_gpu_analytical/mlp.grid_search_auto-4X1-perf@gpu-2023-06-14-05-45-36/Batchsize_1024-num_b_128-auto_layers_1/input_placement_specs.pkl"
    input_placement_specs = None
    with open(path__pkl, 'rb') as f:
        input_placement_specs = pickle.load(f)
    
    new_input_placement_specs = []
    num_current = 0
    for item in input_placement_specs:
        new_item = None
        if item is not None:
            new_item = item
            if num_current < 2:
                new_item.mesh_ids = (0,)
            elif num_current >= 2 and num_current < 4:
                new_item.mesh_ids = (1,)
            elif num_current >= 4 and num_current < 5:
                new_item.mesh_ids = (2,)
            elif num_current >= 5 and num_current < 10:
                new_item.mesh_ids = (0,)
            elif num_current >= 10 and num_current < 12:
                new_item.mesh_ids = (1,)
            else:
                new_item.mesh_ids = (2,)
        else:
            new_item = None
        num_current = num_current + 1
        new_input_placement_specs.append(new_item)

    save_file = f"zhc_test/input_placement_specs_mlp.pkl"
    input_placement_specs = new_input_placement_specs
    
    with open(save_file, 'wb') as f:
        # import pickle
        pickle.dump(input_placement_specs, f)
    
    try:
        result_ = benchmark_suite(args_.suite, num_hosts, num_devices_per_host, args_.exp_name,
                        args_.niter, args_.shard_only, args_.local,
                        args_.profile_driver_time, args.disable_tqdm,
                        args_.use_separate_process)
        
    except:
        result_ = 10e10
    
    return result_ 
    
benchmark_suites = {
    "gpt.tmp": suite_manual_gpt.tmp_suite,
    "gpt.tmp_auto": suite_auto_gpt.tmp_suite,
    "gpt.perf_test_fast_2d": suite_manual_gpt.perf_test_fast_2d_suite,
    "gpt.perf_test_manual": suite_manual_gpt.perf_test_suite,
    "gpt.perf_test_auto": suite_auto_gpt.perf_test_suite,
    "gpt.grid_search_auto": suite_auto_gpt.grid_search_suite,
    "mlp.grid_search_auto": suite_auto_mlp.grid_search_suite_mlp,
    "gpt.correctness_test_auto": suite_auto_gpt.correctness_test_suite,
    "gpt_inference.profile": suite_inference_gpt.profile_suite,
    "gpt_no_embedding_inference.profile": suite_inference_gpt.profile_suite,
    "gpt.config_test": suite_auto_gpt.config_test_suite,
    "gpt.wsc_config_test": suite_auto_gpt.wsc_config_test_suite,
    "mlp.wsc_config_test": suite_auto_mlp.wsc_config_test_suite_mlp,
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

    result_latency = 5e10
    
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

        result_latency = latencies
        
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
        
    return result_latency


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
    parser.add_argument(
        "--only-mapping", action="store_true", dest="only_mapping")
    parser.add_argument("--use-analytical-perf-model",
                        action="store_true", dest="use_analytical_perf_model")
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
        actual_or_virtual = "actualA100"
    args.rst_folder = f"{args.rst_folder}/{args.suite}-{num_devices_per_host}X{num_hosts}-{actual_or_virtual}-{date_str}"
    print(args.rst_folder)
    os.makedirs(args.rst_folder, exist_ok=True)

    global_config.rst_folder = args.rst_folder
    global_config.hardware = args.hardware
    global_config.force_use_fp16 = args.force_use_fp16

    set_global_config(global_config)
    global_config = get_global_config()
    print(global_config.use_analytical_perf_model)
    
    run_in_GA(args, num_hosts, num_devices_per_host)

    # benchmark_suite(args.suite, num_hosts, num_devices_per_host, args.exp_name,
    #                 args.niter, args.shard_only, args.local,
    #                 args.profile_driver_time, args.disable_tqdm,
    #                 args.use_separate_process)
