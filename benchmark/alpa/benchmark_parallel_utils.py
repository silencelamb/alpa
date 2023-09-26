"""Options of a benchmark case."""
import time
from typing import Iterable, Optional, Dict, Any
from collections import namedtuple
import os

import numpy as np
import pandas as pd

import jax
from jax._src.tree_util import tree_flatten, tree_leaves, tree_unflatten
from jax._src.lib import xla_bridge as xb

import alpa
from alpa import (AutoShardingOption, ShardParallel, PipeshardParallel, ConfigParallel,
                  ManualStageOption, WSCManualStageOption, AutoStageOption, AutoLayerOption,
                  global_config, PhysicalDeviceMesh)
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.pipeline_parallel.stage_profiling import HloAnalysisSimulator
from alpa.timer import timers
from alpa.util import (print_used_time, to_str_round,
                       count_communication_primitives, GB,
                       get_submesh_physical_shapes)
from alpa.global_env import get_global_config
from alpa.pipeline_parallel.pipeshard_executable import PipeshardDriverExecutable
from alpa.shard_parallel.auto_sharding import run_backend_compilation
from alpa.mesh_executable import get_grad_sync_channel_ids
from alpa.mesh_profiling import hlo_module_cost_analysis, estimate_hlo_module_cost

BenchmarkCase = namedtuple("BenchmarkCase", [
    "batch_size", "model_config", "num_micro_batches", "parallel_mode",
    "parallel_args"
])

ShardParallelArgs = namedtuple("ShardParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "logical_mesh_shape",
    "force_batch_dim_mapping"
])

UniformParallelArgs = namedtuple("UniformParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "dp", "op", "pp",
    "force_batch_dim_mapping"
])

SearchParallelArgs = namedtuple("SearchParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "num_auto_layers", "auto_stage_option"
])

LoadSolutionParallelArgs = namedtuple("LoadSolutionParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "num_auto_layers",
    "forward_stage_layer_ids", "submesh_physical_shapes",
    "submesh_logical_shapes", "submesh_autosharding_option_dicts"
])

ConfigParallelArgs = namedtuple("ConfigParallelArgs", [
    "stage_num", "input_placement_specs", "partition_index", "pipeline_schedule", "stage_option", "use_remat"
])



def get_pipeshard_parallel_method(benchmark_case: BenchmarkCase,
                                  num_devices_per_host: Optional[int] = None,
                                  allow_mixed_mesh_shape: bool = False,
                                  use_fine_grained_remat: bool = False,
                                  pipeline_schedule: str = "1f1b"):
    """Create the parallel method of a benchmark case.

    Args:
        benchmark_case: The benchmark case.
        num_devices_per_host: The number of devices per host, used in uniform
          parallel mode.
        allow_mixed_mesh_shape: Whether to allow the mixed mesh shape in
          the autosharding pass.
        use_fine_grained_remat: Whether to use fine grained remat. If True,
          the remat pass in auto layer pass will be skipped. This option only
          works for load_solution parallel mode now.
    """

    num_micro_batches = benchmark_case.num_micro_batches
    parallel_mode = benchmark_case.parallel_mode
    parallel_args = benchmark_case.parallel_args

    if parallel_mode == "search":
        assert isinstance(parallel_args, SearchParallelArgs)
        (prefer_reduce_scatter, use_remat, num_auto_layers,
         auto_stage_option) = parallel_args
        add_manual_layer_marker = None
        num_manual_pipeline_stages = None
        add_manual_remat = None
        auto_stage_option["cached_compute_cost"] = None
        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter,
                allow_mixed_mesh_shape=allow_mixed_mesh_shape,
            ),
            pipeline_schedule=pipeline_schedule,
            layer_option=AutoLayerOption(layer_num=num_auto_layers,
                                         remat_layer=use_remat),
            stage_option=AutoStageOption(**auto_stage_option))
    elif parallel_mode == "load_solution":
        assert isinstance(parallel_args, LoadSolutionParallelArgs)
        (prefer_reduce_scatter, use_remat, num_auto_layers,
         forward_stage_layer_ids, submesh_physical_shapes,
         submesh_logical_shapes,
         submesh_autosharding_option_dicts) = parallel_args
        add_manual_layer_marker = None
        num_manual_pipeline_stages = None
        add_manual_remat = None
        if use_fine_grained_remat:
            use_remat = False
        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter,
                allow_mixed_mesh_shape=allow_mixed_mesh_shape,
            ),
            pipeline_schedule=pipeline_schedule,
            layer_option=AutoLayerOption(layer_num=num_auto_layers,
                                         remat_layer=use_remat),
            stage_option=ManualStageOption(forward_stage_layer_ids,
                                           submesh_physical_shapes,
                                           submesh_logical_shapes,
                                           submesh_autosharding_option_dicts))
    elif parallel_mode == "uniform":
        assert isinstance(parallel_args, UniformParallelArgs)
        (prefer_reduce_scatter, use_remat, dp, op, pp,
         force_batch_dim_mapping) = parallel_args
        as_option = AutoShardingOption(
            prefer_reduce_scatter=prefer_reduce_scatter,
            allow_mixed_mesh_shape=allow_mixed_mesh_shape,
        )
        if force_batch_dim_mapping:
            as_option.force_batch_dim_to_mesh_dim = 0
        add_manual_layer_marker = True
        add_manual_remat = use_remat

        logical_mesh_shape = (dp, op)
        num_manual_pipeline_stages = pp
        num_mesh_devices = np.prod(logical_mesh_shape)
        assert num_devices_per_host is not None
        if num_mesh_devices <= num_devices_per_host:
            physical_mesh_shape = (1, num_mesh_devices)
        else:
            assert num_mesh_devices % num_devices_per_host == 0
            physical_mesh_shape = (num_mesh_devices // num_devices_per_host,
                                   num_devices_per_host)

        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=as_option,
            pipeline_schedule=pipeline_schedule,
            layer_option="manual",
            stage_option=ManualStageOption(
                forward_stage_layer_ids=[[i] for i in range(pp)],
                submesh_physical_shapes=[physical_mesh_shape] * pp,
                submesh_logical_shapes=[logical_mesh_shape] * pp,
                submesh_autosharding_option_dicts=[{}] * pp))
    elif parallel_mode == "config":
        assert isinstance(parallel_args, ConfigParallelArgs)
        stage_num, input_placement_specs, partition_index, pipeline_schedule, stage_option, _ = parallel_args

        if isinstance(stage_option, WSCManualStageOption) and stage_option.submesh_physical_shapes is None:
            stage_option: WSCManualStageOption
            stage_option.submesh_physical_shapes = get_submesh_physical_shapes(stage_option.submeshes)
            stage_option.submesh_logical_shapes = stage_option.submesh_physical_shapes
        method = ConfigParallel(
            stage_num=stage_num,
            input_placement_specs=input_placement_specs,
            partition_index=partition_index,
            pipeline_schedule = "1f1b",
            stage_option = stage_option,
            num_micro_batches=num_micro_batches)
        add_manual_layer_marker = None
        add_manual_remat = None
        num_manual_pipeline_stages = stage_num

    else:
        raise ValueError(f"Invalid parallel mode: {parallel_mode}")

    return (method, add_manual_remat, add_manual_layer_marker,
            num_manual_pipeline_stages)


def get_shard_parallel_method(benchmark_case: BenchmarkCase,
                              physical_mesh: PhysicalDeviceMesh,
                              logical_mesh_options: Dict[str, Any] = None):
    """Create the parallel method of a benchmark case.

    Args:
        benchmark_case: The benchmark case.
        num_devices_per_host: The number of devices per host, used in uniform
          parallel mode.
        allow_mixed_mesh_shape: Whether to allow the mixed mesh shape in
          the autosharding pass.
    """
    print_used_time(None)

    num_micro_batches = benchmark_case.num_micro_batches
    parallel_mode = benchmark_case.parallel_mode
    parallel_args = benchmark_case.parallel_args

    if isinstance(parallel_args, ShardParallelArgs):
        (prefer_reduce_scatter, use_remat, logical_mesh_shape,
         force_batch_dim_mapping) = parallel_args
    elif isinstance(parallel_args, UniformParallelArgs):
        (prefer_reduce_scatter, use_remat, dp, op, pp,
         force_batch_dim_mapping) = parallel_args
        assert pp == 1, "Do not support pipeline parallelism for shard parallel"
        logical_mesh_shape = (dp, op)
    else:
        raise ValueError(f"Unsupported parallel mode: {parallel_mode}")

    # Parallel configs
    if num_micro_batches > 1:
        grad_func = alpa.grad
    else:
        num_micro_batches = None
        grad_func = jax.grad

    as_option = AutoShardingOption()
    if force_batch_dim_mapping:  # Always map batch dim to mesh dim 0
        as_option.force_batch_dim_to_mesh_dim = 0
    as_option.prefer_reduce_scatter = prefer_reduce_scatter
    if parallel_mode == "zero-3":
        as_option.force_zero_stage_3 = True
    elif parallel_mode in ["shard-largest"]:
        as_option.force_simple_heuristic = "largest"
        global_config.remat_using_while = True

    if logical_mesh_options is None:
        logical_mesh_options = {}
    logical_mesh = physical_mesh.get_logical_mesh(logical_mesh_shape,
                                                  **logical_mesh_options)
    method = ShardParallel(devices=logical_mesh,
                           num_micro_batches=num_micro_batches,
                           auto_sharding_option=as_option)
    print_used_time("Setup device mesh")

    return method, grad_func


def benchmark_training_executable(niter,
                                  train_step,
                                  executable,
                                  state,
                                  other_train_step_inputs,
                                  profile_driver_time=False):
    print_used_time(None)

    # Benchmark step time
    warmup = 2 if niter >= 5 else 1

    if profile_driver_time:
        # Benchmark latency with driver overhead
        global_config.use_dummy_value_for_benchmarking = False
        global_config.shard_parallel_sync_for_timer = False
        print("Warmup")
        for i in range(warmup):
            state = train_step(state, *other_train_step_inputs)
        executable.sync()
        niter -= warmup
        print("Benchmark")
        tic = time.time()
        for i in range(niter):
            state = train_step(state, *other_train_step_inputs)
        executable.sync()
        e2e_latency = (time.time() - tic) / niter
        latencies = [e2e_latency]
        print(f"latency with dirver overhead: {e2e_latency:.3f}")
    else:
        # Benchmark latency without driver overhead
        for i in range(niter):
            print(f"Iteration {i} ...")
            state = train_step(state, *other_train_step_inputs)
            if isinstance(state, tuple):
                # In case the train_step returns extra info (e.g. loss),
                # Get the actual state out.
                state = state[0]
            executable.sync()

        latencies = executable.get_execution_time_costs()[warmup:]

    print_used_time("Benchmark")

    return latencies


def benchmark_inference_executable(niter,
                                   infer_step,
                                   executable,
                                   params,
                                   other_infer_step_inputs,
                                   profile_driver_time=False):
    print_used_time(None)

    # Benchmark step time
    warmup = 2 if niter >= 5 else 1

    if profile_driver_time:
        global_config.pipeline_check_alive = False
        # Benchmark latency with streaming
        for i in range(warmup):
            _ = infer_step(params, *other_infer_step_inputs)
        executable.sync()
        niter -= warmup

        # Benchmark latency
        losses = []
        start_time = time.time()
        latencies = []
        for i in range(niter):
            print(f"Iteration {i} ...")
            loss = infer_step(params, *other_infer_step_inputs)
            loss.get_remote_buffers_async()
            losses.append(loss)
        for i, loss in enumerate(losses):
            _ = loss._value
            end_time = time.time()
            latencies.append(end_time - start_time)
            start_time = end_time
    else:
        for i in range(niter):
            print(f"Iteration {i} ...")
            _ = infer_step(params, *other_infer_step_inputs)
            executable.sync()

        latencies = executable.get_execution_time_costs()[warmup:]

    print_used_time("Benchmark")

    return latencies


def compile_pipeshard_executable(parallel_mode, train_step, state,
                                 other_train_step_inputs):
    print_used_time(None)
    global_config = get_global_config()
    # import ipdb; ipdb.set_trace()
    executable = train_step.get_executable(state, *other_train_step_inputs)
    print_used_time("Compile (driver)")
    if parallel_mode == "search":
        compilation_times = {
            k: timers(k).elapsed() for k in [
                "stage-construction", "stage-construction-dp",
                "stage-construction-compilation", "stage-construction-profiling"
            ]
        }
        print(
            f"compilation time breakdown: {to_str_round(compilation_times, 2)}")
    else:
        compilation_times = None
    if global_config.full_on_hlo_analysis:
        executable: HloAnalysisSimulator
    else:
        executable: PipeshardDriverExecutable
        # save mapping result
        save_file = f"{global_config.maping_rst_dir}/input_placement_specs.pkl"
        input_placement_specs = executable.input_placement_specs
        with open(save_file, 'wb') as f:
            import pickle
            pickle.dump(input_placement_specs, f)

        input_placement_specs = executable.get_input_placement_specs()
        for idx, specs in enumerate(input_placement_specs):
            save_file = f"{global_config.maping_rst_dir}/input_placement_specs-{idx}.txt"
            with open(save_file, 'w') as f:
                f.write(str(specs))
        

        output_placement_specs = executable.get_output_placement_specs()
        from collections.abc import Iterable
        if isinstance(output_placement_specs, Iterable):
            for idx, specs in enumerate(output_placement_specs):
                save_file = f"{global_config.maping_rst_dir}/output_placement_specs-{idx}.txt"
                with open(save_file, 'w') as f:
                    f.write(str(specs))
        else:
            save_file = f"{global_config.maping_rst_dir}/output_placement_specs.txt"
            with open(save_file, 'w') as f:
                f.write(str(output_placement_specs))
    
    # general result
    schedule_str = executable.schedule.pprint_schedule()
    num_clock = executable.schedule.num_clock
    save_file = f"{global_config.maping_rst_dir}/pprint_schedule_num_clock-{num_clock}.txt"
    with open(save_file, 'w') as f:
        f.write(schedule_str)

    save_file = f"{global_config.maping_rst_dir}/mesh_stage_mapping.txt"
    with open(save_file, 'w') as f:
        _str = str(executable.schedule.mesh_stage_mapping)
        f.write(_str+'\n')
        _str = str(executable.schedule.stage_mesh_mapping)
        f.write(_str)   

    

    if not global_config.only_mapping and not global_config.full_on_hlo_analysis:
        executable.dump_debug_info(global_config.maping_rst_dir)
        executable.sync()
    print_used_time("Compile (worker)")
    return executable, compilation_times


def compile_shard_executable(physical_mesh, train_step, state,
                             other_train_step_inputs):
    print_used_time(None)
    executable = train_step.get_executable(state, *other_train_step_inputs)
    print_used_time("Compile (driver)")

    physical_mesh.sync_workers()
    print_used_time("Compile (workers)")

    # Check sharding strategy
    alloc_mem = executable.get_total_allocation_size()
    ilp_objective = executable.auto_sharding_objective or 0.0
    executable.dump_debug_info("tmp")
    hlo_text = executable.get_hlo_text()
    (n_total, n_all_reduce, n_all_gather, n_reduce_scatter,
     n_all_to_all) = count_communication_primitives(hlo_text)

    print(f"#total: {n_total}, #all-reduce: {n_all_reduce}, "
          f"#all-gather: {n_all_gather}, #reduce-scatter: {n_reduce_scatter}, "
          f"#all-to-all: {n_all_to_all}")
    print(f"alloc_mem: {alloc_mem / GB:.2f} GB")
    return executable, ilp_objective, alloc_mem


def compute_network_anaysis(executable: PipeshardDriverExecutable):
    global_config = get_global_config()
    xla_stages = executable.stages
    estimated_cost_sum = 0   # every op sum
    estimated_cost = 0 # result for compare, using another api
    max_mem = 0
    max_stage_cost = 0
    for idx, xla_computations in enumerate(xla_stages):
        """
        Reference code: alpa/pipeline_parallel/stage_profiling.py
        HloCostModelProfileWorker.compile()
        """
        # import ipdb; ipdb.set_trace()
        sharding_annotated_module = xla_computations.get_hlo_text()
        with open(f"{global_config.maping_rst_dir}/compute_network_anaysis_stage_{idx}_annotated.hlo", 'w') as f:
            f.write(sharding_annotated_module)
        spmd_partitioned_hlo = xla_computations.get_spmd_partitioned()
        with open(f"{global_config.maping_rst_dir}/compute_network_anaysis_stage_{idx}_spmd_partitioned.hlo", 'w') as f:
            f.write(spmd_partitioned_hlo.to_string())
        stage_plan = xla_computations.stage_plan
        logical_mesh_shape = stage_plan.logical_mesh_shape
        num_devices = np.prod(logical_mesh_shape)
        compiled = run_backend_compilation(
            xb.get_backend("gpu"),
            spmd_partitioned_hlo,
            stage_plan,
            num_devices,
            bypass_device_assignment_check=True)

        hlo_module = compiled.hlo_modules()[0]
        hlo_module_str = hlo_module.to_string()
        
        
        with open(f"{global_config.maping_rst_dir}/compute_network_anaysis_stage_{idx}_optimized.hlo", 'w') as f:
            f.write(hlo_module_str)
            
        grad_sync_channel_ids = ""
        if True:
            grad_sync_channel_ids = get_grad_sync_channel_ids(hlo_module)
        peak_memory = compiled.total_allocation_size()/ GB
        max_mem = max(max_mem, peak_memory)        
        estimated_cost_cur = estimate_hlo_module_cost(hlo_module, global_config, None, 1, grad_sync_channel_ids)
        estimated_cost += estimated_cost_cur
        max_stage_cost = max(max_stage_cost, estimated_cost_cur)
        analysis_result = hlo_module_cost_analysis(hlo_module, 1, grad_sync_channel_ids)        
        df = pd.DataFrame.from_dict(analysis_result)
        estimated_cost_cur = df['estimated_time'].sum()
        max_stage_cost = max(max_stage_cost, estimated_cost_cur)
        estimated_cost_sum += estimated_cost_cur
        df.to_excel(f"{global_config.maping_rst_dir}/compute_network_anaysis_stage_{idx}_peak_memory-{peak_memory: .3f}GB.xlsx")
        print(f'compute_network_anaysis: stage_{idx} peak_memory: {peak_memory: .3f} GB !!!!!!')
    
    return estimated_cost_sum, estimated_cost, max_stage_cost, max_mem

def compile_and_benchmark_pipeshard_training_executable(
        parallel_mode,
        niter,
        train_step,
        state,
        other_train_step_inputs,
        profile_driver_time=False):
    executable, compilation_times = compile_pipeshard_executable(
        parallel_mode, train_step, state, other_train_step_inputs)
   
    global_config = get_global_config()
    
    if global_config.full_on_hlo_analysis:
        executable: HloAnalysisSimulator
        estimated_time_sum, estimated_max_mem, stage_times = executable.estimate_cost_on_hlo_analysis()
        max_stage_time = max(stage_times)
        latencies = estimated_time = estimated_time_sum
        global last_dp_cost 
        last_dp_cost = estimated_time_sum
        
    else:
        # add compute and network cost analysis
        estimated_time_sum, estimated_time, max_stage_time, estimated_max_mem = compute_network_anaysis(executable)
    
    if global_config.only_mapping:
        _, _, _, _, _, dp_cost = get_last_dp_result()
        latencies = dp_cost if dp_cost is not None else estimated_time  # use dp_cost
        max_mem_allocated = estimated_max_mem
    else:
        latencies = benchmark_training_executable(
            niter,
            train_step,
            executable,
            state,
            other_train_step_inputs,
            profile_driver_time=profile_driver_time)
        max_mem_allocated = executable.mesh_group.get_max_memory_allocated()

    return latencies, max_mem_allocated, compilation_times, executable, estimated_time_sum, estimated_time, max_stage_time


def compile_and_benchmark_shard_training_executable(physical_mesh,
                                                    niter,
                                                    train_step,
                                                    state,
                                                    other_train_step_inputs,
                                                    profile_driver_time=False):
    executable, ilp_objective, alloc_mem = compile_shard_executable(
        physical_mesh, train_step, state, other_train_step_inputs)
    latencies = benchmark_training_executable(
        niter,
        train_step,
        executable,
        state,
        other_train_step_inputs,
        profile_driver_time=profile_driver_time)
    peak_mem = max(physical_mesh.get_max_memory_allocated(), alloc_mem)
    return latencies, ilp_objective, peak_mem, executable


def compile_and_benchmark_pipeshard_inference_executable(
        parallel_mode,
        niter,
        infer_step,
        params,
        other_inference_step_inputs,
        profile_driver_time=False):
    executable, compilation_times = compile_pipeshard_executable(
        parallel_mode, infer_step, params, other_inference_step_inputs)

    # Preshard params
    params_ps = executable.get_input_placement_specs()[0]
    flat_params, in_tree = tree_flatten(params)
    flat_ps = tree_leaves(params_ps)
    params = tree_unflatten(
        in_tree,
        executable.mesh_group.shard_args_to_arrays(flat_ps, flat_params))
    print_used_time("Preshard (driver)")

    latencies = benchmark_inference_executable(
        niter,
        infer_step,
        executable,
        params,
        other_inference_step_inputs,
        profile_driver_time=profile_driver_time)
    max_mem_allocated = executable.mesh_group.get_max_memory_allocated()

    return latencies, max_mem_allocated, compilation_times, executable
