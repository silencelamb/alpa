"""Benchmark one case of inter-op + intra-op parallelism."""
from functools import partial

from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax

import alpa
from alpa import (parallelize, get_global_cluster,
                  set_global_virtual_physical_mesh, ShardParallel,
                  automatic_remat)
from alpa import (parallelize, get_global_cluster,
                  set_global_virtual_physical_mesh, automatic_remat,
                  global_config, set_global_option_model_type)
from alpa.model.wide_resnet import get_wide_resnet, TrainState
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.util import print_used_time, compute_param_number
from benchmark_parallel_utils import *
from alpa.global_env import get_global_config, set_global_config
from alpa.device_mesh import VirtualPhysicalMesh, get_global_virtual_physical_mesh
from util import compute_wresnet_parameter_count, compute_wresnet_tflops


def compute_metrics(logits, labels):
    metrics = {
        "loss": cross_entropy_loss(logits, labels),
        "accuracy": jnp.mean(jnp.argmax(logits, -1) == labels),
    }
    return metrics


def cross_entropy_loss(logits, labels):
    num_classes = logits.shape[-1]
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


def create_learning_rate_fn():
    """Create learning rate schedule."""
    base_learning_rate = 0.1
    warmup_epochs = 5.0
    steps_per_epoch = 10000
    num_epochs = 100.0

    warmup_fn = optax.linear_schedule(init_value=0.,
                                      end_value=base_learning_rate,
                                      transition_steps=warmup_epochs *
                                      steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,
                                            decay_steps=cosine_epochs *
                                            steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn


def create_train_state(rngkey, model, input_images, learning_rate_fn):
    params = model.init_dummy(rngkey, input_images)
    params, batch_stats = params["params"], params["batch_stats"]

    # dynamic_scale = optim.DynamicScale()
    dynamic_scale = None

    tx = optax.sgd(
        learning_rate=learning_rate_fn,
        momentum=0.9,
        nesterov=True,
    )
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              batch_stats=batch_stats,
                              dynamic_scale=None)
    return state


def get_train_step(learning_rate_fn,
                   use_remat,
                   num_remat_layers,
                   method,
                   grad_func=None):

    if grad_func is None:
        grad_func = alpa.grad

    @parallelize(method=method)
    def train_step(state, batch):

        def loss_fn(params):
            logits, new_model_state = state.apply_fn(
                {
                    "params": params,
                    "batch_stats": state.batch_stats
                },
                batch["images"],
                mutable=["batch_stats"])
            loss = cross_entropy_loss(logits, batch["labels"])
            # weight_penalty_params = jax.tree_leaves(params)
            # weight_decay = 0.0001
            # weight_l2 = sum(
            #     [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
            # weight_penalty = weight_decay * 0.5 * weight_l2
            metrics = {
                "loss": loss,
                "accuracy": jnp.mean(jnp.argmax(logits, -1) == batch["labels"]),
                "lr": learning_rate_fn(step)
            }
            return loss, (new_model_state, metrics)

        if isinstance(method, ShardParallel) and use_remat:
            loss_fn = automatic_remat(loss_fn, layer_num=num_remat_layers)

        step = state.step
        dynamic_scale = state.dynamic_scale

        if dynamic_scale:
            # TODO(lmzheng): handle gradient accumulation for this
            grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
            dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
            # dynamic loss takes care of averaging gradients across replicas
        else:
            grad_fn = grad_func(loss_fn, has_aux=True)
            grads, aux = grad_fn(state.params)
        new_model_state, metrics = aux

        new_state = state.apply_gradients(
            grads=grads, batch_stats=new_model_state["batch_stats"])
        if dynamic_scale:
            # if is_fin == False the gradients contain Inf/NaNs and optimizer
            # state and params should be restored (= skip this step).
            new_state = new_state.replace(
                opt_state=jax.tree_multimap(partial(jnp.where, is_fin),
                                            new_state.opt_state,
                                            state.opt_state),
                params=jax.tree_multimap(partial(jnp.where, is_fin),
                                         new_state.params, state.params))
            metrics["scale"] = dynamic_scale.scale

        return new_state, metrics

    return train_step


def prepare_wresnet_input_and_model(benchmark_case):
    print_used_time(None)
    # Model configs
    (batch_size, model_config, num_micro_batches, parallel_mode,
     parallel_args) = benchmark_case
    (image_size, num_layers, num_channels, width_factor, dtype) = model_config
    if dtype == "fp32":
        dtype = jnp.float32
    elif dtype == "fp16":
        dtype = jnp.float16
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    # Prepare input batch
    num_classes = 1024
    batch = {
        "images":
            jnp.ones((batch_size, image_size, image_size, 3), dtype=dtype),
        "labels":
            jnp.ones((batch_size), dtype=jnp.int32),
    }
    print_used_time("Prepare input")

    # Init train state
    model = get_wide_resnet(num_layers, width_factor, num_channels, num_classes,
                            dtype)

    rngkey = jax.random.PRNGKey(0)
    learning_rate_fn = create_learning_rate_fn()
    state = create_train_state(rngkey, model, batch["images"], learning_rate_fn)
    print_used_time("Create train state")
    return state, batch, learning_rate_fn, rngkey



def compute_wresnet_statistics(benchmark_case, latencies, num_devices):
    batch_size = benchmark_case.batch_size
    total_tflops = 0.0
    
    (image_size, num_layers, num_channels, width_factor, dtype) \
        = benchmark_case.model_config  
    use_remat = benchmark_case.parallel_args.use_remat
    tflops, total_tflops = compute_wresnet_tflops(batch_size,
                                image_size,
                                num_layers,
                                num_channels,
                                width_factor,
                                num_devices,
                                np.mean(latencies),
                                checkpoint_activations=use_remat)
    parameter_count = compute_wresnet_parameter_count(
            image_size, num_layers, num_channels, width_factor)
    
    
    
    return tflops, parameter_count, total_tflops



def benchmark_wresnet_3d_internal(model_type,
                                  benchmark_case: BenchmarkCase,
                                  niter,
                                  num_hosts,
                                  num_devices_per_host,
                                  profile_driver_time=False,
                                  offload=False,):

    # global config
    global_config = get_global_config()
    set_global_option_model_type(model_type)       
    # Connect to the cluster
    if global_config.only_mapping:    
        from alpa import  WSCManualStageOption    
        g_vir_phy_mesh = get_global_virtual_physical_mesh()
        
        if benchmark_case[3] == "config" and isinstance(benchmark_case[4].stage_option, WSCManualStageOption):
            host_ids_ =0
            num_devices_per_host_ = 0
            for item1 in benchmark_case[4].stage_option.submeshes:
                row_max = max(item1[1], item1[3])                
                clo_max = max(item1[0], item1[2])
                if row_max > host_ids_:
                    host_ids_ = row_max
                if clo_max> num_devices_per_host_:
                    num_devices_per_host_ = clo_max
            host_ids_ = host_ids_ + 1
            num_devices_per_host_ = num_devices_per_host_ + 1               
            virtual_mesh = VirtualPhysicalMesh(host_ids=np.arange(host_ids_),
                                            host_info=[g_vir_phy_mesh.host_info[0]]*host_ids_,
                                            num_devices_per_host=num_devices_per_host_,
                                            head_ip=g_vir_phy_mesh.head_ip)                 
            virtual_mesh.submeshes = benchmark_case[4].stage_option.submeshes
            set_global_virtual_physical_mesh(virtual_mesh)
            
            
        else:
            virtual_mesh = VirtualPhysicalMesh(host_ids=np.arange(num_hosts),
                                            host_info=[g_vir_phy_mesh.host_info[0]]*num_hosts,
                                            num_devices_per_host=num_devices_per_host,
                                            head_ip=g_vir_phy_mesh.head_ip)
            set_global_virtual_physical_mesh(virtual_mesh)
    else:
        virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
            host_ids=list(range(num_hosts)),
            num_devices_per_host=num_devices_per_host)
        set_global_virtual_physical_mesh(virtual_mesh)  

    # # Connect to the cluster
    # virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
    #     host_ids=list(range(num_hosts)),
    #     num_devices_per_host=num_devices_per_host)
    # set_global_virtual_physical_mesh(virtual_mesh)

    # Parallel configs
    allow_mixed_mesh_shape = True

    if benchmark_case.parallel_mode == "load_solution":
        use_fine_grained_remat = benchmark_case.parallel_args.use_remat
        fine_grained_remat_num_layers = benchmark_case.model_config.num_layers
    else:
        use_fine_grained_remat = None
        fine_grained_remat_num_layers = None

    (method, add_manual_remat, add_manual_layer_marker,num_manual_pipeline_stages) = get_pipeshard_parallel_method(
         benchmark_case,
         virtual_mesh.num_devices_per_host,
         use_fine_grained_remat=use_fine_grained_remat,
         allow_mixed_mesh_shape=allow_mixed_mesh_shape)

    use_grad_acc = benchmark_case.num_micro_batches > 1
    grad_func = alpa.grad if use_grad_acc else jax.grad
    state, batch, learning_rate_fn, rngkey = prepare_wresnet_input_and_model(
        benchmark_case)
    train_step = get_train_step(learning_rate_fn,
                                False,
                                None,
                                method,
                                grad_func=grad_func)

    (latencies, max_mem_allocated, compilation_times,
     executable, estimated_time_sum, estimated_time, 
     max_stage_cost) = compile_and_benchmark_pipeshard_training_executable(
         benchmark_case.parallel_mode,
         niter,
         train_step,
         state, (batch, ),
         profile_driver_time=profile_driver_time,
         offload=offload)
    # (latencies, max_mem_allocated, compilation_times,
    #  executable) = compile_and_benchmark_pipeshard_training_executable(
    #      benchmark_case.parallel_mode,
    #      niter,
    #      train_step,
    #      state, (batch,),
    #      profile_driver_time=profile_driver_time)

    # Profile submesh executables
    # del state
    # del metrics
    # for i, profiled in enumerate(executable.profile_all_executables()):
    #     pstr = f"Mesh {i}: "
    #     for k in profiled:
    #         pstr += f"Exec {k}: {profiled[k][0]}s; "
    #     print(pstr)

    # Compute statistics
    num_gpus = virtual_mesh.num_devices
    # TODO: compute WResNet TFlops & Param
    # print(f"&&&&&&&&&&&& executable = {executable}")
    # tflops = executable.flop_count / num_gpus / np.mean(latencies) / 1e12
    # parameter_count = compute_param_number(state.params)

    # # NOTE: below is wrong
    # tflops, parameter_count, total_tflops = compute_wresnet_statistics(benchmark_case, latencies, num_gpus)
    import suite_wresnet
    model_config = benchmark_case.model_config
    params_list = suite_wresnet.wresnet_params[tuple(model_config)]
    parameter_count, total_tflops = params_list
    total_tflops *= benchmark_case.batch_size
    tflops = total_tflops / np.mean(latencies) / num_gpus 

    (compute_cost_file_name, forward_stage_layer_ids, submesh_shapes,
     logical_mesh_shapes, autosharding_option_dicts, dp_cost) = get_last_dp_result()

    if global_config.full_on_hlo_analysis:
        estimated_total_time = estimated_time
    else:
        estimated_total_time = estimated_time_sum + (benchmark_case.num_micro_batches-1) * max_stage_cost
        

    metadata = {
        "compilation_times": compilation_times,
        "compute_cost_file_name": compute_cost_file_name,
        "forward_stage_layer_ids": forward_stage_layer_ids,
        "submesh_shapes": submesh_shapes,
        "logical_mesh_shapes": logical_mesh_shapes,
        "autosharding_option_dicts": autosharding_option_dicts,
        "dp_cost": dp_cost,
        "estimated_time_sum": estimated_time_sum,
        "estimated_time": estimated_time,
        "max_stage_cost": max_stage_cost,
        "estimated_total_time": estimated_total_time,
        "total_tflops": total_tflops,
    }
    # NOTE: We need return total_tflops, instead of TFlops -- otherwise for #gpu=1, TFlops=1531.8073(too huge)
    return parameter_count, max_mem_allocated, latencies, tflops, metadata


def benchmark_wresnet_2d_internal(physical_mesh,
                                  benchmark_case,
                                  niter,
                                  profile_driver_time=False,
                                  offload=False,):
    # Model configs
    method, grad_func = get_shard_parallel_method(benchmark_case, physical_mesh)

    use_grad_acc = benchmark_case.num_micro_batches > 1
    grad_func = alpa.grad if use_grad_acc else jax.grad
    state, batch, learning_rate_fn, rngkey = prepare_wresnet_input_and_model(
        benchmark_case)
    train_step = get_train_step(learning_rate_fn,
                                False,
                                None,
                                method,
                                grad_func=grad_func)

    (latencies, ilp_objective, peak_mem,
     executable) = compile_and_benchmark_shard_training_executable(
         physical_mesh,
         niter,
         train_step,
         state, (batch,),
         profile_driver_time=profile_driver_time)

    # Compute statistics
    num_gpus = physical_mesh.num_devices
    tflops = executable.flop_count / num_gpus / np.mean(latencies) / 1e12
    parameter_count = compute_param_number(state.params)
    metadata = {
        "ilp_objective": ilp_objective,
    }
    return parameter_count, peak_mem, latencies, tflops, metadata
