"""Benchmark one case of inter-op + intra-op parallelism."""
import jax
import jax.numpy as jnp
import numpy as np
import optax

import alpa
from alpa import (parallelize, get_global_cluster,
                  set_global_virtual_physical_mesh, automatic_remat,
                  global_config, set_global_option_model_type)
from alpa.model.bert_model import BertConfig, FlaxBertForMaskedLMModule
from alpa.model.model_util import TrainState
from alpa.model.gpt_model import FlaxGPTForLMModule
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.util import print_used_time
from alpa.device_mesh import VirtualPhysicalMesh, get_global_virtual_physical_mesh
from alpa.global_env import get_global_config, set_global_config

from flax import linen as nn
from suite_manual_gpt import MLPModelConfig, GPTModelConfig


from util import compute_gpt_parameter_count,compute_mlp_parameter_count, compute_gpt_tflops,compute_mlp_tflops
from benchmark_parallel_utils import (
    BenchmarkCase, get_pipeshard_parallel_method, get_shard_parallel_method,
    compile_and_benchmark_pipeshard_training_executable,
    compile_and_benchmark_shard_training_executable)


def report_pipeline_breakdown(executable, timer_names, niter):
    overall_costs = executable.get_execution_time_costs(timer_name="overall")

    print(">>> overall: {}...".format(overall_costs))
    other_percentage = [100.0] * niter
    other = overall_costs
    for timer_name in timer_names:
        costs = executable.get_execution_time_costs(timer_name=timer_name)
        if len(costs) == 0:
            costs = [0.0] * niter
        percentage = [
            cost / overall_costs[i] * 100 for i, cost in enumerate(costs)
        ]
        other = [remain - costs[i] for i, remain in enumerate(other)]
        other_percentage = [
            remain - percentage[i] for i, remain in enumerate(other_percentage)
        ]
        strs = []
        for i, cost in enumerate(costs):
            strs.append(str(cost) + f" ({percentage[i]:.1f}) ")
        print_string = ",".join(strs)
        print(">>> {}: {}".format(timer_name, print_string))

    # print unknown overhead
    strs = []
    for i, remain in enumerate(other):
        strs.append(" " + str(remain) + f" ({other_percentage[i]:.1f})")
    print_string = ",".join(strs)
    print(">>> {}: {}".format("Others: ", print_string))


def create_train_state(rngkey, model, batch, dtype):
    params = model.init_dummy(rngkey, batch["input_ids"],
                              batch["attention_mask"], batch["token_type_ids"],
                              batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax.chain(
        #optax.clip_by_global_norm(1.0),  # TODO(lmzheng): fix reduce-scatter for this
        optax.adamw(learning_rate=1e-2, mask=weight_decay_mask))
    use_master_copy = (dtype == jnp.float16)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              use_master_copy=use_master_copy,
                              dynamic_scale=None)
    return state


def create_train_state_aval(rngkey, model, batch, dtype):
    params = jax.eval_shape(model.init, rngkey, batch["input_ids"],
                            batch["attention_mask"], batch["token_type_ids"],
                            batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax.chain(
        #optax.clip_by_global_norm(1.0),  # TODO(lmzheng): fix reduce-scatter for this
        optax.adamw(learning_rate=1e-2, mask=weight_decay_mask))
    use_master_copy = (dtype == jnp.float16)
    state = TrainState.create_aval(apply_fn=model.apply,
                                   params=params,
                                   tx=tx,
                                   use_master_copy=use_master_copy,
                                   dynamic_scale=None)
    return state


def get_train_step(parallel_method,
                   use_fine_grained_remat=False,
                   fine_grained_remat_num_layers=None,
                   grad_func=None):

    if grad_func is None:
        grad_func = alpa.grad

    @parallelize(method=parallel_method)
    def train_step(state, batch, rng_key):

        def loss_func(params):
            rngs = {"dropout": rng_key}
            logits = state.apply_fn(params,
                                    batch["input_ids"],
                                    batch["attention_mask"],
                                    batch["token_type_ids"],
                                    batch["position_ids"],
                                    deterministic=True,
                                    rngs=rngs)[0]
            label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
            labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
            loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1),
                            axis=-1)
            loss = (label_mask * loss).sum() / label_mask.sum()
            return loss

        if use_fine_grained_remat:
            loss_func = automatic_remat(loss_func, layer_num=fine_grained_remat_num_layers)

        grads = grad_func(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        # TODO(lmzheng): add dynamic scaling for mixed-precision training
        return new_state

    return train_step

def get_train_step_mlp(parallel_method,
                   use_fine_grained_remat=False,
                   fine_grained_remat_num_layers=None,
                   grad_func=None):

    if grad_func is None:
        grad_func = alpa.grad

    @parallelize(method=parallel_method)
    def train_step(state, batch, rng_key):

        def loss_func(params,x,y):
            rngs = {"dropout": rng_key}
            logits = state.apply_fn(params,x)
            # out = logits + jnp.array(range(128)).reshape((-1, 1))
            out = logits
            loss = jnp.mean((out - y)**2)
            return loss        

        grads = grad_func(loss_func)(state.params,batch["x"],batch["y"])
        new_state = state.apply_gradients(grads=grads)        #
        return new_state
    return train_step


class mlp_Model(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        for i in range(16):
            x = nn.Dense(features=1024,dtype= jnp.float16,use_bias=True)(x)
            x = nn.relu(x)
            x = nn.Dense(features=1024,dtype= jnp.float16,use_bias=True)(x)
        return x

def create_train_state_mlp(rngkey, model, batch, dtype):    
    params = model.init(rngkey, batch["x"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax.chain(
        #optax.clip_by_global_norm(1.0),  # TODO(lmzheng): fix reduce-scatter for this
        optax.adamw(learning_rate=1e-2, mask=weight_decay_mask))
    use_master_copy = (dtype == jnp.float16)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              use_master_copy=use_master_copy,
                              dynamic_scale=None)
    return state
                      

def prepare_gpt_bert_input_and_model(model_type,
                                     benchmark_case,
                                     add_manual_remat=None,
                                     add_manual_layer_marker=None,
                                     num_manual_pipeline_stages=None,
                                     aval_train_state=True,
                                     tie_word_embeddings=False):
    print_used_time(None)
    batch_size = benchmark_case.batch_size
    
    num_layers = 0
    hidden_size = 0
    use_bias = 0
    batch = {}
    # import ipdb; ipdb.set_trace()
    if (type(benchmark_case.model_config) is GPTModelConfig):        
            
        (seq_len, hidden_size, num_layers, num_heads,
        vocab_size) = benchmark_case.model_config
        dtype = jnp.float16
        # Prepare input batch
        batch = {
            "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "token_type_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "position_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        }
        print_used_time("Prepare input")

        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
            tie_word_embeddings=tie_word_embeddings,
            gradient_checkpointing=add_manual_remat,
            add_manual_pipeline_markers=add_manual_layer_marker,
            pipeline_mp_size=num_manual_pipeline_stages,
        )    
    elif (type(benchmark_case.model_config) is MLPModelConfig):
        num_layers, hidden_size, use_bias = benchmark_case.model_config
        batch = {
            "x": jnp.ones((128, hidden_size)),
            "y": jnp.ones((128,  hidden_size))}

    # Init train state
    if model_type == "bert":
        model = FlaxBertForMaskedLMModule(bert_config, dtype=dtype)
    elif model_type == "gpt":
        model = FlaxGPTForLMModule(bert_config, dtype=dtype)
    elif model_type == "mlp":        
        model = mlp_Model()   
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    
    if (type(benchmark_case.model_config) is MLPModelConfig):
        # state = create_train_state_mlp(rngkey, model, batch, dtype)
        params = model.init(rngkey, batch['x'])
        tx = optax.adam(learning_rate=1e-2)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, dynamic_scale=None)
        # state = train_step(state, {"x": x, "y": y})
    else:
        if aval_train_state:
            state = create_train_state_aval(rngkey, model, batch, dtype)
        else:
            state = create_train_state(rngkey, model, batch, dtype)
    print_used_time("Create train state")
    return state, batch, rngkey


def compute_gpt_bert_statistics(benchmark_case, latencies, num_devices):
    batch_size = benchmark_case.batch_size
    
    if (type(benchmark_case.model_config) is MLPModelConfig):
        pass 
        (num_layers, hidden_size, use_bias) = benchmark_case.model_config 
        tflops = compute_mlp_tflops(batch_size,                                    
                                    num_layers,
                                    hidden_size,                                    
                                    num_devices,
                                    np.mean(latencies)
                                    )
        parameter_count = compute_mlp_parameter_count(num_layers, hidden_size)
    else:        
        (seq_len, hidden_size, num_layers, num_heads,
        vocab_size) = benchmark_case.model_config  
        use_remat = benchmark_case.parallel_args.use_remat
        tflops = compute_gpt_tflops(batch_size,
                                    seq_len,
                                    num_layers,
                                    hidden_size,
                                    vocab_size,
                                    num_devices,
                                    np.mean(latencies),
                                    checkpoint_activations=use_remat)
        parameter_count = compute_gpt_parameter_count(num_layers, hidden_size,
                                                    vocab_size)
    
    
    
    return tflops, parameter_count


def benchmark_gpt_bert_3d_internal(model_type,
                                   benchmark_case: BenchmarkCase,
                                   niter,
                                   num_hosts,
                                   num_devices_per_host,
                                   aval_train_state=True,
                                   profile_driver_time=False):
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
                row_max = max(item1[0], item1[2])                
                clo_max = max(item1[1], item1[3])
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

    # Parallel configs
    if benchmark_case.parallel_mode == "load_solution":
        use_fine_grained_remat = benchmark_case.parallel_args.use_remat
        fine_grained_remat_num_layers = benchmark_case.model_config.num_layers
    else:
        use_fine_grained_remat = None
        fine_grained_remat_num_layers = None
    (method, add_manual_remat, add_manual_layer_marker,num_manual_pipeline_stages) = get_pipeshard_parallel_method(
         benchmark_case,
         virtual_mesh.num_devices_per_host,
         use_fine_grained_remat=use_fine_grained_remat)

    state, batch, rngkey = prepare_gpt_bert_input_and_model(
        model_type,
        benchmark_case,
        add_manual_remat=add_manual_remat,
        add_manual_layer_marker=add_manual_layer_marker,
        num_manual_pipeline_stages=num_manual_pipeline_stages,
        aval_train_state=aval_train_state)
    if (type(benchmark_case.model_config) is MLPModelConfig):        
        train_step = get_train_step_mlp(method, use_fine_grained_remat, fine_grained_remat_num_layers)
    else:
        train_step = get_train_step(method, use_fine_grained_remat, fine_grained_remat_num_layers)

    (latencies, max_mem_allocated, compilation_times,
     executable, estimated_time_sum, estimated_time, 
     max_stage_cost) = compile_and_benchmark_pipeshard_training_executable(
         benchmark_case.parallel_mode,
         niter,
         train_step,
         state, (batch, rngkey),
         profile_driver_time=profile_driver_time)

    
    tflops, parameter_count = compute_gpt_bert_statistics(benchmark_case, latencies, virtual_mesh.num_devices)

    # report_pipeline_breakdown(executable,
    #                           ["resharding_send", "resharding_recv",
    #                            "compute"],
    #                           niter)

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
        "estimated_total_time": estimated_total_time
    }

    return parameter_count, max_mem_allocated, latencies, tflops, metadata


def benchmark_gpt_bert_2d_internal(physical_mesh,
                                   model_type,
                                   benchmark_case,
                                   niter,
                                   profile_driver_time=False):
    method, grad_func = get_shard_parallel_method(benchmark_case, physical_mesh)

    state, batch, rngkey = prepare_gpt_bert_input_and_model(
        model_type,
        benchmark_case,
        add_manual_remat=benchmark_case.parallel_args.use_remat,
        aval_train_state=global_config.use_dummy_value_for_benchmarking)

    train_step = get_train_step(method, grad_func=grad_func)

    (latencies, ilp_objective, peak_mem,
     executable) = compile_and_benchmark_shard_training_executable(
         physical_mesh,
         niter,
         train_step,
         state, (batch, rngkey),
         profile_driver_time=profile_driver_time)

    tflops, parameter_count = compute_gpt_bert_statistics(
        benchmark_case, latencies, physical_mesh.num_devices)
    metadata = {
        "ilp_objective": ilp_objective,
    }
    return parameter_count, peak_mem, latencies, tflops, metadata
