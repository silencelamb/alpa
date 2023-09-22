"""Only test that memory offloader can produce mem infos."""
import os
import unittest

from alpa import (init, parallelize, PipeshardParallel,
                  AutoLayerOption, AutoStageOption, prefetch)
from alpa.device_mesh import get_global_cluster
from alpa.testing import assert_allclose, get_mlp_train_state_and_step, \
                         get_bert_layer_train_state_and_step
from alpa.global_env import set_global_config, get_global_config

GB = 1 << 30  # Gigabyte
MB = 1 << 20  # Megabyte
TOPS =  10 ** 12 # TOPS

# strategy 3 test case
def test_1_pipeline_parallel():
    init(cluster="ray")
    state, batch, train_step = get_mlp_train_state_and_step(batch_size=32,
                                                            hidden_size=20480,
                                                            num_layers=4,
                                                            add_manual_pipeline_marker=False)
    # Serial execution
    expected_output = train_step(state, batch)

    print(" ---------------  alpa do parallel call ------------------")
    global_config = get_global_config()
    print("******---> ", global_config.use_analytical_perf_model)

    auto_stage_option = AutoStageOption(
                                    submesh_physical_shape_space="power_of_two",
                                    submesh_logical_shape_space="single_node_model_parallel",
                                    stage_imbalance_tolerance=1.0,
                                    use_hlo_cost_model=True,
                                    profiling_database_filename="/code/alpa/benchmark/alpa/prof_database.pkl")
    # Parallel execution
    layer_num = min(get_global_cluster().num_devices, 2)
    p_train_step = parallelize(
        train_step,
        method=PipeshardParallel(
            num_micro_batches=2,
            layer_option=AutoLayerOption(layer_num=2),
            stage_option=auto_stage_option
            ), 
        )
    hlo_estimator = p_train_step.get_executable(state, batch)
    hlo_estimator.estimate_cost_on_hlo_analysis()

# strategy 1 test case
def test_2_pipeline_parallel():
    init(cluster="ray")
    state, batch, train_step = get_mlp_train_state_and_step(batch_size=1024,
                                                            hidden_size=2048,
                                                            num_layers=66,
                                                            add_manual_pipeline_marker=False)
    # Serial execution
    expected_output = train_step(state, batch)

    print(" ---------------  alpa do parallel call ------------------")
    global_config = get_global_config()
    print("******---> ", global_config.use_analytical_perf_model)

    auto_stage_option = AutoStageOption(
                                    submesh_physical_shape_space="power_of_two",
                                    submesh_logical_shape_space="single_node_model_parallel",
                                    stage_imbalance_tolerance=1.0,
                                    use_hlo_cost_model=True,
                                    profiling_database_filename="/code/alpa/benchmark/alpa/prof_database.pkl")
    # Parallel execution
    layer_num = min(get_global_cluster().num_devices, 2)
    p_train_step = parallelize(
        train_step,
        method=PipeshardParallel(
            num_micro_batches=2,
            layer_option=AutoLayerOption(layer_num=2),
            stage_option=auto_stage_option
            ), 
        )
    hlo_estimator = p_train_step.get_executable(state, batch)
    hlo_estimator.estimate_cost_on_hlo_analysis()


# strategy 3 test case
def test_3_pipeline_parallel():
    init(cluster="ray")
    state, batch, train_step = get_mlp_train_state_and_step(batch_size=32,
                                                            hidden_size=20480,
                                                            num_layers=16,
                                                            add_manual_pipeline_marker=False)
    # Serial execution
    expected_output = train_step(state, batch)

    print(" ---------------  alpa do parallel call ------------------")
    global_config = get_global_config()
    print("******---> ", global_config.use_analytical_perf_model)

    auto_stage_option = AutoStageOption(
                                    submesh_physical_shape_space="power_of_two",
                                    submesh_logical_shape_space="single_node_model_parallel",
                                    stage_imbalance_tolerance=1.0,
                                    use_hlo_cost_model=True,
                                    profiling_database_filename="/code/alpa/benchmark/alpa/prof_database.pkl")
    # Parallel execution
    layer_num = min(get_global_cluster().num_devices, 2)
    p_train_step = parallelize(
        train_step,
        method=PipeshardParallel(
            num_micro_batches=2,
            layer_option=AutoLayerOption(layer_num=2),
            stage_option=auto_stage_option
            ), 
        )
    hlo_estimator = p_train_step.get_executable(state, batch)
    hlo_estimator.estimate_cost_on_hlo_analysis()

if __name__ == "__main__":
    global_config = get_global_config()
    global_config.full_on_hlo_analysis = True
    global_config.use_analytical_perf_model = True
    global_config.gpu_config["analytical_perf_gpu::card_mem"] = 1*GB
    set_global_config(global_config)
    
    test_1_pipeline_parallel()
    # test_2_pipeline_parallel()
    # test_3_pipeline_parallel()