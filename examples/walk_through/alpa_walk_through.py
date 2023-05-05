"""Some basic tests to test installation."""
import os
import unittest

from alpa import (init, parallelize, ShardParallel, PipeshardParallel,
                  AutoLayerOption, ManualStageOption, prefetch)
from alpa.device_mesh import get_global_cluster
from alpa.testing import assert_allclose, get_mlp_train_state_and_step
from benchmark.alpa.benchmark_parallel_utils import UniformParallelArgs



class InstallationTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    def test_1_shard_parallel(self):
        state, batch, train_step = get_mlp_train_state_and_step(batch_size=128,
                                                                hidden_size=128,
                                                                num_layers=4)

        # Serial execution
        expected_output = train_step(state, batch)

        # Parallel execution
        p_train_step = parallelize(train_step,
                                   method=ShardParallel(num_micro_batches=2))
        actual_output = p_train_step(state, batch)

        # Check results
        assert_allclose(expected_output, actual_output)

    def test_2_pipeline_parallel(self):
        init(cluster="ray")
        # dp, op, pp = 1, 2, 2
        dp, op, pp = 2, 2, 2
        debug_info_dir = './executable_dump_info_8GPU/'
        logical_mesh_shape = (dp, op)
        physical_mesh_shape = (1, dp*op)

        state, batch, train_step = get_mlp_train_state_and_step(batch_size=32,
                                                                hidden_size=128,
                                                                num_layers=4,
                                                                add_manual_pipeline_marker=True)

        # Serial execution
        expected_output = train_step(state, batch)

        # Parallel execution
        layer_num = min(get_global_cluster().num_devices, 2)
        p_train_step = parallelize(
            train_step,
            method=PipeshardParallel(
                num_micro_batches=2,
                layer_option="manual",
                stage_option=ManualStageOption(
                    forward_stage_layer_ids=[[i] for i in range(pp)],
                    submesh_physical_shapes=[physical_mesh_shape] * pp,
                    submesh_logical_shapes=[logical_mesh_shape] * pp,
                    submesh_autosharding_option_dicts=[{}] * pp))
            )
        import pdb; pdb.set_trace()
        executable = p_train_step.get_executable(state, batch)
        
        executable.dump_debug_info(debug_info_dir)
        for idx, stage in enumerate(executable.stages):
            with open(f'{debug_info_dir}/train_step_stage_{idx}_sharded_annotated.hlo', 'w') as f:
                f.write(stage.sharding_annotated_module_str)
            with open(f'{debug_info_dir}/train_step_stage_{idx}_spmd_partitioned.hlo', 'w') as f:
                f.write(stage.spmd_partitioned_hlo_module.to_string())
                
        import pdb; pdb.set_trace()
        actual_output = p_train_step(state, batch)
        # import pdb; pdb.set_trace()
        # Check results
        prefetch(actual_output)
        assert_allclose(expected_output, actual_output)


def suite():
    s = unittest.TestSuite()
    # s.addTest(InstallationTest("test_1_shard_parallel"))
    s.addTest(InstallationTest("test_2_pipeline_parallel"))
    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
