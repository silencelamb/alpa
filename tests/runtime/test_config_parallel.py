"""Test following another parallel strategy."""
import unittest
import time
import numpy as np
import pickle

import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
from jax._src.api import make_jaxpr
import jax.numpy as jnp
import optax

import alpa
from alpa import (init, shutdown, parallelize, ShardParallel, PipeshardParallel,
                  CreateStateParallel, ConfigParallel, ManualStageOption)
from alpa.pipeline_parallel.pipeshard_executable import PipeshardDriverExecutable
from alpa.device_mesh import get_global_virtual_physical_mesh, VirtualPhysicalMesh


class ConfigParallelTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def tearDown(self):
        shutdown()

    def run_test(self):
        use_bias = True
        batch_size = 32
        input_dim = output_dim = hidden_dim = 8

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=hidden_dim, use_bias=use_bias)(x)
                x = nn.Dense(features=hidden_dim, use_bias=use_bias)(x)
                x = nn.Dense(features=hidden_dim, use_bias=use_bias)(x)
                x = nn.Dense(features=output_dim, use_bias=use_bias)(x)
                return x

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"])**2)

            grads = alpa.grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        def eval_step(params, batch):
            out = state.apply_fn(params, batch["x"])
            return jnp.mean((out - batch["y"])**2)

        def create_state():
            model = Model()
            rngkey = jax.random.PRNGKey(0)
            params = model.init(rngkey, jnp.ones((1, input_dim)))
            tx = optax.adam(learning_rate=1e-2)
            return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        train_batch = {
            "x": jnp.ones((batch_size, input_dim)),
            "y": jnp.ones((batch_size, output_dim)),
        }

        num_micro_batches = 2
        state = create_state()

        g_vir_phy_mesh = get_global_virtual_physical_mesh()
        num_hosts = 1
        num_devices_per_host = 8
        virtual_mesh = VirtualPhysicalMesh(host_ids=np.arange(num_hosts),
                                           host_info=[g_vir_phy_mesh.host_info[0]]*num_hosts,
                                           num_devices_per_host=num_devices_per_host,
                                           head_ip=g_vir_phy_mesh.head_ip)

        s_option = ManualStageOption([[0], [1]], [(1, 4)] * 2, [(1, 4)] * 2,
                                     [{}] * 2)
        # method = PipeshardParallel(num_micro_batches=2, 
        #                            layer_option=alpa.AutoLayerOption(layer_num=2),
        #                            stage_option=s_option)

        # train_step_new_1 = parallelize(train_step, method=method)
        # # import pdb; pdb.set_trace()
        # executable = train_step_new_1.get_executable(state, train_batch)
        # executable: PipeshardDriverExecutable
        # import pdb; pdb.set_trace()
        # input_placement_specs = executable.input_placement_specs
        # output_placement_specs = executable.output_placement_specs
        # mesh_group = executable.mesh_group.parent
        # import pdb; pdb.set_trace()
        
        # with open('./input_placement_specs.pkl', 'wb') as f:
        #     pickle.dump(input_placement_specs, f)


        with open('./input_placement_specs_new.pkl', 'rb') as f:
            input_placement_specs = pickle.load(f)
        train_step_new_2 = parallelize(train_step,
                                method=alpa.ConfigParallel(
                                    stage_num=2,
                                    input_placement_specs=input_placement_specs,
                                    virtual_mesh=virtual_mesh,
                                    pipeline_schedule = "1f1b",
                                    stage_option = s_option,
                                    # pipeline_schedule = "inference",
                                    num_micro_batches=num_micro_batches))

        executable_2 = train_step_new_2.get_executable(state, train_batch)
        executable_2: PipeshardDriverExecutable
        import pdb; pdb.set_trace()
        input_placement_specs_new = executable_2.input_placement_specs

        with open('./input_placement_specs_new.pkl', 'wb') as f:
            pickle.dump(input_placement_specs_new, f)


        # actual = jax.tree_flatten(
        #     eval_step.get_last_executable().get_input_placement_specs()[0])[0]
        # expected = jax.tree_flatten(
        #     train_step.get_last_executable().get_input_placement_specs()
        #     [0].params)[0]
        # assert actual == expected

    def test_pipeshard_parallel(self):
        self.run_test()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(ConfigParallelTest("test_pipeshard_parallel"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
