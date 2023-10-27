"""Follow the parallelization strategy of another function."""
import logging

from jax._src.lib import xla_extension as xe
from jax.core import ClosedJaxpr
from jax.interpreters import partial_eval as pe
from jax.tree_util import tree_leaves

from alpa.mesh_executable import (NormalMeshDriverExecutable,
                                  GradAccMeshDriverExecutable)
from alpa.parallel_plan import PlacementSpec
from alpa.pipeline_parallel.compile_executable import (
    compile_pipeshard_executable)
from alpa.pipeline_parallel.layer_construction import (ManualLayerOption,
                                                       FollowLayerOption,
                                                       FollowIdxLayerOption)
from alpa.pipeline_parallel.stage_construction import UniformStageOption
from alpa.shard_parallel.auto_sharding import (run_auto_sharding_pass,
                                               AutoShardingOption)
from alpa.util import (jaxpr_to_hlo_module, undefined_sharding_spec_proto)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compile_config_parallel_executable(fun, stage_num, in_tree, out_tree_thunk,
                                       static_argnums, donated_invars,
                                       batch_invars, virtual_mesh,
                                       num_micro_batches, input_placement_specs,
                                       partition_index, pipeline_schedule, layer_option,
                                       stage_option, *avals):
    

    # def is_leave(x):
    #     return isinstance(x, PlacementSpec) or x is None

    # input_placement_specs = tree_leaves(input_placement_specs, is_leave)

    num_micro_batches = num_micro_batches or 1

    if layer_option == "manual":
        layer_option = ManualLayerOption()
    elif layer_option == "follow":
        layer_option = FollowLayerOption(input_placement_specs, stage_num)
    elif layer_option == "follow_idx":
        layer_option = FollowIdxLayerOption(partition_index, stage_num)
    else:
        raise ValueError(f"Invalid layer option: {layer_option}")

    
    if layer_option == "follow":
        # follow sharding specs from input
        input_shardings = [x.sharding_specs[0] if x is not None else None for x in input_placement_specs]
    else:
        # use ILP solve sharding spec
        input_shardings = None

    # TODO(lmzheng): handle ReplicatedDistributedArray, tied embedding

    return compile_pipeshard_executable(
        fun, in_tree, out_tree_thunk, static_argnums, donated_invars,
        batch_invars, virtual_mesh, num_micro_batches, pipeline_schedule,
        AutoShardingOption(prefer_reduce_scatter=True, force_batch_dim_to_mesh_dim=-1), layer_option,
        stage_option, input_shardings, None, *avals)
