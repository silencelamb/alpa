"""All global configurations for this project."""
import gc
import os
from enum import Enum

from .mesh_search import gen_collective_cost_dict


GB = 1 << 30  # Gigabyte
MB = 1 << 20  # Megabyte
TOPS = 10 ** 12  # TOPS
ns = 10 ** -9  # ns
us = 10**(-6)


class PrimitiveType(Enum):
    INVALID = 0
    PRED = 1
    S8 = 2
    S16 = 3
    S32 = 4
    S64 = 5
    U8 = 6
    U16 = 7
    U32 = 8
    U64 = 9
    F16 = 10
    F32 = 11
    BF16 = 16
    F64 = 12
    C64 = 15
    C128 = 18
    TUPLE = 13
    OPAQUE_TYPE = 14
    TOKEN = 17


class GlobalConfig:
    """The global configuration of alpa."""

    def __init__(self):
        ########## Options of device mesh ##########
        # See https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
        self.xla_client_mem_fraction = float(
            os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", 0.9))
        self.xla_gpu_autotune_level = 4
        # The threshold to tigger a batched deletion on workers.
        self.delete_remote_arrays_threshold = 50
        # Whether to use AWS EFA network interface
        self.use_aws_efa = os.environ.get("ALPA_USE_AWS_EFA",
                                          "").lower() in ["true", "1"]
        # Random seed used for compilation
        self.compile_random_seed = 42
        # Random seed used for runtime
        self.runtime_random_seed = 42

        ########## Options of shard_parallel ##########
        # Whether to sync before and after the executable for accurate internal
        # timer
        self.shard_parallel_sync_for_timer = False

        ########## Options of pipeline_parallel ##########
        # Whether to debug with pipeshard runtime. If turned on, no physical
        # resource is required until launching PipeshardExecutable.
        self.debug_with_pipeshard_runtime = True
        # Whether to use the whole cluster for stage profiling. If not, only
        # use the given mesh.
        # it should be set as False, when we want to excute mapping in arbitrary mesh
        self.profile_with_whole_ray_cluster = False
        # Stage construction profiling time threshold.
        self.profile_timeout = 500
        # Stage construction profiling retry threshold.
        # Some communication patterns may meet deadlock, so it needs retry.
        self.profile_maximum_retry = 2
        # Whether to forcely set stage construction's submesh choices
        self.overwrite_submesh_choices = None
        self.always_donate_micro_batch_vars = True

        ########## Options of pipeline runtime ##########
        self.pipeline_check_alive = True
        # Whether to sync before and after the executable for accurate internal
        # timer
        self.pipeline_sync_for_timer = False
        # Whether to use distributed compilation in pipeline parallel for
        # each stage. Disabling it helps debug.
        self.pipeline_distributed_compile = False
        # Whether to use single-byte signal tensor for send/recv.
        # This is a debug option.
        self.pipeline_use_signal_send_recv = False
        # Whether to use the scatter-gater/local-all-gather optimization.
        self.use_local_allgather = True
        self.eagerly_create_communicators = True
        self.use_memzero_for_gradient_accumulation = False
        # Cross mesh resharding mode. Possible choices: {"send_recv",
        # "broadcast"}
        self.resharding_mode = "send_recv"
        # Which nccl to use. Possible choices: {"cupy",
        # "xla_extension"}
        self.nccl_mode = "cupy"

        ########## Options of XLA compilation ##########
        # Whether to use xla while instruction for preventing CSE in
        # rematerialization
        self.remat_using_while = False

        ########## Options of benchmark ##########
        # If true, the system is allowed to use dummy values during
        # tensor creation and copy to reduce the initialization and copy time.
        # This will produce wrong results but is acceptable for
        # data-independent benchmarks.
        self.use_dummy_value_for_benchmarking = False

        ########## Options of monkey patch ##########
        self.flax_always_use_fp16_embedding = False

        ########## Options of logging ##########
        self.print_compilation_time = False
        self.print_auto_layer_stats = True

        ########## Options of save jaxpr ##########
        self.save_jaxpr_json = False
        self.save_jaxpr_dir = "saved_jaxpr_json"
        self.save_jaxpr_json_file = "saved_jaxpr.json"

        ########## Options of mapping ################
        # when only_mapping is True, only do mapping, does not get excutable or do acutal compute
        self.only_mapping = True
        # full on hlo analysis, don't depend on cuda platform any more
        self.full_on_hlo_analysis = True
        # result folder
        self.rst_folder = ""
        # mapping result dir
        self.maping_rst_dir = ""
        # whether using analytical performance model
        self.use_analytical_perf_model = False
        self.hardware = "gpu"
        # self.hardware = "wsc"
        self.force_use_fp16 = False
        self.gpu_config = {
            "analytical_perf::hardware": "gpu",
            # A100 PCIe卡的算力
            "analytical_perf::compute_dict": {
                PrimitiveType.F16.value: 312 * TOPS,
                PrimitiveType.F32.value: 156 * TOPS,
            },
            "analytical_perf_gpu::card_num": 8,
            "analytical_perf_gpu::card_mem": 80 * GB,
            # "analytical_perf_gpu::card_bw": 600 * GB,
            "analytical_perf_gpu::card_bw": 200 * GB,
            # "analytical_perf_gpu::card_bw": 900 * GB,
            "analytical_perf_gpu::node_bw": int(25/8 * GB),   # alpa 里是 25Gbps
            # "analytical_perf_gpu::node_bw": 200 * GB,
            # "analytical_perf_gpu::node_bw": 600 * GB,   # nv link
            "analytical_perf_gpu::ddr_bandwidth": 500 * GB,  # ddr bandwidth, GB/s
            "analytical_perf_gpu::pcie_bandwidth": 32 * GB,  # PCIE 4.0 x 16 lane, GB/s
            "analytical_perf::cmp_ul": 0.7,
            "analytical_perf::bw_ul": 0.7
        }
        # tx8 config
        self.wsc_config = {
            "analytical_perf::hardware": "wsc",
            "analytical_perf_wsc::die_r_num": 5,
            "analytical_perf_wsc::die_c_num": 4,
            "analytical_perf::compute_dict": {
                PrimitiveType.F16.value: int(256/16 * TOPS),
                PrimitiveType.F32.value: int(256/16 * TOPS),
                # PrimitiveType.F32.value: int( 20/36  * TOPS),
            },
            "analytical_perf_wsc::tile_r_num": 4,
            "analytical_perf_wsc::tile_c_num": 4,
            "analytical_perf_wsc::tile_mem": 3 * MB,  # sram size / tile
            "analytical_perf_wsc::tile_bw": 128 * GB,
            "analytical_perf_wsc::die_bw": 25 * GB,
            "analytical_perf_wsc::ddr_bandwidth": 100 * GB,  # ddr bandwidth, GB/s
            "analytical_perf_wsc::ddr_mem":  12 * GB,   # add 2023-10-31
            "analytical_perf_wsc::pcie_bandwidth": 32 * GB,  # PCIE 4.0 x 16 lane, GB/s
            "analytical_perf_wsc::die_alpha": 100 * ns,  # add 2023-10-31, d2d latency, ns
            # add  2023-10-31, mesh topo-aware collective
            "analytical_perf::use_greedy_coll_cost": False,
            "analytical_perf::cmp_ul": 0.8,
            "analytical_perf::bw_ul": 0.8
        }

        # Tesla DOJO  config
        # TODO add dojo config @dehao
        self.dojo_config = {
            "analytical_perf::hardware": "wsc",
            "analytical_perf_wsc::die_r_num": 5,
            "analytical_perf_wsc::die_c_num": 5,
            # NOTE: 361 match 19*19 instead of 354 -- single tile compute capacity is 362/22TFLOPS
            "analytical_perf::compute_dict": {
                PrimitiveType.F16.value: int(362/361 * TOPS),
                PrimitiveType.F32.value: int(22/361 * TOPS),
                # PrimitiveType.F32.value: int( 20/36  * TOPS),
            },
            "analytical_perf_wsc::tile_r_num": 19,
            "analytical_perf_wsc::tile_c_num": 19,
            # SRAM size / tile - int(1.25MB)
            "analytical_perf_wsc::tile_mem": 1 * MB,
            "analytical_perf_wsc::tile_bw": 14 * GB,

            "analytical_perf_wsc::die_bw": 2048 * GB,
            # NOTE: ddr only consider one, not 5 edge of training Tile
            "analytical_perf_wsc::ddr_bandwidth": 800 * GB,  # ddr bandwidth, GB/s
            # add 2023-10-31, config as 32*5/25 for each die
            "analytical_perf_wsc::ddr_mem":  6553 * MB,
            "analytical_perf_wsc::pcie_bandwidth": 160 * GB,  # PCIE 4.0 x 80 lane, GB/s

            "analytical_perf_wsc::die_alpha": 100 * ns,  # add 2023-10-31, d2d latency, ns
            # add  2023-10-31, mesh topo-aware collective
            "analytical_perf::use_greedy_coll_cost": False,
            "analytical_perf::cmp_ul": 0.8,
            "analytical_perf::bw_ul": 0.8
        }
        # Wafer-Scale GPU config
        # TODO add Wafer-Scale GPU config @dehao
        self.wsgpu_config = {
            "analytical_perf::hardware": "wsc",
            "analytical_perf_wsc::die_r_num": 6,
            "analytical_perf_wsc::die_c_num": 4,
            # NVIDIA T4 compute capacity
            "analytical_perf::compute_dict": {
                PrimitiveType.F16.value: int(65 * TOPS),
                PrimitiveType.F32.value: int(8.1 * TOPS),
                # PrimitiveType.F32.value: int( 20/36  * TOPS),
            },
            "analytical_perf_wsc::tile_r_num": 1,
            "analytical_perf_wsc::tile_c_num": 1,

            "analytical_perf_wsc::tile_mem": 4*MB,  # SRAM size / tile
            "analytical_perf_wsc::tile_bw": 320 * GB,

            "analytical_perf_wsc::die_bw": 1536 * GB,  # 1.5TB
            "analytical_perf_wsc::ddr_bandwidth": 1536 * GB,  # ddr bandwidth, GB/s
            # NOTE: two 3D stacked-HBM = 2 * 4GB
            "analytical_perf_wsc::ddr_mem":  2 * 4 * GB,   # add 2023-10-31
            "analytical_perf_wsc::pcie_bandwidth": 32 * GB,  # PCIE 4.0 x 16 lane, GB/s

            # GPM interconnect
            "analytical_perf_wsc::die_alpha": 20 * ns,  # add 2023-10-31, d2d latency, ns
            # add  2023-10-31, mesh topo-aware collective
            "analytical_perf::use_greedy_coll_cost": False,
            "analytical_perf::cmp_ul": 0.8,
            "analytical_perf::bw_ul": 0.8,
        }

        self.use_greedy_collective_cost = False
        self.collective_cost_dict = None


global_config = GlobalConfig()


def get_global_config():
    return global_config


def get_collective_cost_dict():
    die_row_num = global_config.wsc_config["analytical_perf_wsc::die_r_num"]
    die_col_num = global_config.wsc_config["analytical_perf_wsc::die_c_num"]
    if global_config.wsc_config["analytical_perf::use_greedy_coll_cost"]:
        global_config.collective_cost_dict = gen_collective_cost_dict(
            die_row_num, die_col_num)
        # print(global_config.collective_cost_dict)


def set_global_config(global_config_new: GlobalConfig):
    global global_config
    global_config = global_config_new


# Other environment setup
is_worker = os.environ.get("ALPA_IS_WORKER", "False") == "True"

os.environ["XLA_FLAGS"] = os.environ.get(
    "XLA_FLAGS", "") + " --xla_gpu_enable_async_all_reduce=false"
