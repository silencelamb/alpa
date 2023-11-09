from benchmark_one_case import benchmark_one_case
from suite_manual_gpt import GPTModelConfig
from suite_manual_bert import BERTModelConfig
from suite_wresnet import WResNetModelConfig
from benchmark_parallel_utils import (BenchmarkCase, SearchParallelArgs,
                                      LoadSolutionParallelArgs, ConfigParallelArgs,
                                      UniformParallelArgs)
all_models = {
    # GPT models
    "GPT-125M": GPTModelConfig(1024, 768, 12, 12, 51200),
    "GPT-350M": GPTModelConfig(1024, 1024, 24, 16, 51200),
    "GPT-760M": GPTModelConfig(1024, 1536, 24, 16, 51200),
    "GPT-1.3B": GPTModelConfig(1024, 2048, 24, 32, 51200),
    "GPT-2.6B": GPTModelConfig(1024, 2560, 32, 32, 51200),
    "GPT-6.7B": GPTModelConfig(1024, 4096, 32, 32, 51200),
    # "GPT-15B": GPTModelConfig(1024, 5120, 48, 40, 51200),
    # "GPT-39B": GPTModelConfig(1024, 8192, 48, 64, 51200),
    # "GPT-76B": GPTModelConfig(1024, 10240, 60, 80, 51200),

    # BERT models
    "BERT-Tiny": BERTModelConfig(512, 128, 2, 8, 30522),
    "BERT-Mini": BERTModelConfig(512, 256, 4, 8, 30522),
    "BERT-Small": BERTModelConfig(512, 512, 4, 8, 30522),
    "BERT-Medium": BERTModelConfig(512, 512, 8, 8, 30522),
    "BERT-Base": BERTModelConfig(512, 768, 12, 12, 30522),
    "BERT-Large": BERTModelConfig(512, 1024, 24, 16, 30522),
    "BERT-LL": BERTModelConfig(512, 1536, 24, 16, 30522),
    # "BERT-LLL": BERTModelConfig(512, 2048, 24, 32, 30522),
    # "BERT-LLLL": BERTModelConfig(512, 2560, 32, 32, 30522),
    # "BERT-LLLLL": BERTModelConfig(512, 4096, 32, 32, 30522),
    # "BERT-LLLLLL": BERTModelConfig(512, 5120, 48, 40, 30522),
    # "BERT-LLLLLLL": BERTModelConfig(512, 8192, 48, 64, 30522),
    # "BERT-LLLLLLLL": BERTModelConfig(512, 10240, 60, 80, 30522),

    # ResNet models
    "ResNet-25.56M": WResNetModelConfig(224, 50, 64, 1, "fp16"),  # resnet50
    "ResNet-44.55M": WResNetModelConfig(224, 101, 64, 1, "fp16"), # resnet101
    "ResNet-60.19M": WResNetModelConfig(224, 152, 64, 1, "fp16"), # resnet152
    # WResNet models
    "WResNet-68.88M": WResNetModelConfig(224, 50, 64, 2, "fp16"), # wresnet50-2
    "WResNet-126.88M": WResNetModelConfig(224, 101, 64, 2, "fp16"), # wresnet101-2
    
    # "WResNet-250M": WResNetModelConfig(224, 50, 160, 2, "fp16"),
    # "WResNet-500M": WResNetModelConfig(224, 50, 224, 2, "fp16"),
    # "WResNet-1B": WResNetModelConfig(224, 50, 320, 2, "fp16"),
    # "WResNet-2B": WResNetModelConfig(224, 50, 448, 2, "fp16"),
    # "WResNet-4B": WResNetModelConfig(224, 50, 640, 2, "fp16"),
    # "WResNet-6.8B": WResNetModelConfig(224, 50, 320, 16, "fp16"),
    # "WResNet-13B": WResNetModelConfig(224, 101, 320, 16, "fp16"),
}

all_layers = {
        # GPT models
    "GPT-125M": 12,
    "GPT-350M": 24,
    "GPT-760M": 24,
    "GPT-1.3B": 24,
    "GPT-2.6B": 24,
    "GPT-6.7B": 24,
    # "GPT-15B": 48,
    # "GPT-39B": 48,
    # "GPT-76B": 60,

    # BERT models
    "BERT-Tiny": 2,
    "BERT-Mini": 4,
    "BERT-Small": 4,
    "BERT-Medium": 8,
    "BERT-Base": 12,
    "BERT-Large": 12,
    "BERT-LL": 12,
    # "BERT-LLL": 24,
    # "BERT-LLLL": BERTModelConfig(512, 2560, 32, 32, 30522),
    # "BERT-LLLLL": BERTModelConfig(512, 4096, 32, 32, 30522),
    # "BERT-LLLLLL": BERTModelConfig(512, 5120, 48, 40, 30522),
    # "BERT-LLLLLLL": BERTModelConfig(512, 8192, 48, 64, 30522),
    # "BERT-LLLLLLLL": BERTModelConfig(512, 10240, 60, 80, 30522),

    # ResNet models
    "ResNet-25.56M": 12,  # resnet50
    "ResNet-44.55M": 12, # resnet101
    "ResNet-60.19M": 12, # resnet152
    # WResNet models
    "WResNet-68.88M": 12, # wresnet50-2
    "WResNet-126.88M": 12, # wresnet101-2
    
    # "WResNet-250M": WResNetModelConfig(224, 50, 160, 2, "fp16"),
    # "WResNet-500M": WResNetModelConfig(224, 50, 224, 2, "fp16"),
    # "WResNet-1B": WResNetModelConfig(224, 50, 320, 2, "fp16"),
    # "WResNet-2B": WResNetModelConfig(224, 50, 448, 2, "fp16"),
    # "WResNet-4B": WResNetModelConfig(224, 50, 640, 2, "fp16"),
    # "WResNet-6.8B": WResNetModelConfig(224, 50, 320, 16, "fp16"),
    # "WResNet-13B": WResNetModelConfig(224, 101, 320, 16, "fp16"),
}

prefer_reduce_scatter = True
use_remat = True
force_dp_dict = {"force_batch_dim_to_mesh_dim": 0}



def get_solution_cases(model_spec, num_micro_batches, num_auto_layers,
                      forward_stage_layer_ids, submesh_physical_shapes,
                      submesh_logical_shapes,
                      submesh_autosharding_option_dicts, batch_size):
    return[
        BenchmarkCase(
            batch_size, model_spec, num_micro_batches,
            "load_solution",
            LoadSolutionParallelArgs(prefer_reduce_scatter, use_remat,
                                     num_auto_layers, forward_stage_layer_ids,
                                     submesh_physical_shapes,
                                     submesh_logical_shapes,
                                     submesh_autosharding_option_dicts))
    ]

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

wsc_perf_suite = {
#             [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]],
    25: get_solution_cases(batch_size=1000,
        model_spec=all_models.values(),
                           num_micro_batches=10, num_auto_layers=10,
            forward_stage_layer_ids=[[0, 1], [2, 3], [4, 5], [6, 7], [8,9]],
            submesh_physical_shapes=[(1, 5)] * 5, submesh_logical_shapes=[(1, 5)] * 5,
             submesh_autosharding_option_dicts=[force_dp_dict] * 5),

    24: flatten_list([

        # NOTE: fit for small models with layer=12
        get_solution_cases(batch_size = 1536,
        model_spec=mod,num_micro_batches=12,
        num_auto_layers = layers,forward_stage_layer_ids=[[i for i in range(layers)]],
        submesh_physical_shapes=
        [(6, 4)],submesh_logical_shapes= [(4, 6)],
        submesh_autosharding_option_dicts= [{}] * 1) for mod, layers in zip(all_models.values(), all_layers.values())


        # get_solution_cases(batch_size = 1536,
        #    # auto_layers max value = 16, otherwise != num_layers
        #    model_specs=all_models.values(),num_micro_batches=12,
        #    num_auto_layers = 24,forward_stage_layer_ids=[[i for i in range(24)]],
        #    submesh_physical_shapes=
        #    [(6, 4)],submesh_logical_shapes= [(2, 2), (2, 2), (4, 2), (4, 2)],
        #      submesh_autosharding_option_dicts= [{}] * 1),
        # Correct!
        #get_solution_cases(batch_size = 1536,
         #   # auto_layers max value = 16, otherwise != num_layers
          #  model_specs=all_models.values(),num_micro_batches=12,
           # num_auto_layers = 24,forward_stage_layer_ids=[[i for i in range(24)]],
           # submesh_physical_shapes=
           # [(6, 4)],submesh_logical_shapes= [(24, 1)],
            #  submesh_autosharding_option_dicts= [{}] * 1),
            
        #     # Correct
        #     get_solution_cases(batch_size = 1536,
        #     model_specs=all_models.values(),num_micro_batches=12,
        #     num_auto_layers = 24,forward_stage_layer_ids=[[i for i in range(24)]],
        #     submesh_physical_shapes=
        #   [(6, 4)],submesh_logical_shapes= [(1, 24)],
        #      submesh_autosharding_option_dicts= [{}] * 1),

        # get_solution_cases(batch_size = 1536,
        #    # auto_layers max value = 16, otherwise != num_layers
        #    model_specs=all_models.values(),num_micro_batches=12,
        #    num_auto_layers = 24,forward_stage_layer_ids=[[i] for i in range(24)],
        #    submesh_physical_shapes=
        #    [(1, 1)*24],submesh_logical_shapes= [(1, 1)*24],
        #      submesh_autosharding_option_dicts= [{}] * 24),

        # #    Correct 
        #    get_solution_cases(batch_size = 1536,
        #    model_specs=all_models.values(),num_micro_batches=12,
        #    num_auto_layers = 24,forward_stage_layer_ids=[[i for i in range(24)]],
        #    submesh_physical_shapes=
        #    [(6, 4)],submesh_logical_shapes= [(4, 6)],
        #     submesh_autosharding_option_dicts= [{}] * 1),

        #    # Wrong!
        # get_solution_cases(batch_size = 1536,
        #    model_specs=all_models.values(),num_micro_batches=12,
        #    num_auto_layers = 24,forward_stage_layer_ids=[[i for i in range(24)]],
        #    submesh_physical_shapes=
        #    [(6, 4)],submesh_logical_shapes= [(6, 4)],
        #      submesh_autosharding_option_dicts= [{}] * 1),
        #    #NOTE: (2, 12) Correct
        #     get_solution_cases(batch_size = 1536,
        #     model_specs=all_models.values(),num_micro_batches=12,
        #     num_auto_layers = 24,forward_stage_layer_ids=[[i for i in range(24)]],
        #     submesh_physical_shapes=
        #     [(6, 4)],submesh_logical_shapes= [(12, 2)],
        #       submesh_autosharding_option_dicts= [force_dp_dict] * 1),

            ]),
}

