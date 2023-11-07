import os
import time

import numpy as np
import torch

GB = 1 << 30


def write_tsv(heads, values, filename, print_line=True):
    """Write tsv data to a file."""
    assert len(heads) == len(values)

    values = [str(x) for x in values]

    with open(filename, "a") as fout:
        fout.write("\t".join(values) + "\n")

    if print_line:
        line = ""
        for i in range(len(heads)):
            line += heads[i] + ": " + values[i] + "  "
        print(line)


def benchmark_func(run_func, sync_func=None, warmup=1, repeat=3, number=5):
    """Benchmark the execution time of a function."""
    costs = []

    # Warmup
    for i in range(warmup):
        run_func()

    # Benchmark
    for i in range(repeat):
        if sync_func:
            sync_func()
        tic = time.time()

        for j in range(number):
            run_func()

        if sync_func:
            sync_func()
        costs.append(time.time() - tic)

    return np.array(costs) / number


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


def get_torch_memory_usage(print_info=False):
    """Get accurate gpu memory usage by querying torch runtime"""
    import torch
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    if print_info:
        print("allocated: %.2f GB" % (allocated / GB), flush=True)
        print("reserved:  %.2f GB" % (reserved / GB), flush=True)
    return allocated


# NOTE: generated by GPT-4 -- 
def compute_wresnet_tflops(batch_size, image_size, num_layers, 
                            num_channels, width_factor, num_gpus,
                            latency, backward=True, checkpoint_activations=False):

    from torchvision.models import resnet50, resnet101
    from thop import profile
    model = resnet50()
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input, ))

    return 0, 0


import torch.nn as nn


def generate_transformer_input(model_tokenizer, input_shape, device="cpu"):
    """Automatically generates data in the form of transformes model input format.
    
    Args:
        input_shape (tuple):transformers model input shape: (batch_size, seq_len).
        tokenizer (transformer.model.tokenization): transformers model tokenization.tokenizer.

    Returns:
        dict: data format of transformers model input, it is a dict contain 'input_ids', 'attention_mask', sometime contain 'token_type_ids'.
    """

    if input_shape is None:
        input_shape = [1, 128]  # defautl (batch_size=1, seq_len=128)

    max_length = input_shape[1]
    model_input_ids = []
    model_attention_mask = []
    model_token_type_ids = []
    model_position_ids = []

    inp_seq = ""
    for _ in range(input_shape[0]):
        inputs = model_tokenizer.encode_plus(
            inp_seq,
            add_special_tokens=True,
            # truncation_strategy='longest_first',
            max_length=max_length,
        )
        origin_length = len(inputs["input_ids"])
        padding_length = max_length - origin_length

        for key in inputs.keys():
            if key == "input_ids":
                input_ids = inputs["input_ids"]
                pad_token = model_tokenizer.pad_token_id if model_tokenizer.pad_token_id else 0
                input_ids = input_ids + ([pad_token] * padding_length)
                assert len(input_ids) == max_length, "len(input_ids) must equal max_length"
                model_input_ids.append(input_ids)
            elif key == "attention_mask":
                attention_mask = [1] * origin_length
                attention_mask = attention_mask + ([0] * padding_length)
                assert len(attention_mask) == max_length, "len(attention_mask) must equal max_length"
                model_attention_mask.append(attention_mask)
            elif key == "token_type_ids":
                token_type_ids = inputs['token_type_ids']
                pad_token_segment_id = 0
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
                assert len(token_type_ids) == max_length, "len(token_type_ids) must equal max_length"
                model_token_type_ids.append(token_type_ids)
            elif key == "position_ids":  # chatglm2 use position id
                position_ids = inputs['position_ids']
                for i in range(origin_length, max_length):
                    position_ids.append(i)
                assert len(position_ids) == max_length, "len(position_ids) must equal max_length"
                model_position_ids.append(position_ids)

    # Batch size input_shape[0], sequence length input_shape[128]
    inputs = {}
    if len(model_input_ids) > 0:
        inputs.update({"input_ids": torch.tensor(model_input_ids).to(device)})
    if len(model_attention_mask) > 0:
        inputs.update({"attention_mask": torch.tensor(model_attention_mask).to(device)})
    if len(model_token_type_ids) > 0:
        inputs.update({'token_type_ids': torch.tensor(model_token_type_ids).to(device)})
    if len(model_position_ids) > 0:
        inputs.update({'position_ids': torch.tensor(model_position_ids).to(device)})

    return inputs.values()

def compute_bert(batch_size,
                       seq_len,
                       num_layers,
                       hidden_size,
                       vocab_size,
                       num_heads,
                       num_gpus,
                       latency,
                       backward=True,
                       checkpoint_activations=False):
    
    from transformers import BertModel, BertTokenizer, BertConfig
    import torch
    from calflops import calculate_flops
    from thop import profile

    # 加载预训练的BERT模型和分词器
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name, max_length=1024)
    model = BertModel(
        config=BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads
        )
    )
    inputs = generate_transformer_input(tokenizer, (batch_size, seq_len))


    flops, macs, params = calculate_flops(model=model, 
                                    input_shape=(batch_size, seq_len), transformer_tokenizer=tokenizer)
    # with torch.no_grad():
    #     # inputs = (input_ids_batch, label_ids_batch, mask_ids_batch, fast_mode)
    #     macs, params = profile(model, inputs, verbose=False)
    
    return flops, params
    # # # 输入文本
    # text = "Hello, how are you?"

    # # # 分词和编码
    # tokens = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
    # input_ids = tokens['input_ids']
    # attention_mask = tokens['attention_mask']

    # # 前向传播
    # outputs = model(input_ids, attention_mask=attention_mask)

    # # 获取BERT模型的输出
    # last_hidden_state = outputs.last_hidden_state
    # pooler_output = outputs.pooler_output

    # # 打印输出
    # print("Last hidden state shape:", last_hidden_state.shape)
    # print("Pooler output shape:", pooler_output.shape)

    # import jax
    # # wrapped = jax.xla_computation(fun)
    # wrapped = jax.jit(mod)
    # hlo = wrapped(*args).as_hlo_module()
    # client = jax.lib.xla_bridge.get_backend()
    # ans = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, hlo)


    # factor = 24
    # if backward:
    #     factor += 48
    # if checkpoint_activations:
    #     factor += 24

    # total_flop = factor * batch_size * seq_len * (hidden_size ** 2) * num_layers * \
    #       (1 + seq_len / (6 * hidden_size)) \
    #       + 6 * batch_size * seq_len * hidden_size * vocab_size
    # total_tflops = total_flop / 1e12
    # Note: The above formula does not count the first embedding table lookup
    # because it is a sparse operation.
    # If we use dense dot to compute the first embedding table lookup,
    # then the last term in total_flops should be
    # "+ 10 * batch_size * seq_len * hidden_size * vocab_size".
    # tflops = total_flop / latency / num_gpus / 1e12
    # return tflops, total_tflops



def compute_gpt_parameter_count(num_layers, hidden_size, vocab_size):
    return num_layers * (
        # self-attention
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) +
        # mlp
        hidden_size * (4 * hidden_size + 1) + hidden_size * 4 *
        (hidden_size + 1) +
        # layer norm
        hidden_size * 4) + vocab_size * (hidden_size + 1)


def compute_gpt_tflops(batch_size,
                       seq_len,
                       num_layers,
                       hidden_size,
                       vocab_size,
                       num_gpus,
                       latency,
                       backward=True,
                       checkpoint_activations=False):
    factor = 24
    if backward:
        factor += 48
    if checkpoint_activations:
        factor += 24

    total_flop = factor * batch_size * seq_len * (hidden_size ** 2) * num_layers * \
          (1 + seq_len / (6 * hidden_size)) \
          + 6 * batch_size * seq_len * hidden_size * vocab_size
    total_tflops = total_flop / 1e12
    # Note: The above formula does not count the first embedding table lookup
    # because it is a sparse operation.
    # If we use dense dot to compute the first embedding table lookup,
    # then the last term in total_flops should be
    # "+ 10 * batch_size * seq_len * hidden_size * vocab_size".
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops, total_tflops


def compute_mlp_tflops(batch_size,                       
                       num_layers,
                       hidden_size,                       
                       num_gpus,
                       latency,
                       backward=True,
                       checkpoint_activations=False):
    factor = 24
    if backward:
        factor += 48
    if checkpoint_activations:
        factor += 24
    # import ipdb; ipdb.set_trace()
    total_flop = factor * batch_size * (hidden_size ** 2) * num_layers + 6 * batch_size  * hidden_size 
    # Note: The above formula does not count the first embedding table lookup
    # because it is a sparse operation.
    # If we use dense dot to compute the first embedding table lookup,
    # then the last term in total_flops should be
    # "+ 10 * batch_size * seq_len * hidden_size * vocab_size".
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops


def compute_moe_tflops(batch_size,
                       seq_len,
                       num_layers,
                       hidden_size,
                       group_size,
                       vocab_size,
                       num_expert,
                       num_gpus,
                       latency,
                       mlp_factor=8,
                       checkpoint_activations=False):
    factor = 4 if checkpoint_activations else 3
    # num_layers / 2 attention block
    pure_transformer = batch_size * seq_len * (hidden_size ** 2) * (8 + 4 * mlp_factor) +\
        4 * batch_size * (seq_len ** 2) * hidden_size
    pure_transformer = pure_transformer * factor

    # num_layers / 2 attention-moe block
    # transformer
    moe_transformer = batch_size * seq_len * (hidden_size ** 2) * 8  +\
        4 * batch_size * (seq_len ** 2) * hidden_size
    # expert FFNs:
    # moe_transformer += 2 * batch_size * seq_len * (hidden_size ** 2) * mlp_factor * 2
    moe_transformer += 8 * batch_size * seq_len * (hidden_size**2) * mlp_factor

    # softmax
    moe_transformer += 2 * batch_size * seq_len * hidden_size * num_expert
    # top-2 gating
    moe_transformer += 2 * (batch_size * seq_len) * 2 * group_size
    # dispatch + combine
    moe_transformer += 2 * batch_size * seq_len * hidden_size * 2 * group_size * 2

    moe_transformer = moe_transformer * factor

    # vocab
    embedding = 6 * batch_size * seq_len * hidden_size * vocab_size

    total_flop = pure_transformer * num_layers / 2 + \
                 moe_transformer * num_layers / 2 + embedding
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops

# NOTE: generated by GPT-4
def compute_wresnet_parameter_count(image_size, num_layers, num_channels, width_factor):
    return num_layers * (num_channels*width_factor + 1)

def compute_bert_parameter_count(seq_len, hidden_size, num_layers, num_heads, vocab_size):
    # Refer to https://stackoverflow.com/questions/64485777/how-is-the-number-of-parameters-be-calculated-in-bert-model
    # (512, 768, 12, 12, 30522)
    return (    
        # Embedding Matrices 
        vocab_size*hidden_size + seq_len*hidden_size+2*hidden_size+2*hidden_size+ 
        # Total parameters for 12 layer
        num_layers* (
        # Attention head
        num_heads*(3*(hidden_size*(hidden_size/num_heads+1)))+
        # Dense weight+Layer Norm
        hidden_size*(hidden_size+1)+2*hidden_size+
        # Position wise feedforward
        (4*hidden_size)*(hidden_size+1+hidden_size)+hidden_size+
        2*hidden_size
        )+
        # output layer
        hidden_size*(hidden_size+1)
    )


def compute_mlp_parameter_count(num_layers, hidden_size):
    return num_layers * (hidden_size * (4 * hidden_size + 1) + hidden_size * 4 *(hidden_size + 1) )
       


def compute_moe_parameter_count(num_layers,
                                hidden_size,
                                vocab_size,
                                num_expert,
                                mlp_factor=8,
                                tie_embedding=True):
    pure_transformer = \
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) + \
        hidden_size * (mlp_factor * hidden_size + 1) + hidden_size * mlp_factor * (hidden_size + 1) + \
        hidden_size * 4
    moe_transformer = \
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) + \
        num_expert * (hidden_size * (mlp_factor * hidden_size + 1) + hidden_size * mlp_factor * (hidden_size + 1)) + \
        hidden_size * 4

    # embedding
    embedding_factor = 1 if tie_embedding else 2
    embedding = embedding_factor * vocab_size * (hidden_size + 1)

    if num_expert == 1:
        return pure_transformer * num_layers + embedding
    else:
        half = num_layers / 2
        return half * pure_transformer + half * moe_transformer + embedding
