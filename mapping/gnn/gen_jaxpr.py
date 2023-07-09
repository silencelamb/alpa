from alpa.model.bert_model import BertConfig, FlaxBertForMaskedLMModule
from alpa.model.gpt_model import FlaxGPTForLMModule
from benchmark.alpa.suite_auto_gpt import gpt_specs
from benchmark.alpa.suite_auto_moe import moe_specs
from benchmark.alpa.suite_wresnet import wresnet_specs
from benchmark.alpa.suite_manual_gpt import GPTModelConfig
import jax.numpy as jnp
import jax

def gen_gpt_jaxpr():
    for model_param, config in gpt_specs.items():
        print(model_param, config)
        (seq_len, hidden_size, num_layers, num_heads,
        vocab_size) = config
        dtype = jnp.float16
        batch_size = 8
        add_manual_remat = False
        bert_config = BertConfig(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    num_attention_heads=num_heads,
                    intermediate_size=hidden_size * 4,
                    num_hidden_layers=num_layers,
                    type_vocab_size=0,
                    tie_word_embeddings=False,
                    gradient_checkpointing=add_manual_remat,
                    add_manual_pipeline_markers=False
                )    
        batch = {
            "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "token_type_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "position_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        }
        rngkey = jax.random.PRNGKey(0)
        model = FlaxGPTForLMModule(bert_config, dtype=dtype)
        params = model.init_dummy(rngkey, batch["input_ids"],
                            batch["attention_mask"], batch["token_type_ids"],
                            batch["position_ids"])
        jaxpr = jax.make_jaxpr(model.apply)(params, **batch)
        return jaxpr


if __name__ == '__main__':
    jaxpr = gen_gpt_jaxpr()
    import pdb; pdb.set_trace()
    print(jaxpr)
        