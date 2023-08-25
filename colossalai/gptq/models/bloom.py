from dataclasses import dataclass, field, fields


@dataclass
class GPTQBloomConfig():
    layer_name = "BloomBlock"
    layer_blocks = "transformer.h"
    linear_names = [["self_attention.query_key_value"], ["self_attention.dense"], ["mlp.dense_h_to_4h"],
                    ["mlp.dense_4h_to_h"]]
    model_names = ["transformer.word_embeddings", "transformer.word_embeddings_layernorm", "transformer.ln_f"]
    attention = "self_attention"
    mlp = "mlp"
