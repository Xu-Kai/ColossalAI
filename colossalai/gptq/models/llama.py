from dataclasses import dataclass, field, fields


@dataclass
class GPTQLlamaConfig():
    layer_name = "LlamaDecoderLayer"
    layer_blocks = "model.layers"
    linear_names = [["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"],
                    ["mlp.up_proj", "mlp.gate_proj"], ["mlp.down_proj"]]
    model_names = ["model.embed_tokens", "model.norm"]
    attention = "self_attn"
    mlp = "mlp"
