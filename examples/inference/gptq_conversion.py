# from transformers import AutoTokenizer, TextGenerationPipeline
import argparse
import logging
import warnings

from transformers import LlamaForCausalLM, LlamaTokenizer

from colossalai.inference.quant.gptq import GPTQQuantizeConfig, load_quantized, quantize, save_quantized
from tests.kit.model_zoo import model_zoo

logging.basicConfig(format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")

import torch

# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

gptq_temp_state_buffer = None
gptq_temp_dq_buffer = None


def _post_init_gptq_buffer(model: torch.nn.Module, quantized_config) -> None:
    from colossalai.inference.quant.gptq.cai_gptq import CaiQuantLinear
    HAS_GPTQ_CUDA = False
    try:
        from colossalai.kernel.op_builder.gptq import GPTQBuilder
        gptq_cuda = GPTQBuilder().load()
        HAS_GPTQ_CUDA = True
    except ImportError:
        warnings.warn('CUDA gptq is not installed')
        HAS_GPTQ_CUDA = False
    max_dq_buffer_size = 1
    max_inner_outer_dim = 1
    use_act_order = quantized_config.desc_act
    print("use_act_order", use_act_order)
    for name, submodule in model.named_modules():
        if isinstance(submodule, CaiQuantLinear):
            max_dq_buffer_size = max(max_dq_buffer_size, submodule.qweight.numel() * 8)

            if use_act_order:
                max_inner_outer_dim = max(max_inner_outer_dim, submodule.infeatures, submodule.outfeatures)
            bits = submodule.bits
    if not (HAS_GPTQ_CUDA and bits == 4):
        return
    global gptq_temp_state_buffer, gptq_temp_dq_buffer
    max_input_len = 1
    if use_act_order:
        max_input_len = 1024
    # The temp_state buffer is required to reorder X in the act-order case.
    # The temp_dq buffer is required to dequantize weights when using cuBLAS, typically for the prefill.
    gptq_temp_state_buffer = torch.zeros((max_input_len, max_inner_outer_dim),
                                         dtype=torch.float16,
                                         device=torch.cuda.current_device())
    gptq_temp_dq_buffer = torch.zeros((1, max_dq_buffer_size), dtype=torch.float16, device=torch.cuda.current_device())

    gptq_cuda.prepare_buffers(torch.device(torch.cuda.current_device()), gptq_temp_state_buffer, gptq_temp_dq_buffer)
    # Using the default from exllama repo here.
    matmul_recons_thd = 8
    matmul_fused_remap = False
    matmul_no_half2 = False
    gptq_cuda.set_tuning_params(matmul_recons_thd, matmul_fused_remap, matmul_no_half2)

    torch.cuda.empty_cache()


def gptq_convert(args):

    pretrained_model_dir = args.pretrined_path

    quantized_model_dir = args.save_path

    pretrained_model_dir = "/home/lcxk/data3/llama-7b-hf"
    quantized_model_dir = "/home/lcxk/data3/test_tp_infer/model_data"

    tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_dir)

    examples = [
        tokenizer(
            "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm.")
    ]

    model = LlamaForCausalLM.from_pretrained(pretrained_model_dir, pad_token_id=tokenizer.eos_token_id)
    generate_kwargs = dict(max_new_tokens=64, do_sample=False)

    quantize_config = GPTQQuantizeConfig(
        bits=4,    # quantize model to 4-bit
        group_size=128,    # it is recommended to set the value to 128
        desc_act=False,    # set to False can significantly speed up inference but the perplexity may slightly bad
    )
    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    quantize(model, quantize_config, examples)
    save_quantized(model, quantize_config, quantized_model_dir)
    model.cuda()
    _post_init_gptq_buffer(model, quantize_config)
    print(
        tokenizer.decode(
            model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device), **generate_kwargs)[0]))


def test_gptq_convert(args):

    pretrained_model_dir = args.pretrined_path

    quantized_model_dir = args.save_path

    pretrained_model_dir = "/home/lcxk/data3/llama-7b-hf"
    quantized_model_dir = "/home/lcxk/data3/test_tp_infer/model_test"

    tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_dir)

    examples = [
        tokenizer(
            "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm.")
    ]

    sub_model_zoo = model_zoo.get_sub_registry('transformers_llama_for_casual_lm')
    generate_kwargs = dict(max_new_tokens=64, do_sample=False)

    for name, (model_fn, data_gen_fn, _, _, _) in sub_model_zoo.items():
        model = model_fn()
        model = model.half()
        data = data_gen_fn()
        examples = data

        quantize_config = GPTQQuantizeConfig(
            bits=4,    # quantize model to 4-bit
            group_size=128,    # it is recommended to set the value to 128
            desc_act=False,    # set to False can significantly speed up inference but the perplexity may slightly bad
        )
        # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
        model.cuda()

        quantize(model, quantize_config, data)

        save_quantized(model, quantize_config, quantized_model_dir)
        model = load_quantized(quantized_model_dir)
        model.cuda()
        _post_init_gptq_buffer(model, quantize_config)
        print(
            tokenizer.decode(
                model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device),
                               **generate_kwargs)[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pp', '--pretrained_path', type=str, help='Model path', required=True)
    parser.add_argument('-sp', '--save_path', type=str, help='Model path', required=True)

    args = parser.parse_args()

    gptq_convert(args)
