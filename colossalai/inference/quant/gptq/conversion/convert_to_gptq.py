import copy
import json
from dataclasses import dataclass, field, fields
from logging import getLogger
from os.path import isdir, isfile, join
from typing import Dict, List, Optional, Union

import accelerate
import torch
import torch.nn as nn
import transformers
from accelerate.hooks import remove_hook_from_module
from safetensors.torch import save_file as safe_save
from torch import LongTensor
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import no_init_weights
from transformers.utils.generic import ContextManagers
from transformers.utils.hub import CommitOperationAdd, PushToHubMixin, cached_file, create_commit, create_repo

from ._utils import (
    CPU,
    CUDA_0,
    SUPPORTED_MODELS,
    find_layers,
    get_device,
    get_module_by_name_prefix,
    get_module_by_name_suffix,
    make_quant,
    make_sure_no_tensor_in_meta_device,
    move_to_device,
    pack_model,
)
from .gptq import GPTQ

logger = getLogger(__name__)

import os


@dataclass
class LlamaLayers():
    layer_type = "LlamaDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"],
                            ["mlp.up_proj", "mlp.gate_proj"], ["mlp.down_proj"]]


@dataclass
class BloomLayers():
    layer_type = "BloomBlock"
    layers_block_name = "transformer.h"
    outside_layer_modules = ["transformer.word_embeddings", "transformer.word_embeddings_layernorm", "transformer.ln_f"]
    inside_layer_modules = [["self_attention.query_key_value"], ["self_attention.dense"], ["mlp.dense_h_to_4h"],
                            ["mlp.dense_4h_to_h"]]


models_map = {"llama": LlamaLayers, "bloom": BloomLayers}


@dataclass
class GPTQQuantizeConfig(PushToHubMixin):
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})
    group_size: int = field(default=-1)
    damp_percent: float = field(default=0.01)
    desc_act: bool = field(default=True)
    static_groups: bool = field(default=False)
    sym: bool = field(default=True)
    true_sequential: bool = field(default=True)
    model_name_or_path: Optional[str] = field(default=None)
    model_file_base_name: Optional[str] = field(default=None)

    def __post_init__(self):
        fields_info = fields(self)

        if self.bits not in fields_info[0].metadata["choices"]:
            raise ValueError(f"only support quantize to {fields_info[0].metadata['choices']} bits.")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("unless equal to -1, group_size must greater then 0.")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

    def save_pretrained(self, save_dir: str, **kwargs):
        with open(join(save_dir, "quantize_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        commit_hash = kwargs.pop("_commit_hash", None)

        quantize_config_filename = "quantize_config.json"
        if os.path.isdir(save_dir):    # Local
            resolved_config_file = join(save_dir, quantize_config_filename)
        else:    # Remote
            resolved_config_file = cached_file(
                save_dir,
                quantize_config_filename,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                use_auth_token=use_auth_token,
                revision=revision,
                local_files_only=local_files_only,
                subfolder=subfolder,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                _commit_hash=commit_hash,
            )

        with open(resolved_config_file, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    def to_dict(self):
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "damp_percent": self.damp_percent,
            "desc_act": self.desc_act,
            "static_groups": self.static_groups,
            "sym": self.sym,
            "true_sequential": self.true_sequential,
            "model_name_or_path": self.model_name_or_path,
            "model_file_base_name": self.model_file_base_name,
        }


def _prepare_examples_for_quantization(
    model_config,
    examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
    batch_size: int = 1,
):

    def _convert_tensor_to_list(tensor):
        if isinstance(tensor, torch.Tensor):
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.long()
            return tensor.cpu().numpy().tolist()
        return [tensor]

    new_examples = []
    for example in examples:
        input_ids = _convert_tensor_to_list(example["input_ids"])
        attention_mask = _convert_tensor_to_list(example["attention_mask"])
        if "labels" in example:
            labels = _convert_tensor_to_list(example["labels"])
        elif "label" in example:
            labels = _convert_tensor_to_list(example["label"])
        elif "label_ids" in example:
            labels = _convert_tensor_to_list(example["label_ids"])
        else:
            labels = copy.deepcopy(input_ids)
        new_examples.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
    pad_token_id = model_config.pad_token_id
    if not pad_token_id:
        pad_token_id = model_config.eos_token_id

    new_examples = [
        collate_data(new_examples[start:start + batch_size], pad_token_id)
        for start in range(0, len(new_examples), batch_size)
    ]
    for new_example in new_examples:
        del new_example["labels"]

    return new_examples


def collate_data(blocks: List[Dict[str, List[List[int]]]], pad_token_id: int) -> Dict[str, LongTensor]:

    def pad_block(block, pads):
        return torch.cat((pads.to(block.device), block), dim=-1)

    input_ids_blocks = [LongTensor(block["input_ids"]) for block in blocks]
    attention_mask_blocks = [LongTensor(block["attention_mask"]) for block in blocks]
    label_blocks = [LongTensor(block["labels"]) for block in blocks]

    bsz = len(blocks)
    inp_max_len = max([block.size(-1) for block in input_ids_blocks])
    label_max_len = max([block.size(-1) for block in label_blocks])

    for i in range(bsz):
        block_bsz, block_inp_len = input_ids_blocks[i].shape
        block_label_len = label_blocks[i].shape[-1]
        pad_num = inp_max_len - block_inp_len
        if pad_num > 0:
            input_ids_blocks[i] = pad_block(input_ids_blocks[i], torch.ones((block_bsz, pad_num)) * pad_token_id)
            attention_mask_blocks[i] = pad_block(attention_mask_blocks[i], torch.zeros((block_bsz, pad_num)))
        label_pad_num = label_max_len - block_label_len
        if label_pad_num > 0:
            label_blocks[i] = pad_block(label_blocks[i], torch.ones((block_bsz, label_pad_num)) * -100)

    return {
        "input_ids": torch.cat(input_ids_blocks, dim=0).long(),
        "attention_mask": torch.cat(attention_mask_blocks, dim=0).long(),
        "labels": torch.cat(label_blocks, dim=0).long()
    }


@torch.inference_mode()
def quantize(model: PreTrainedModel,
             quantize_config: GPTQQuantizeConfig,
             examples: List[Dict[str, Union[List[int], torch.LongTensor]]],
             batch_size: int = 1,
             use_cuda_fp16: bool = True,
             autotune_warmup_after_quantized: bool = False,
             cache_examples_on_gpu: bool = True):

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        for name, device in device_map.items():
            if device == "cpu":
                module = get_module_by_name_suffix(model, name)
                remove_hook_from_module(module, recurse=True)
                accelerate.cpu_offload_with_hook(module, CUDA_0)

    layer_inputs = []
    attention_masks = []
    position_ids = []
    layer_input_kwargs = []
    layer_outputs = []

    examples = _prepare_examples_for_quantization(model.config, examples, batch_size)
    model_block_names = models_map[model.config.model_type]()
    print(model.config)

    class LayerHijacker(nn.Module):
        """hijack layer's forward pass to cache data"""

        def __init__(self, m, device):
            super().__init__()
            self.module = m
            self.data_device = device if cache_examples_on_gpu else CPU

        def forward(self, inp=None, **kwargs):
            if inp is None:    # some models use all key-value arguments in forward pass call
                for kwarg_name in ["hidden_states"]:
                    if kwarg_name in kwargs:
                        inp = kwargs[kwarg_name]
                        break
            layer_inputs.append(move_to_device(inp, self.data_device))
            attention_masks.append(kwargs["attention_mask"].to(self.data_device))
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to_device(pos_ids, self.data_device))
            one_kwargs = dict()
            for k, v in kwargs.items():    # make sure other arguments also be captured
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    if isinstance(v, torch.Tensor):
                        one_kwargs[k] = move_to_device(v, self.data_device)
                    else:
                        one_kwargs[k] = v
            layer_input_kwargs.append(one_kwargs)
            raise ValueError

    forward_pass_use_cache = model.config.use_cache
    model.config.use_cache = False

    num_batches = len(examples)
    layers = get_module_by_name_prefix(model, model_block_names.layers_block_name)

    force_layer_back_to_cpu = False
    if get_device(layers[0]) == CPU:
        layers[0] = layers[0].to(CUDA_0)
        force_layer_back_to_cpu = True

    cur_layer_device = get_device(layers[0])
    ori_outside_layer_module_devices = {}
    for module_name in model_block_names.outside_layer_modules:
        module = get_module_by_name_prefix(model, module_name)

        if module is None:
            continue

        ori_outside_layer_module_devices[module_name] = get_device(module)
        if module is not None:
            move_to_device(module, cur_layer_device)

    # get inputs for first layer
    layers[0] = LayerHijacker(layers[0], cur_layer_device)
    for example in examples:
        for k, v in example.items():
            if len(v.shape) == 1:
                v = v.unsqueeze(0)
            example[k] = move_to_device(v, cur_layer_device)
        try:
            model(**example)
        except ValueError:
            pass
    layers[0] = layers[0].module

    move_to_device(layers[0], CPU if force_layer_back_to_cpu else cur_layer_device)
    for module_name in model_block_names.outside_layer_modules:
        module = get_module_by_name_prefix(model, module_name)
        if module is not None:
            move_to_device(module, ori_outside_layer_module_devices[module_name])

    torch.cuda.empty_cache()

    inside_layer_modules = model_block_names.inside_layer_modules
    if not quantize_config.true_sequential:
        inside_layer_modules = [sum(inside_layer_modules, [])]
    quantizers = {}
    for i in range(len(layers)):
        print(f"Start quantizing layer {i + 1}/{len(layers)}")
        layer = layers[i]
        force_layer_back_to_cpu = False
        if get_device(layer) == CPU:
            move_to_device(layer, CUDA_0)
            force_layer_back_to_cpu = True
        cur_layer_device = get_device(layer)

        full = find_layers(layer)
        for names in inside_layer_modules:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer.configure(
                    quantize_config.bits,
                    perchannel=True,
                    sym=quantize_config.sym,
                    mse=False,
                )

            def add_batch(name):

                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(num_batches):
                layer_input = move_to_device(layer_inputs[j], cur_layer_device)
                layer_attention_mask = move_to_device(attention_masks[j], cur_layer_device)
                additional_layer_inputs = {"attention_mask": layer_attention_mask}
                layer_position_ids = None if not position_ids else move_to_device(position_ids[j], cur_layer_device)
                if layer_position_ids is not None:
                    additional_layer_inputs["position_ids"] = layer_position_ids
                for k, v in layer_input_kwargs[j].items():
                    if isinstance(v, torch.Tensor):
                        additional_layer_inputs[k] = move_to_device(v, cur_layer_device)
                    else:
                        additional_layer_inputs[k] = v
                layer(layer_input, **additional_layer_inputs)
            for h in handles:
                h.remove()

            for name in subset:
                print(f'Quantizing {name} in layer {i + 1}/{len(layers)}...')
                scale, zero, g_idx = gptq[name].fasterquant(percdamp=quantize_config.damp_percent,
                                                            group_size=quantize_config.group_size,
                                                            actorder=quantize_config.desc_act,
                                                            static_groups=quantize_config.static_groups)
                quantizers[f'{model_block_names.layers_block_name}.{i}.{name}'] = (
                    gptq[name].quantizer.to(CPU if force_layer_back_to_cpu else cur_layer_device),
                    move_to_device(scale, CPU if force_layer_back_to_cpu else cur_layer_device),
                    move_to_device(zero, CPU if force_layer_back_to_cpu else cur_layer_device),
                    move_to_device(g_idx, CPU if force_layer_back_to_cpu else cur_layer_device))
                gptq[name].free()

        for j in range(num_batches):
            layer_input = move_to_device(layer_inputs[j], cur_layer_device)
            layer_attention_mask = move_to_device(attention_masks[j], cur_layer_device)
            additional_layer_inputs = {"attention_mask": layer_attention_mask}
            layer_position_ids = None if not position_ids else move_to_device(position_ids[j], cur_layer_device)
            if layer_position_ids is not None:
                additional_layer_inputs["position_ids"] = layer_position_ids
            for k, v in layer_input_kwargs[j].items():
                if isinstance(v, torch.Tensor):
                    additional_layer_inputs[k] = move_to_device(v, cur_layer_device)
                else:
                    additional_layer_inputs[k] = v
            layer_output = move_to_device(
                layer(layer_input, **additional_layer_inputs)[0], cur_layer_device if cache_examples_on_gpu else CPU)
            layer_outputs.append(layer_output)

        layers[i] = move_to_device(layer, CPU if force_layer_back_to_cpu else cur_layer_device)
        del layer
        del gptq
        del layer_inputs
        layer_inputs, layer_outputs = layer_outputs, []
        torch.cuda.empty_cache()

    pack_model(model=model,
               quantizers=quantizers,
               bits=quantize_config.bits,
               group_size=quantize_config.group_size,
               use_cuda_fp16=use_cuda_fp16,
               desc_act=quantize_config.desc_act,
               warmup_triton=autotune_warmup_after_quantized,
               force_layer_back_to_cpu=force_layer_back_to_cpu)
    if device_map:
        model = remove_hook_from_module(model, recurse=True)
        # model = simple_dispatch_model(model, device_map)
    model.config.use_cache = forward_pass_use_cache
    torch.cuda.empty_cache()


def save_quantized(model,
                   quantize_config,
                   save_dir: str,
                   use_safetensors: bool = False,
                   safetensors_metadata: Optional[Dict[str, str]] = None,
                   **kwargs):
    """alias of save_quantized"""
    logger.warning("you are using save_pretrained, which will re-direct to save_quantized.")
    os.makedirs(save_dir, exist_ok=True)

    model.to(CPU)

    model_base_name = quantize_config.model_file_base_name or f"gptq_model-{quantize_config.bits}bit-{quantize_config.group_size}g"
    if use_safetensors:
        model_save_name = model_base_name + ".safetensors"
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        if safetensors_metadata is None:
            safetensors_metadata = {}
        elif not isinstance(safetensors_metadata, dict):
            raise TypeError("safetensors_metadata must be a dictionary.")
        else:
            logger.debug(f"Received safetensors_metadata: {safetensors_metadata}")
            new_safetensors_metadata = {}
            converted_keys = False
            for key, value in safetensors_metadata.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    converted_keys = True
                    try:
                        new_key = str(key)
                        new_value = str(value)
                    except Exception as e:
                        raise TypeError(
                            f"safetensors_metadata: both keys and values must be strings and an error occured when trying to convert them: {e}"
                        )
                    if new_key in new_safetensors_metadata:
                        logger.warning(
                            f"After converting safetensors_metadata keys to strings, the key '{new_key}' is duplicated. Ensure that all your metadata keys are strings to avoid overwriting."
                        )
                    new_safetensors_metadata[new_key] = new_value
            safetensors_metadata = new_safetensors_metadata
            if converted_keys:
                logger.debug(
                    f"One or more safetensors_metadata keys or values had to be converted to str(). Final safetensors_metadata: {safetensors_metadata}"
                )

        # Format is required to enable Accelerate to load the metadata
        # otherwise it raises an OSError
        safetensors_metadata['format'] = "pt"

        # Store the quantization configuration as safetensors metadata
        # safetensors_metadata['auto_gptq_version'] = str(__version__)
        safetensors_metadata['gptq_bits'] = str(quantize_config.bits)
        safetensors_metadata['gptq_group_size'] = str(quantize_config.group_size)
        safetensors_metadata['gptq_desc_act'] = str(quantize_config.desc_act)
        safetensors_metadata['gptq_damp_percent'] = str(quantize_config.damp_percent)

        safe_save(state_dict, join(save_dir, model_save_name), safetensors_metadata)
    else:
        model_save_name = model_base_name + ".bin"
        torch.save(model.state_dict(), join(save_dir, model_save_name))

    model.config.save_pretrained(save_dir)
    quantize_config.save_pretrained(save_dir)
    quantize_config.model_name_or_path = save_dir
    quantize_config.model_file_base_name = model_base_name


def load_quantized(model_name_or_path: Optional[str],
                   device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
                   max_memory: Optional[dict] = None,
                   device: Optional[Union[str, int]] = None,
                   low_cpu_mem_usage: bool = False,
                   torch_dtype: Optional[torch.dtype] = None,
                   use_cuda_fp16: bool = True,
                   quantize_config: Optional[GPTQQuantizeConfig] = None,
                   model_basename: Optional[str] = None,
                   use_safetensors: bool = False,
                   trust_remote_code: bool = False,
                   **kwargs):
    """load quantized model from local disk"""

    # Parameters related to loading from Hugging Face Hub
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", "")
    commit_hash = kwargs.pop("_commit_hash", None)

    cached_file_kwargs = {
        "cache_dir": cache_dir,
        "force_download": force_download,
        "proxies": proxies,
        "resume_download": resume_download,
        "local_files_only": local_files_only,
        "use_auth_token": use_auth_token,
        "revision": revision,
        "subfolder": subfolder,
        "_raise_exceptions_for_missing_entries": False,
        "_commit_hash": commit_hash,
    }

    # == step1: prepare configs and file names == #
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, **cached_file_kwargs)

    if config.model_type not in SUPPORTED_MODELS:
        raise TypeError(f"{config.model_type} isn't supported yet.")

    model_block_names = models_map[config.model_type]()

    if quantize_config is None:
        quantize_config = GPTQQuantizeConfig.from_pretrained(model_name_or_path, **cached_file_kwargs, **kwargs)

    if model_basename is None:
        if quantize_config.model_file_base_name:
            model_basename = quantize_config.model_file_base_name
        else:
            model_basename = f"gptq_model-{quantize_config.bits}bit-{quantize_config.group_size}g"

    quantize_config.model_name_or_path = model_name_or_path
    quantize_config.model_file_base_name = model_basename

    extensions = []
    if use_safetensors:
        extensions.append(".safetensors")
    else:
        extensions += [".bin", ".pt"]

    model_name_or_path = str(model_name_or_path)
    is_local = isdir(model_name_or_path)

    resolved_archive_file = None
    if is_local:
        model_save_name = join(model_name_or_path, model_basename)
        for ext in extensions:
            if isfile(model_save_name + ext):
                resolved_archive_file = model_save_name + ext
                break
    else:    # remote
        for ext in extensions:
            resolved_archive_file = cached_file(model_name_or_path, model_basename + ext, **cached_file_kwargs)
            if resolved_archive_file is not None:
                break

    if resolved_archive_file is None:    # Could not find a model file to use
        raise FileNotFoundError(f"Could not find model in {model_name_or_path}")

    model_save_name = resolved_archive_file

    # == step2: convert model to gptq-model (replace Linear with QuantLinear) == #
    def skip(*args, **kwargs):
        pass

    if torch_dtype is None:
        torch_dtype = torch.float16

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    transformers.modeling_utils._init_weights = False

    init_contexts = [no_init_weights()]
    if low_cpu_mem_usage:
        init_contexts.append(accelerate.init_empty_weights(include_buffers=False))
    lm_head_name = "lm_head"

    with ContextManagers(init_contexts):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype)

        layers = find_layers(model)
        ignore_layers = [lm_head_name] + model_block_names.outside_layer_modules
        for name in list(layers.keys()):
            if any([name.startswith(ignore_layer) for ignore_layer in ignore_layers]):
                logger.info(f"{name} not been quantized, will be ignored when make_quant.")
                del layers[name]

        make_quant(
            model,
            layers,
            quantize_config.bits,
            quantize_config.group_size,
            use_cuda_fp16=use_cuda_fp16,
            desc_act=quantize_config.desc_act,
        )
        model.tie_weights()

    print("model name:", dir(model))
    # == step3: load checkpoint and dispatch == #
    if isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
        raise ValueError("If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                         "'sequential'.")
    if isinstance(device_map, dict):
        max_memory = None
    else:
        if device is None and not device_map and not max_memory:
            device_map = "auto"
        if device is not None:
            device = torch.device(device)
            if not max_memory and not device_map:
                device_map = {"": device.index if device.type == "cuda" else device.type}
        if not isinstance(device_map, dict) and device_map != "sequential":
            max_memory = accelerate.utils.get_balanced_memory(model=model,
                                                              max_memory=max_memory,
                                                              no_split_module_classes=[model_block_names.layer_type],
                                                              low_zero=(device_map == "balanced_low_0"))
    if not isinstance(device_map, dict):
        device_map = accelerate.infer_auto_device_map(model,
                                                      max_memory=max_memory,
                                                      no_split_module_classes=[model_block_names.layer_type])

    if low_cpu_mem_usage:
        make_sure_no_tensor_in_meta_device(model,
                                           quantize_config.desc_act,
                                           quantize_config.group_size,
                                           bits=quantize_config.bits)

    accelerate.utils.modeling.load_checkpoint_in_model(model,
                                                       checkpoint=model_save_name,
                                                       device_map=device_map,
                                                       offload_state_dict=True,
                                                       offload_buffers=True)

    # == step4: set seqlen == #
    model_config = model.config.to_dict()
    seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
    if any([k in model_config for k in seq_len_keys]):
        for key in seq_len_keys:
            if key in model_config:
                model.seqlen = model_config[key]
                break
    else:
        logger.warning("can't get model's sequence length from model config, will set to 4096.")
        model.seqlen = 4096
    return model
