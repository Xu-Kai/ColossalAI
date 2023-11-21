import argparse
import time

import torch
import torch.distributed as dist

import colossalai
import colossalai.utils.device as device_utils
from colossalai.inference import InferenceEngine
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn
from colossalai.utils.device import get_current_device

GIGABYTE = 1024**3
MEGABYTE = 1024 * 1024


def data_gen(batch_size: int = 4, seq_len: int = 512):
    input_ids = torch.randint(10, 30000, (batch_size, seq_len), device=get_current_device())
    attention_mask = torch.ones_like(input_ids)
    data = dict(input_ids=input_ids, attention_mask=attention_mask)
    return data


def print_details_info(outputs, model_config, args, whole_end2end):
    msg: str = ""

    if dist.get_rank() == 0:
        msg += "-------Perf Summary-------\n"
        if args.verbose:
            timestamps = outputs[1]
            prefill = []
            encoder = []
            end2end = []
            for timestamp in timestamps:
                prefill.append(timestamp[1] - timestamp[0])
                encoder.append(
                    sum(timestamp[i + 1] - timestamp[i] for i in range(1, len(timestamp) - 1)) / (len(timestamp) - 2)
                )
                end2end.append(timestamp[-1] - timestamp[0])

            mb_avg_end2end = sum(end2end) / len(end2end)
            mb_avg_latency = mb_avg_end2end / (args.output_len * args.mb_size)

            msg += f"Average prefill time: {sum(prefill) / len(prefill) * 1000:.2f} ms\n"
            msg += f"Average encode time: {sum(encoder) / len(encoder) * 1000:.2f} ms\n"
            msg += f"Average micro batch end2end time: {mb_avg_end2end * 1000:.2f} ms\n"
            msg += f"Average micro batch per token latency: {mb_avg_latency * 1000:.2f} ms\n"

        whole_avg_latency = whole_end2end / (args.output_len * args.batch_size)
        num_layers = getattr(model_config, "num_layers", model_config.num_hidden_layers)
        num_parameters = num_layers * model_config.hidden_size * model_config.hidden_size * 12 / args.pp_size
        if args.dtype in ["fp16", "bf16"]:
            num_bytes = 2
        else:
            num_bytes = 4

        msg += f"Whole batch end2end time: {whole_end2end * 1000:.2f} ms\n"
        msg += f"Whole batch per token latency: {whole_avg_latency * 1000:.2f} ms\n"
        msg += f"Throughput: {args.output_len * args.batch_size / whole_end2end:.2f} tokens/s\n"
        msg += f"Flops: {num_parameters * num_bytes / whole_avg_latency / 1e12:.2f} TFLOPS\n"

    if torch.cuda.is_available():
        msg += f"-------Memory Summary Device:{device_utils.current_device()}-------\n"
        msg += f"Max memory allocated: {device_utils.max_memory_allocated() / GIGABYTE:.2f} GB\n"
        msg += f"Max memory reserved: {device_utils.max_memory_reserved() / GIGABYTE:.2f} GB\n"

    print(msg)


def benchmark_gptq_inference(args):
    from auto_gptq import AutoGPTQForCausalLM

    # load quantized model to the first GPU
    model = AutoGPTQForCausalLM.from_quantized(
        args.gptq_model, device=torch.cuda.current_device(), inject_fused_attention=False
    )

    if dist.get_rank() == 0:
        print("Model loaded")
    engine = InferenceEngine(
        pp_size=args.pp_size,
        tp_size=args.tp_size,
        dtype=args.dtype,
        micro_batch_size=args.mb_size,
        model=model,
        verbose=args.verbose,
        max_batch_size=args.batch_size,
        max_input_len=args.seq_len,
        max_output_len=args.output_len,
        quant="gptq",
    )
    data = data_gen(args.batch_size, args.seq_len)

    N_WARMUP_STEPS = 2

    for _ in range(N_WARMUP_STEPS):
        engine.generate(data)

    torch.cuda.synchronize()
    whole_end2end = time.time()
    outputs = engine.generate(data)
    torch.cuda.synchronize()
    whole_end2end = time.time() - whole_end2end

    print_details_info(outputs, model.config, args, whole_end2end)


def benchmark_auto_gptq_inference(args):
    from auto_gptq import AutoGPTQForCausalLM

    # load quantized model to the first GPU
    model = AutoGPTQForCausalLM.from_quantized(args.gptq_model, device=torch.cuda.current_device())

    if dist.get_rank() == 0:
        print("Model loaded")

    data = data_gen(args.batch_size, args.seq_len)
    generate_kwargs = dict(max_new_tokens=args.output_len, do_sample=False, use_cache=True)
    N_WARMUP_STEPS = 2

    for _ in range(N_WARMUP_STEPS):
        model.generate(**data, **generate_kwargs)

    torch.cuda.synchronize()
    whole_end2end = time.time()
    outputs = model.generate(**data, **generate_kwargs)
    torch.cuda.synchronize()
    whole_end2end = time.time() - whole_end2end

    print_details_info(outputs, model.config, args, whole_end2end)


def hybrid_inference(rank, world_size, port, args):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    if args.bench_type == "cai-gptq":
        benchmark_gptq_inference(args)
    elif args.bench_type == "auto-gptq":
        benchmark_auto_gptq_inference(args)


@rerun_if_address_is_in_use()
@clear_cache_before_run()
def benchmark(args):
    spawn(hybrid_inference, nprocs=args.tp_size * args.pp_size, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gptq_model",
        help="the path of gptq model",
        type=str,
    )
    parser.add_argument(
        "--bench_type",
        help="the type of benchmark type: 'cai-gptq', 'auto-gptq'",
        type=str,
        choices=["cai-gptq", "auto-gptq"],
        default="cai-gptq",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("-s", "--seq_len", type=int, default=8, help="sequence length")
    parser.add_argument("--mb_size", type=int, default=1, help="micro_batch_size")
    parser.add_argument("--pp_size", type=int, default=1, help="pipeline size")
    parser.add_argument("--tp_size", type=int, default=1, help="pipeline size")
    parser.add_argument("--output_len", type=int, default=128, help="Output length")
    parser.add_argument("--dtype", type=str, default="fp16", help="data type")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    args = parser.parse_args()
    benchmark(args)
