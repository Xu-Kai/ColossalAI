import argparse
import os
import random
import copy
import math
from tqdm import tqdm
import numpy as np

import torch
import torch.distributed as dist
import transformers

from transformers import AutoTokenizer, RobertaTokenizer, AutoModelForCausalLM, GPTNeoXTokenizerFast,AutoModel,T5ForConditionalGeneration,T5Tokenizer,AutoModelForSeq2SeqLM,GPTNeoXForCausalLM
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer


from utils import jload, jdump, is_rank_0
import logging as logger
import deepspeed

random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)

PROMPT_DICT = {
    "prompt_input":
        ("Below is an instruction that describes a task, paired with an input that provides further context. "
         "Write a response that appropriately completes the request.\n\n"
         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"),
    "prompt_no_input": ("Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"),
}


def generate(args):
    # torch.cuda.set_per_process_memory_fraction(0.4)

    deepspeed.init_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    actor = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    infer_dtype = "float16"
    kwargs = dict(replace_with_kernel_inject=True, max_out_tokens=1536)
    model = deepspeed.init_inference(
        actor,
        mp_size=1,
        base_dir=args.model_path,
        dtype=getattr(torch, infer_dtype),
        **kwargs,
    )
    actor = model

    
    if is_rank_0():
        logger.info(f"{args.model_name}")


    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    elif args.model == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                                  padding_side="right",
                                                  use_fast=False,
                                                  )
        tokenizer.eos_token = '<\s>'
    else:
        if args.model_name.startswith("dolly"):
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
        elif args.model_name.startswith('opt'):
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b", padding_side="left")
        elif args.model_name.startswith("chatglm"):
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        elif args.model_name.startswith("mpt"):
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
        elif args.model_name.startswith("fastchat-t5"):
            tokenizer = T5Tokenizer.from_pretrained(args.model_path, use_fast=False, padding_side="left")
        elif args.model_name.startswith("RedPajama"):
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
        elif args.model_name.startswith("pythia"):
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
        elif args.model_name.startswith("phoenix"):
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
        elif args.model_name.startswith("Cerebras"):
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
        elif args.model_name.startswith("stablelm"):
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
        elif args.model_name.startswith("h2ogpt"):
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
        
        # if tokenizer.eos_token is None:
        #     tokenizer.eos_token = '<|endoftext|>'
        if tokenizer.pad_token is None:    
            tokenizer.pad_token = tokenizer.eos_token
        # raise ValueError(f'Unsupported model "{args.model}"')
    
    questions = []
    if args.max_datasets_size is not None:
        questions = random.sample(jload(args.dataset), args.max_datasets_size)
        if is_rank_0():
            logger.info(
                f"Limiting dataset to {args.max_datasets_size} examples.")
        questions = questions[rank:args.max_datasets_size:world_size]
    else:
        # questions=jload(args.dataset)[rank*75:rank*75+75]
        questions=jload(args.dataset)

    answers = copy.deepcopy(questions)

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources = [
        prompt_input.format_map(example) if example.get(
            "input", "") != "" else prompt_no_input.format_map(example)
        for example in questions
    ]

    if is_rank_0():
        logger.info("Tokenizing inputs... This may take some time...")

    input_ids_list = []

    for string in sources:
        input_ids = tokenizer.encode(string, return_tensors='pt').squeeze(0)
        input_ids_list.append(input_ids)

    bar = tqdm(range(math.ceil(len(input_ids_list)/args.batch_size)),
               desc=f'steps', disable=not is_rank_0())

    # if args.model_name.startswith("dolly"):
    #     generate_text = InstructionTextGenerationPipeline(model=actor, tokenizer=tokenizer)
    # actor.eval()
    prompt_input = "### Question:\n{instruction}\n{input}\n### Answer:\n"
    prompt_no_input = "### Question:\n{instruction}\n### Answer:\n"
    with torch.no_grad():
        for i in range(0, len(questions), args.batch_size):
            # batch = input_ids_list[i:i+args.batch_size]
            # batch = [i.flip(dims=[0]) for i in batch]
            # batch = torch.nn.utils.rnn.pad_sequence(batch,
            #                                         batch_first=True,
            #                                         padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0).to(torch.cuda.current_device())
            # batch = batch.flip(dims=[1])
            # attention_mask = batch.ne(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

            # outputs = actor.generate(batch, attention_mask=attention_mask,
            #                                max_length=args.max_length,
            #                                do_sample=True,
            #                                top_k=50,
            #                                top_p=0.95,
            #                                num_return_sequences=1)
            # batch=questions[i:i+args.batch_size]
            # encoded_batch=tokenizer(batch, return_tensors='pt', padding=True)
            # inst=questions[i]['instruction'] if questions[i]['input']=="" else questions[i]['instruction']+" "+questions[i]['input']
            
            batch=questions[i:i+args.batch_size]
            batch_input=[prompt_input.format_map(example) if example.get(
                            "input", "") != "" else prompt_no_input.format_map(example) for example in batch]
            encoded_batch=tokenizer(batch_input, return_tensors='pt', padding=True).to(torch.cuda.current_device())
            
            if args.model_name.startswith("dolly"):
                available_tokens = 2048 - encoded_batch['input_ids'].size(1)
                outputs = actor.generate(**encoded_batch, 
                                         max_new_tokens=512 if available_tokens>512 else available_tokens,
                                         do_sample=True,
                                         top_p=0.92,
                                         top_k=0)
                res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            elif args.model_name.startswith("opt"):
                available_tokens = 2048 - encoded_batch['input_ids'].size(1)
                outputs = actor.generate(**encoded_batch, 
                                         max_new_tokens=512 if available_tokens>512 else available_tokens,
                                         do_sample=True,
                                         temperature=0.7)
                res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            elif args.model_name.startswith("chatglm"):
                inst = batch[0]["instruction"] if batch[0]["input"]=="" else batch[0]["instruction"]+batch[0]["input"]
                batch_input = copy.deepcopy(inst)
                encoded_batch=tokenizer(batch_input, return_tensors='pt', padding=True).to(torch.cuda.current_device())
                res = actor.chat(tokenizer, inst, history=[], max_length=1024+encoded_batch['input_ids'].size(1))
            elif args.model_name.startswith("mpt"):
                available_tokens = 2048 - encoded_batch['input_ids'].size(1)
                outputs = actor.generate(**encoded_batch, 
                                         max_new_tokens=512 if available_tokens>512 else available_tokens)
                res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            elif args.model_name.startswith("RedPajama"):
                available_tokens = 2048 - encoded_batch['input_ids'].size(1)
                outputs = actor.generate(**encoded_batch,
                                         max_new_tokens=512 if available_tokens>512 else available_tokens,
                                         do_sample=True,
                                         temperature=0.7, 
                                         top_p=0.7, 
                                         top_k=50)
                res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            elif args.model_name.startswith("fastchat-t5"):
                available_tokens = 2048 - encoded_batch['input_ids'].size(1)
                outputs = actor.generate(**encoded_batch,
                                         max_new_tokens=512 if available_tokens>512 else available_tokens,
                                         do_sample=True,
                                         temperature=0.7)
                res = tokenizer.batch_decode(outputs, skip_special_tokens=True, spaces_between_special_tokens=False,)
            elif args.model_name.startswith("pythia"):
                available_tokens = 2048 - encoded_batch['input_ids'].size(1)
                outputs = actor.generate(**encoded_batch,
                                         max_new_tokens=512 if available_tokens>512 else available_tokens)
                res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            elif args.model_name.startswith("phoenix"):
                available_tokens = 2048 - encoded_batch['input_ids'].size(1)
                outputs = actor.generate(**encoded_batch,
                                         max_new_tokens=512 if available_tokens>512 else available_tokens)
                res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            elif args.model_name.startswith("Cerebras"):
                available_tokens = 2048 - encoded_batch['input_ids'].size(1)
                outputs = actor.generate(**encoded_batch, 
                                         max_new_tokens=512 if available_tokens>512 else available_tokens,
                                         num_beams=5, 
                                         early_stopping=True,
                                         no_repeat_ngram_size=2)
                res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            elif args.model_name.startswith("stablelm"):
                available_tokens = 2048 - encoded_batch['input_ids'].size(1)
                outputs = actor.generate(**encoded_batch,
                                         max_new_tokens=512 if available_tokens>512 else available_tokens,
                                         temperature=0.7,
                                         do_sample=True)
                res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            elif args.model_name.startswith("h2ogpt"):
                available_tokens = 2048 - encoded_batch['input_ids'].size(1)
                outputs = actor.generate(**encoded_batch,
                                         max_new_tokens=512 if available_tokens>512 else available_tokens)
                res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
            for j in range(len(batch)):
                if len(res[j].split("### Answer:"))>=2:
                    answers[i + j]['output'] = res[j].split("### Answer:")[1].strip()
                else:
                    answers[i + j]['output'] = res[j]
                
            bar.update()
    
    jdump(answers, os.path.join(args.answer_path,
          f'{args.model_name}_answers_rank{rank}.json'))

    if is_rank_0():
        logger.info(f'Peak CUDA mem: {torch.cuda.max_memory_allocated()/1024**3:.3f} GB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini',
                                 'colossalai_zero2', 'colossalai_zero2_cpu'],
                        default='naive')
    parser.add_argument('--model', default='gpt2',
                        choices=['gpt2', 'bloom', 'opt', 'roberta', 'llama', 'a'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_datasets_size', type=int, default=None)
    parser.add_argument('--answer_path', type=str, default="answer")
    parser.add_argument('--max_length', type=int, default=1024)
    args = parser.parse_args()
    generate(args)