import os
from typing import Dict

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from enum import Enum

# padding token to add to tokenizer by default
DEFAULT_PAD_TOKEN = "[PAD]"

# create set of special tokens that will be encountered in the dataset
class SpecialTokens(str, Enum):
    begin_target = "<|begintarget|>"
    end_target = "<|endtarget|>"
    begin_context = "<|begincontext|>"
    end_context = "<|endcontext|>"
    system = "<|system|>"
    user = "<|user|>"
    begin_last_user_utterance = "<|beginlastuserutterance|>"
    end_last_user_utterance = "<|endlastuserutterance|>"
    begin_dsts = "<|begindsts|>"
    end_dsts = "<|enddsts|>"
    begin_dst = "<|begindst|>"
    end_dst = "<|enddst|>"
    begin_belief = "<|beginbelief|>"
    end_belief = "<|endbelief|>"
    begin_response = "<|beginresponse|>"
    end_response = "<|endresponse|>"
    begin_action = "<|beginaction|>"
    end_action = "<|endaction|>"
    begin_user_action = "<|beginuseraction|>"
    end_user_action = "<|enduseraction|>"
    sys_actions = "<|sysactions|>"
    begin_intent = "<|beginintent|>"
    end_intent = "<|endintent|>"
    begin_requested_slots = "<|beginrequestedslots|>"
    end_requested_slots = "<|endrequestedslots|>"
    pad_token = "<|pad|>"
    bos_token = "<|startoftext|>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]
     
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def get_tokenizer(args, model, add_tokens=False):
    # model_name = "mistralai/Mistral-7B-v0.1"
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print('='*50)
        print('Getting tokenizer...')
        print('='*50)
    if add_tokens:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=SpecialTokens.pad_token.value,
            bos_token=SpecialTokens.bos_token.value,
            eos_token=SpecialTokens.end_target.value,
            additional_special_tokens=SpecialTokens.list(),
        )
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    return model, tokenizer

def get_pretrained_model(args):
    # model_name = "mistralai/Mistral-7B-v0.1"
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print('='*50)
        print('Getting model...')
        print('='*50)
    
    # figure out model setup
    n_gpus = torch.cuda.device_count()
    max_memory = {i: args.max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    if args.training_method == 'qlora':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.training_method == 'lora':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    else:
        raise ValueError(f'Unsupported finetuning method: {args.training_method}')

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        # use_flash_attention_2=True, # leading to an error
    )
    return model

def prepare_peft_model(args, model, tokenizer):
    model = prepare_model_for_kbit_training(model)

    # lora config
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print('='*50)
        print('Adding LoRA modules...')
        print('='*50)
    config = LoraConfig(
        r=args.lora_r, # 32
        lora_alpha=args.lora_alpha, # 64
        lora_dropout=args.lora_dropout, # 0.0
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, config)
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print('='*50)
        print('\nLoRA Model Info:')
        model.print_trainable_parameters()
        print('='*50)
    return model, tokenizer
