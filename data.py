import os
from dataclasses import dataclass
from typing import Sequence, Dict
import copy

import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from datasets import load_dataset

# default index ignored by CE loss in PyTorch
IGNORE_INDEX = -100

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]

        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            input_ids.append(torch.tensor(tokenized_source + tokenized_target))
            if not self.train_on_source:
                labels.append(
                    torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                )
            else:
                labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
            'labels': labels,
        }
        return data_dict

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def load_alpaca_dataset(args):
    # ['instruction', 'input', 'output', 'text']
    dataset = load_dataset("tatsu-lab/alpaca")

    # convert all data into proper prompt format
    dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
    
    # create an evaluation set
    dataset = dataset['train'].train_test_split(test_size=0.05, shuffle=True, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    if (
        args.max_train_samples is not None
        and len(train_dataset) > args.max_train_samples
    ):
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if (
        args.max_eval_samples is not None
        and len(eval_dataset) > args.max_eval_samples
    ):
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    return train_dataset, eval_dataset

def load_assistant_chat_dataset(args):
    dataset = load_dataset("smangrul/assistant_chatbot_dataset")
    dataset = dataset["train"].train_test_split(0.2)
    text_column = "context"
    label_column = "target"

    # prep the dataset
    def preprocess_function(example):
        inputs = example[text_column]
        target = example[label_column]
        return {
            'input': inputs,
            'output' : target,
            'text': inputs + target,
        }

    dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset["train"].column_names,
        desc="Running tokenizer on dataset",
    )
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    if (
        args.max_train_samples is not None
        and len(train_dataset) > args.max_train_samples
    ):
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if (
        args.max_eval_samples is not None
        and len(eval_dataset) > args.max_eval_samples
    ):
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    return train_dataset, eval_dataset

def extract_vicuna_dataset(example):
    # vicuna examples only have instruction at key "text" (no input)
    prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    example = {'instruction': example['text']}
    return {'input': prompt_format.format(**example)}

def load_vicuna_eval_dataset():
    dataset = load_dataset("json", data_files={
        'test': 'data/vicuna_questions.json'
    })['test']
    
    # keep text (raw text) and input (formatted input) columns
    dataset = dataset.map(
        extract_vicuna_dataset,
        remove_columns=['question_id', 'category'],
    )
    return dataset

if __name__=='__main__':
    train_ds, test_ds = load_assistant_chat_dataset()