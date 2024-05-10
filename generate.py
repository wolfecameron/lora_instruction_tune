import argparse

import transformers
from peft import PeftModel

from data import load_vicuna_eval_dataset, load_alpaca_dataset
from args import ModelArguments, TrainingArguments, GenerationArguments, DataArguments
from setup import (
    get_tokenizer,
    get_pretrained_model,
)

hfparser = transformers.HfArgumentParser((
    ModelArguments, TrainingArguments, DataArguments, GenerationArguments,
))
model_args, training_args, data_args, generation_args, extra_args = \
    hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
args = argparse.Namespace(
    **vars(model_args), **vars(training_args), **vars(data_args), **vars(generation_args),
)

# get PEFT (or pretrained) model for evaluation
model = get_pretrained_model(args)
model, tokenizer = get_tokenizer(args, model, add_tokens=(args.dataset == 'assistant-chat'))
if not args.use_pretrained_model:
    model = PeftModel.from_pretrained(
        model,
        args.checkpoint_path,
    )
model.eval()

# get data to check
# training_dataset, test_dataset = load_assistant_chat_dataset(tokenizer, process_data=False)
if args.dataset == 'vicuna':
    dataset = load_vicuna_eval_dataset()
elif args.dataset == 'alpaca':
    _, dataset = load_alpaca_dataset(args)
else:
    raise ValueError(f'Unknown dataset {args.dataset}')

for ex in dataset:
    batch = tokenizer(
        f"{tokenizer.bos_token}{ex['input']}",
        max_length=args.source_max_len,
        truncation=True,
        add_special_tokens=False,
        return_tensors='pt',
    )
    target = None
    batch = {k: v.cuda() for k, v in batch.items()}
    result = model.generate(
        **batch,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **vars(generation_args),
    )
    result = tokenizer.decode(result[0], skip_special_tokens=False).split('### Response:')[1].strip()

    # display the results
    print('\n\n')
    print('='*50)
    print('Prompt')
    print('='*50)
    print(f"\n{tokenizer.bos_token}{ex['input']}\n")
    if 'output' in ex.keys():
        print('='*50)
        print('Target')
        print('='*50)
        print(f"\n{ex['output']}{tokenizer.eos_token}\n")
    print('='*50)
    print('Model Output')
    print('='*50)
    print(f"\n{result}\n")
    print('='*50)
    input()
    