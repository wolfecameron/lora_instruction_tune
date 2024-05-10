import os
os.environ["WANDB_PROJECT"] = "lora-finetuning"
import argparse
import json

from tqdm import tqdm
import wandb
import torch
import transformers
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import load_dataset

from args import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    GenerationArguments,
)
from setup import (
    get_tokenizer,
    get_pretrained_model,
    prepare_peft_model,
)
from data import (
    load_assistant_chat_dataset,
    load_alpaca_dataset,
    load_vicuna_eval_dataset,
    DataCollatorForCausalLM,
)

# parse all the arguments
hfparser = transformers.HfArgumentParser((
    ModelArguments, DataArguments, TrainingArguments, GenerationArguments,
))
model_args, data_args, training_args, generation_args, extra_args = \
    hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
args = argparse.Namespace(
    **vars(model_args), **vars(data_args), **vars(training_args)
)

# get model and prepare for PEFT training
model = get_pretrained_model(args)
model, tokenizer = get_tokenizer(args, model, add_tokens=(args.dataset == 'assistant-chat'))
model, tokenizer = prepare_peft_model(args, model, tokenizer)

# get the data
sft_collator = DataCollatorForCausalLM(
    tokenizer=tokenizer,
    source_max_len=args.source_max_len,
    target_max_len=args.target_max_len,
    train_on_source=args.train_on_source,
)

if args.dataset == 'assistant-chat':
    train_dataset, test_dataset = load_assistant_chat_dataset(args)
elif args.dataset == 'alpaca':
    train_dataset, test_dataset = load_alpaca_dataset(args)
else:
    raise ValueError(f'Invalid dataset: {args.dataset}')

training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(args.output_dir, args.run_name),
    report_to=args.report_to,
    run_name=args.run_name,
    do_train=args.do_train,
    do_eval=args.do_eval,
    do_predict=args.do_predict,
    num_train_epochs=args.num_train_epochs,
    warmup_ratio=args.warmup_ratio,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    weight_decay=args.weight_decay,
    learning_rate=args.learning_rate,
    max_grad_norm=args.max_grad_norm,
    lr_scheduler_type=args.lr_scheduler_type,
    optim=args.optim,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=args.logging_steps,
    eval_steps=args.eval_steps,
    evaluation_strategy=args.evaluation_strategy,
    label_names=["labels"],
    save_steps=args.save_steps,
    save_strategy=args.save_strategy,
    save_total_limit=args.save_total_limit,
    ddp_find_unused_parameters=False,
    group_by_length=False,
    seed=args.seed,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=sft_collator,
)

if args.do_train:
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print('='*50)
        print('Running training...')
        print('='*50)
    model.config.use_cache = False # need to reenable for inference
    trainer.train()

    # save final model at end of training
    trainer.args.output_dir = os.path.join(trainer.args.output_dir, 'final/')
    trainer.save_model()
    
if args.do_predict:
    # run prediction on a single GPU
    if os.environ.get('LOCAL_RANK', '0') == '0':
        print('='*50)
        print('Running prediction...')
        print('='*50)
        predict_dataset = load_vicuna_eval_dataset()
        all_results = []
        for ex in tqdm(predict_dataset, total=len(predict_dataset)):
            # text -- just the pure text
            # input -- formatted text
            input_tokens = trainer.tokenizer(
                f"{trainer.tokenizer.bos_token}{ex['input']}",
                max_length=args.source_max_len,
                truncation=True,
                add_special_tokens=False,
                return_tensors='pt',
            )
            input_tokens = {k: v.cuda() for k, v in input_tokens.items()}
            result = trainer.model.generate(
                input_ids=input_tokens['input_ids'],
                attention_mask=input_tokens['attention_mask'],
                pad_token_id=trainer.tokenizer.eos_token_id,
                **vars(generation_args),
            )
            str_result = trainer.tokenizer.decode(result[0], skip_special_tokens=True)
            all_results.append([ex['text'], ex['input'], str_result])
        
        if 'wandb' in args.report_to:
            pred_columns = [
                'text',
                'input',
                'prediction',
            ]
            pred_table = wandb.Table(columns=pred_columns, data=all_results)
            wandb.log({'test/predictions': pred_table})
        else:
            with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
                for res in all_results:
                    example = {
                        'text': res[0],
                        'input': res[1],
                        'prediction': res[2],
                    }
                    fout.write(json.dumps(example) + '\n')

