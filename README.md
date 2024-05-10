# LoRA Instruction Tuning

[![Downloads][downloads-badge]][releases]

This repo contains some simple Python code (based upon HuggingFace) for instruction tuning common LLMs with LoRA/QLoRA.
The repo contains training code, as well as several different scripts for evaluating model generations.

[Setup](#setup) •
[Details](#details) •
[Usage](#usage) •
[Future Work](#future-work)

## Setup

Install necessary dependencies as follows

```
> conda create -n lora_tuning python=3.11 anaconda
> conda activate lora_tuning
> pip install -r requirements.txt
```

## Details

The repo support instruction tuning with [LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314), based upon the [PEFT](https://huggingface.co/docs/peft/en/index) (from HuggingFace) and [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index).
Currently, the example scripts instruction tune the [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/) model, though other models can be specified via the `--model_name_or_path` argument.

A breakdown of the main files within the resposity is as follows...
   > | File                 | Description                           |
   > | ---------------| ------------------------------------- |
   > | [train.py](train.py)       | Main training code      |
   > | [generate.py](generate.py) | Script for examining model output                 |
   > | [setup.py](setup.py)   | Functions for downloading and configuring models/tokenizers |
   > | [data.py](data.py)         | Code for configuring datasets |
   > | ./scripts | Scripts for training/evaluation |
   > | <td colspan=2><details><summary>See all scripts...</summary><ul><li colspan="2">**[train.sh](./scripts/train.sh)**: run instruction tuning (2x3090 GPUs)</li><li>**[generate.sh](./scripts/generate.sh)**: examine model outputs</li></ul></details>
   > | ./data | Supplemental data files |
   > | <td colspan=2><details><summary>See all files...</summary><ul><li colspan="2">**[vicuna_questions.json](./data/vicuna_questions.json)**: evaluation questions from [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)</ul></details>

The training process supports either the Alpaca or Assistant Chatbot dataset.
Evaluation is performed using the set of questions proposed for evaluating Vicuna (see [here](https://github.com/lm-sys/vicuna-blog-eval)).
However, model outputs can be observed over arbitrary datasets by leveraging the `generate.py` script.
The training process logs all metrics to wandb (assuming `--report_to wandb` is specified in the arguments), as well as generates model outputs for the vicuna evaluation set that are logged to wandb at the end of training. 

## Usage

Example scripts are located in the `./scripts` folder and can be run as follows:
```
> bash ./scripts/train.sh
> bash ./scripts/generate.sh
```
These scripts can also be customized by tweaking their arguments.
See [args.py](args.py) for a full list of arguments for the model, training, data, and generation.

## Future Work

This repository is very simplistic for now.
Future efforts will likely include:
* Expansion to more datasets (for training and evaluation)
* Implementing an [LLM-as-a-judge](https://arxiv.org/abs/2306.05685) style evaluation pipeline
* Adding evaluation on [MMLU](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu)