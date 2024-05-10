from dataclasses import dataclass, field
from typing import Optional

import transformers

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-v0.1")
    checkpoint_path: Optional[str] = field(default='./result')
    use_pretrained_model: bool = field(default=False)

@dataclass
class DataArguments:
    dataset: str = field(default='alpaca')
    source_max_len: int = field(default=512)
    target_max_len: int = field(default=512)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    output_dir: str = field(default='./results',)
    run_name: Optional[str] = field(default='qlora-alpaca-00')
    training_method: Optional[str] = field(default='qlora')
    lora_r: int = field(default=64)
    lora_alpha: float = field(default=16)
    lora_dropout: float = field(default=0.0)
    report_to: str = field(default='none')
    train_on_source: Optional[bool] = field(default=False)
    num_train_epochs: Optional[float] = field(default=None)
    max_steps: Optional[int] = field(default=None)
    warmup_ratio: float = field(default=0.03)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    weight_decay: float = field(default=0.0)
    learning_rate: float = field(default=0.0002)
    max_grad_norm: float = field(default=0.3)
    lr_scheduler_type: str = field(default='constant')
    optim: str = field(default='paged_adamw_32bit')
    save_strategy: str = field(default='steps')
    save_steps: int = field(default=250)
    save_total_limit: int = field(default=40)
    logging_steps: int = field(default=10)
    eval_steps: int = field(default=250)
    evaluation_strategy: str = field(default='steps')
    max_memory: str = field(default='24000MB')
    seed: int = field(default=42)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    do_predict: bool = field(default=False)

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)
