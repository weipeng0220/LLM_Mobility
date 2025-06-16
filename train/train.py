import numpy as np
from utils.args import *
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers
from transformers import Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
import torch
from peft.tuners.lora import LoraLayer

from llama_flash_attn_replace import *
from model.Decoder_CausalLLM import *

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    model_type: Optional[str] = field(default="llama")

@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default='./hf_cache')
    model_max_length: int = field(default=512,
                                  metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
                                  )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    bits: int = field(
        default=16,
        metadata={"help": "Bit precision for model weights (e.g., 16, 8, or 4)."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 precision."}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Use float16 precision."}
    )

def train(params):
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args= parser.parse_args_into_dataclasses()

    replace_llama_attn(inference=False)

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        z_latent_size=params.z_latent_size
    )

    model=DecoderLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    model.get_tokenizer(tokenizer)

    for name, param in model.named_parameters():
        if 'linear_projection' in name:
            param.requires_grad = True
            print(f"Trainable: {name}")
        else:
            param.requires_grad = False

    config = LoraConfig(
        r=params.lora_r,
        lora_alpha=params.lora_alpha,
        target_modules=params.lora_target_modules,
        lora_dropout=params.lora_dropout,
        bias=params.lora_bias,
        task_type=params.task_type
    )

    model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    model.config.use_cache = False  # required for gradient checkpointing
    model.enable_input_require_grads()  # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    [p.requires_grad_() for n, p in model.named_parameters() if
     any([k in n for k in training_args.trainable_params.split(",")])]

    tuned_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            tuned_params.append(name)
    print(tuned_params)










if __name__ == '__main__':
    params = param_settings('SH')
    train(params)
