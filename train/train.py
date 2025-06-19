import numpy as np
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers
from transformers import Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, get_scheduler
from peft import prepare_model_for_int8_training
import torch
from peft.tuners.lora import LoraLayer
from torch.optim import AdamW

import sys
import os
sys.path.append(os.path.abspath('../'))

from llama_flash_attn_replace import *
from utils.args import *
from model.Decoder_CausalLLM import *
from model.VAE_CausalLLM import *
from utils.data_collator import *

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")
    model_type: Optional[str] = field(default="llama")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str = field(default='./hf_cache')
    output_dir: str = field(default='./model_output')
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
        default=4,
        metadata={"help": "Bit precision for model weights (e.g., 16, 8, or 4)."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 precision."}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Use float16 precision."}
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )

def train(params,remaining_args):
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args= parser.parse_args_into_dataclasses(remaining_args)


    # replace_llama_attn(inference=False)

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        z_latent_size=params.z_latent_size
    )
    config.z_latent_size=params.z_latent_size
    config.traj_length=params.traj_length

    model_decoder=DecoderLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        device_map={"": training_args.device}
    )
    # 如果 linear_projection 是自定义层，确保它不被量化
    if hasattr(model_decoder, 'linear_projection'):
        # 重新创建为 float32 或 bfloat16
        model_decoder.linear_projection = torch.nn.Linear(
            model_decoder.linear_projection.in_features,
            model_decoder.linear_projection.out_features,
            dtype=torch.float32
        ).to(model_decoder.device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    model_decoder.get_tokenizer(tokenizer)

    for name, param in model_decoder.named_parameters():
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

    model_decoder = get_peft_model(model_decoder, config)

    for name, module in model_decoder.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module.to(torch.bfloat16)
        if 'norm' in name:
            module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                   module.to(torch.bfloat16)

    model_decoder.config.use_cache = False  # required for gradient checkpointing
    model_decoder.enable_input_require_grads()  # required for gradient checkpointing
    model_decoder.gradient_checkpointing_enable()  # enable gradient checkpointing

    if training_args.bits in [4, 8]:
        print('training_args.bits in [4, 8]')
        model_decoder.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model_decoder = prepare_model_for_int8_training(model_decoder, use_gradient_checkpointing=training_args.gradient_checkpointing)

    device = next(model_decoder.parameters()).device

    params.device= device

    [p.requires_grad_() for n, p in model_decoder.named_parameters() if
     any([k in n for k in training_args.trainable_params.split(",")])]

    tuned_params = []
    for name, param in model_decoder.named_parameters():
        if param.requires_grad:
            tuned_params.append(name)
    print(tuned_params)

    model=VAE_CausalLLM(params,model_decoder)
    model.to(device)
    model.train()


    # for name, module in model.named_modules():
    #     if "bitsandbytes" in str(type(module)):
    #         print(f"{name} → {type(module)}")
    #
    # exit()

    optimizer = AdamW(model.parameters(), lr=params.learning_rate,weight_decay=params.weight_decay)
    scheduler = get_scheduler(
        name="linear",  # 线性衰减调度器
        optimizer=optimizer,
        num_warmup_steps=5000,
        num_training_steps=20000,
    )

    uid_traj=pickle.load(open(params.path_traj, 'rb'))
    uid_mask_day=pickle.load(open(params.uid_mask_day, 'rb'))
    user_attr=pickle.load(open(params.path_attr, 'rb'))

    loss_sum=0
    count=0

    for e in range(params.epoch_num):
        batch_all = get_batch_home_info(uid_traj, uid_mask_day, user_attr)
        batch_train = batch_all['train'][:3000]

        for trajs_tensor, home_ids_tensor in collate_batch_data(uid_traj, batch_train, params):
            # trajs_tensor=trajs_tensor.to(params.device)
            # home_ids_tensor=home_ids_tensor.to(params.device)

            trajs_tensor = trajs_tensor.to(device)
            home_ids_tensor = home_ids_tensor.to(device)

            output,kl_loss=model(trajs_tensor, home_ids_tensor)
            llm_loss=output.loss
            loss=kl_loss+llm_loss

            loss.backward()
            loss_sum+=loss.item()
            count+=1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            ##
            logits=output.logits
            pred=torch.argmax(logits, dim=-1)
            ##
            if (count+1)%2==0:
                print(f'Loss: all={loss_sum/count:.3f}')
                loss_sum=0
                count=0

    ## save model
    torch.save(model.state_dict(), f"{params.path_save}model.pth")
    model_decoder.config.save_pretrained(params.path_save)
    model_decoder.save_pretrained(params.path_save)
    tokenizer.save_pretrained(params.path_save)

if __name__ == '__main__':
    params,remaining_args = param_settings('SH')
    train(params,remaining_args)



