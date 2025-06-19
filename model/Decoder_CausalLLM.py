from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Conversation import *

## Special Tokens
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_HIDDEN_TOKEN = "<HIDDEN>"
DEFAULT_PRE_TOKEN = "<PRE>"
DEFAULT_START_TOKEN = "<START>"
DEFAULT_END_TOKEN = "<END>"


class DecoderLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super(DecoderLlamaForCausalLM, self).__init__(config)
        self.config = config
        self.linear_projection = nn.Linear(config.z_latent_size, config.hidden_size)
        self.prompt = prompt()

    def get_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.initialize_tokenizer_and_embedding_resize()

    def forward(self, Z_hidden, home_loc_ids, traj_target=None, **kwargs):  # [batch_size,z_hidden_size], [batch_size, 1], [batch_size, 24]
        z_projection = self.linear_projection(Z_hidden)  # [batch_size, hidden_size]
        input_batch = []

        if traj_target is not None:
            for batch_idx in range(Z_hidden.shape[0]):
                home_loc_id = home_loc_ids[batch_idx]
                traj_seq = traj_target[batch_idx]
                inputs = self.prompt.get_prompt_train(home_loc_id, traj_seq, self.config.traj_length)
                input_batch.append(inputs)
        else:
            for batch_idx in range(Z_hidden.shape[0]):
                home_loc_id = home_loc_ids[batch_idx]
                inputs = self.prompt.get_prompt_eval(home_loc_id, self.config.traj_length)
                input_batch.append(inputs)

        input_tokenized = self.tokenizer(
            input_batch,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        )
        input_ids = input_tokenized['input_ids']
        input_ids=input_ids.to(next(self.parameters()).device)
        if traj_target is not None:
            labels = input_ids.clone()
        else:
            labels = None

        inputs_embeds = self.model.embed_tokens(input_ids)
        new_input_embeds = []

        for i in range(Z_hidden.size(0)):
            if traj_target is not None:
                pre_idx_list = torch.where(input_ids[i] == self.config.pre_token_id)[0]
                if len(pre_idx_list) > 0:
                    labels[i, :pre_idx_list[0] + 2] = IGNORE_INDEX

            start_idx_list = torch.where(input_ids[i] == self.config.start_token_id)[0]
            if len(start_idx_list) == 0:
                continue
            start_idx = start_idx_list[0].item()

            cur_new_input_embeds = torch.cat((inputs_embeds[i][:start_idx + 1],
                                              z_projection[i].unsqueeze(0),
                                              inputs_embeds[i][start_idx + 2:]), dim=0)
            new_input_embeds.append(cur_new_input_embeds)
        inputs_embeds = torch.stack(new_input_embeds, dim=0)

        attention_mask = input_tokenized['attention_mask']  # (labels != self.tokenizer.pad_token_id).long()
        # 从kwargs中移除可能重复的参数
        forward_kwargs = kwargs.copy()
        if 'attention_mask' in forward_kwargs:
            del forward_kwargs['attention_mask']
        if 'input_ids' in forward_kwargs:
            del forward_kwargs['input_ids']
        if 'inputs_embeds' in forward_kwargs:
            del forward_kwargs['inputs_embeds']
        if 'labels' in forward_kwargs:
            del forward_kwargs['labels']
        # return super(DecoderLlamaForCausalLM, self).forward(attention_mask=attention_mask,
        #                                                     inputs_embeds=inputs_embeds,
        #                                                     return_dict=True,
        #                                                     labels=labels,
        #                                                     **forward_kwargs)

        return super(DecoderLlamaForCausalLM, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True
        )

    def prepare_inputs_for_generation(self, Z_hidden, home_loc_ids):
        return {"Z_hidden": Z_hidden,"home_loc_ids": home_loc_ids}

    def initialize_tokenizer_and_embedding_resize(self):
        special_tokens_dict = dict()
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if self.tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        num_new_tokens_llm = self.tokenizer.add_special_tokens(special_tokens_dict)
        ## Add special tokens related to this work
        num_new_tokens_llm_ours = self.tokenizer.add_tokens(
            [DEFAULT_HIDDEN_TOKEN, DEFAULT_PRE_TOKEN, DEFAULT_START_TOKEN, \
             DEFAULT_END_TOKEN], special_tokens=True)
        num_new_tokens = num_new_tokens_llm + num_new_tokens_llm_ours

        self.config.hidden_token_id, self.config.pre_token_id, \
            self.config.start_token_id, self.config.end_token_id = self.tokenizer.convert_tokens_to_ids(
            [DEFAULT_HIDDEN_TOKEN, DEFAULT_PRE_TOKEN, DEFAULT_START_TOKEN, \
             DEFAULT_END_TOKEN])
        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        self.resize_token_embeddings(len(self.tokenizer))
