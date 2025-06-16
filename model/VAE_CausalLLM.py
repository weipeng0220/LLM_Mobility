from torch.distributions import Normal
from Decoder_CausalLLM import *
from VAE_Encoder import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_CausalLLM(nn.Module):
    def __init__(self, params, decoder):
        super(VAE_CausalLLM, self).__init__()
        self.parmas = params
        self.device=params.device
        self.encoder=VAE_Encoder(params)
        self.deocder=decoder
        self.zxpo_mu=nn.Sequential(
            nn.Linear(params.z_hidden_size,512),
            nn.ReLU(),
            nn.Linear(512,params.z_latent_size)
        )
        self.zxpo_std = nn.Sequential(
            nn.Linear(params.z_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, params.z_latent_size),
            nn.Softplus()
        )
        self.clamp_min=params.clamp_min
    def forward(self,trajectory, home_location,mode='train'):
        latent_vector=self.encoder(trajectory,home_location)
        zxpr_rv=Normal(torch.zeros(latent_vector.size()[0]).to(self.device),torch.ones(latent_vector.size()[0]).to(self.device))
        if mode == 'train':
            zxpo_rv=Normal(self.zxpo_mu(latent_vector),self.zxpo_std(latent_vector)+self.clamp_min)
            z=zxpo_rv.rsample()
        else:
            z=zxpr_rv.rsample()
        output=self.decoder(z,home_location,trajectory)
        return output








