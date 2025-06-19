import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from model.Decoder_CausalLLM import *
from model.VAE_Encoder import *

class VAE_CausalLLM(nn.Module):
    def __init__(self, params, decoder):
        super(VAE_CausalLLM, self).__init__()
        self.params = params
        self.device=params.device
        self.encoder=VAE_Encoder(params)
        self.decoder=decoder
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
    def forward(self,trajectory, home_location):
        latent_vector=self.encoder(trajectory,home_location)
        zxpr_rv = Normal(torch.zeros(home_location.size()[0], self.params.z_latent_size).to(self.device),
                         torch.ones(home_location.size()[0], self.params.z_latent_size).to(self.device))
        zxpo_rv = Normal(self.zxpo_mu(latent_vector), self.zxpo_std(latent_vector) + self.clamp_min)
        z = zxpo_rv.rsample()
        output=self.decoder(Z_hidden=z,home_loc_ids=home_location,traj_target=trajectory)
        kl_loss=self.get_kl_loss(zxpr_rv,zxpo_rv)
        return output,kl_loss
    def generate(self,home_location):
        zxpr_rv = Normal(torch.zeros(home_location.size()[0],self.params.z_latent_size).to(self.device),
                         torch.ones(home_location.size()[0],self.params.z_latent_size).to(self.device))
        z=zxpr_rv.rsample()
        output=self.decoder.generate(z,home_location)
        return output
    def get_kl_loss(self,zxpr,zxpo):
        return kl_divergence(zxpo, zxpr).mean()









