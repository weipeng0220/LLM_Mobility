import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEncoder(nn.Module):
    '''
    Revised from pytorch tutorial
    '''
    def __init__(self, dim, dropout, seq_len):
        super().__init__()

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(seq_len, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pe', pe)

    def forward(self, x):

        # (B, L, E) -> (L, B, E)
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        # (L, B, E) -> (B, L, E)
        x = x.transpose(0, 1)

        return self.dropout(x)

class VAE_Encoder(nn.Module):
    def __init__(self,params):
        super(VAE_Encoder, self).__init__()
        self.params = params
        self.device=self.params.device
        self.loc_embedding = nn.Embedding(self.params.loc_max+1, self.params.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.params.embedding_dim,
            nhead=self.params.head_num,
            dim_feedforward=self.params.dim_forward,
            dropout=self.params.dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.params.layer_num
        )

        self.to_latent = nn.Linear(
            self.params.embedding_dim,
            self.params.z_hidden_size
        )

    def forward(self, trajectory, home_location):
        """
        Forward pass for trajectory encoding and latent representation generation.

        Args:
            trajectory (Tensor): [batch_size, seq_len] - Sequence of location indices.
            home_location (Tensor): [batch_size,] - Home location indices.

        Returns:
            Tensor: Latent representation tensor.
        """
        # Embed the input trajectory and home location
        traj_embed = self.loc_embedding(trajectory)  # [batch_size, seq_len, embedding_dim]
        home_embed = self.loc_embedding(home_location)  # [batch_size, embedding_dim]

        # Positional encoding for the trajectory embeddings
        pos_encoder = PositionEncoder(
            self.params.embedding_dim, self.dropout, self.traj_length
        ).to(self.device)
        encoded_input = pos_encoder(traj_embed)

        # Generate padding mask for Transformer
        padding_mask = self.get_key_padding_mask(
            trajectory, self.params.padding_value
        ).to(self.device)

        # Pass through Transformer encoder
        transformer_output = self.transformer_encoder(
            encoded_input, src_key_padding_mask=padding_mask
        )

        # Average the sequence outputs across the time dimension
        sequence_summary = torch.mean(transformer_output, dim=1)  # [batch_size, embedding_dim]

        # Concatenate with home embedding and project to latent space
        combined_input = torch.cat([sequence_summary, home_embed], dim=1)  # [batch_size, 2 * embedding_dim]
        latent_vector = self.to_latent(combined_input)  # [batch_size, latent_dim]

        return latent_vector

    def get_key_padding_mask(self, tokens, padding_value):
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == padding_value] = 1  # float('-inf')
        return key_padding_mask.bool()






