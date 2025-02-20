import torch
import math

from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class GPT2LikeModel(nn.Module):
    def __init__(self, emb_dim=1024, vocab_size=30522):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(emb_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=8, batch_first=True)
        self.decoder = nn.Linear(emb_dim, vocab_size)
        
    def forward(self, x, masks):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        if x is None:
            x = self.encoder_layer(x, src_mask=generate_square_subsequent_mask(x.size(1)), is_causal=True)
        else:
            x = self.encoder_layer(x, src_mask=masks)
        return self.decoder(x)

    

def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
