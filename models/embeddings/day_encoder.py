import torch
import torch.nn as nn
import math


class DayEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, min_val: float = 1.0, max_val: float = 31.0):
        super().__init__()
        self.d_model = d_model
        self.min_val = min_val
        self.max_val = max_val
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = (pe + 1) / 2
        pe = pe * (max_val - min_val) + min_val
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len]  # type: ignore
        return x + pe

