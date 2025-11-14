from typing import List, Optional, Union

import torch
import torch.nn as nn

from models.embeddings.day_encoder import DayEncoding
from models.embeddings.text_encoder import TextEncoder


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        text_embed_dim: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        max_seq_length: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.device = device or torch.device("cpu")
        
        if text_encoder is None:
            self.text_encoder = TextEncoder(device=self.device)
            text_embed_dim = 1536
        else:
            self.text_encoder = text_encoder
            if text_embed_dim is None:
                text_embed_dim = 1536
        
        self.text_projection = nn.Linear(text_embed_dim, d_model)
        
        self.day_embedding = nn.Embedding(31, d_model)
        
        self.day_encoding = DayEncoding(d_model, max_len=32, min_val=1.0, max_val=31.0)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        descriptions: Union[str, List[str]],
        days: Union[int, List[int], torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        
        text_embeddings = self.text_encoder(descriptions)
        text_embeddings = text_embeddings.to(self.device)
        
        text_projected = self.text_projection(text_embeddings)
        
        if isinstance(days, (int, list)):
            if isinstance(days, int):
                days = [days]
            days_tensor = torch.tensor(days, dtype=torch.long, device=self.device) - 1
        else:
            days_tensor = days.to(self.device) - 1
        
        days_tensor = torch.clamp(days_tensor, 0, 30)
        day_embeddings = self.day_embedding(days_tensor)
        
        combined = text_projected + day_embeddings
        
        combined = combined.unsqueeze(1)
        
        pe = self.day_encoding.pe[:, days_tensor, :]  # type: ignore
        combined = combined + pe
        
        combined = self.dropout(combined)
        
        output = self.transformer_encoder(combined)
        
        return output

