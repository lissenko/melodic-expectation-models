import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from models.features import get_input_size


def collate_fn(batch):
    inputs, targets, lengths = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-100)
    max_len = max(lengths)
    masks = torch.zeros(len(inputs), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        masks[i, :length] = True
    return inputs_padded, targets_padded, masks


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class MelodyTransformer(nn.Module):
    def __init__(self, features, num_pitches=128, d_model=320, nhead=8,
                 num_layers=5, dim_feedforward=1280, dropout=0.1):
        super().__init__()
        self.features = features
        self.num_pitches = num_pitches
        self.d_model = d_model

        self.input_projection = nn.Linear(get_input_size(features), d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.pitch_out = nn.Linear(d_model, num_pitches)
        self._init_weights()

    def _init_weights(self):
        for m in [self.input_projection, self.pitch_out]:
            m.weight.data.uniform_(-0.1, 0.1)
            m.bias.data.zero_()

    def _causal_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, x, padding_mask):
        x = self.pos_encoder(self.input_projection(x) * math.sqrt(self.d_model))
        out = self.transformer_encoder(
            x,
            mask=self._causal_mask(x.size(1), x.device),
            src_key_padding_mask=~padding_mask,
        )
        return self.pitch_out(self.dropout(out))

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", ckpt)
        model = cls(
            features=cfg["features"],
            num_pitches=cfg.get("num_pitches", 128),
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        norm = {
            "max_duration": ckpt["max_duration"],
            "max_onset": ckpt["max_onset"],
            "max_ioi": ckpt["max_ioi"],
        }
        return model, norm
