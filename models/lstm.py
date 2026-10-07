import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from models.features import FEATURE_DIM, get_input_size


def collate_fn(batch):
    inputs, targets, lengths = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-100)
    max_len = max(lengths)
    masks = torch.zeros(len(inputs), max_len)
    for i, length in enumerate(lengths):
        masks[i, :length] = 1.0
    return inputs_padded, targets_padded, masks


class MelodyLSTM(nn.Module):
    def __init__(self, features, num_pitches=128, hidden_size=720, num_layers=2, dropout=0.5):
        super().__init__()
        self.features = features
        self.num_pitches = num_pitches
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=get_input_size(features),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.pitch_out = nn.Linear(hidden_size, num_pitches)

    def forward(self, x, masks, hidden=None):
        lengths = masks.sum(dim=1).cpu().long()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(packed, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return self.pitch_out(self.dropout(output)), hidden

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            features=ckpt["features"],
            num_pitches=128,
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        norm = {
            "max_duration": ckpt["max_duration"],
            "max_onset": ckpt["max_onset"],
            "max_ioi": ckpt["max_ioi"],
        }
        return model, norm
