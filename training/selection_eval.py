"""Mean information content (bits per note, every note) of one model on held-out melodies."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.features import MelodyDataset, with_start


def mean_ic(model, melodies, norms, features, collate, device, batch_size=128):
    was_training = model.training
    model.eval()
    loader = DataLoader(MelodyDataset(melodies, *norms, features), batch_size=batch_size,
                        shuffle=False, collate_fn=collate)
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            if hasattr(model, "lstm"):
                logits, _ = model(with_start(x), m, hidden=None)
            else:
                logits = model(with_start(x), m.bool())
            nll = F.cross_entropy(logits.reshape(-1, model.num_pitches), y.reshape(-1).clamp(min=0),
                                  reduction="none")
            mask = m.reshape(-1).float()
            total += float((nll * mask).sum()); n += int(mask.sum())
    model.train(was_training)
    return total / n / np.log(2)
