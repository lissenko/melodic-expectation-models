"""Optimiser, schedule and tuned values, defined once for every trainer.

The optimiser and schedule are identical across architectures so that neither is
a confound in the architecture comparison; learning rate and epoch count are
tuned per architecture by the same procedure.
"""
import torch.optim as optim

# Learning rate and epoch count per architecture, chosen on the configuration set of the
# paper. Both use cosine decay over TMAX epochs and stop early at EPOCHS.
LR = {"lstm": 1e-3, "transformer": 1e-3}
EPOCHS = {"lstm": 9, "transformer": 12}
TMAX = 30
ETA_MIN = 1e-6
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.98)
EPS = 1e-9


def build(model, lr, tmax=TMAX):
    opt = optim.AdamW(model.parameters(), lr=lr, betas=BETAS,
                      eps=EPS, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tmax, eta_min=ETA_MIN)
    return opt, sched
