import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.features import MelodyDataset, process_midi_folder, FEATURE_DIM
from models.transformer import MelodyTransformer, collate_fn

PAPER_FEATURES = ["pitch", "contour", "duration", "scale_degree", "ioi"]


def train(model, loader, criterion, optimizer, device, clip_norm):
    model.train()
    total_loss = 0.0
    for inputs, targets, masks in tqdm(loader, leave=False):
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

        input_seq = inputs[:, :-1]
        target_seq = targets[:, 1:]
        input_masks = masks[:, :-1]
        target_masks = masks[:, 1:]

        optimizer.zero_grad()
        logits = model(input_seq, input_masks)

        logits = logits.reshape(-1, model.num_pitches)
        target_seq = target_seq.reshape(-1)
        target_masks_flat = target_masks.reshape(-1).float()

        loss = (criterion(logits, target_seq) * target_masks_flat).sum() / target_masks_flat.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train Transformer melody model")
    parser.add_argument("--data", required=True, help="Directory of training MIDI files")
    parser.add_argument("--out", required=True, help="Output checkpoint path (e.g. model.pth)")
    parser.add_argument("--features", default=",".join(PAPER_FEATURES),
                        help="Comma-separated feature names")
    parser.add_argument("--d_model", type=int, default=320)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dim_feedforward", type=int, default=1280)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    features = [f.strip() for f in args.features.split(",")]
    for f in features:
        if f not in FEATURE_DIM:
            raise ValueError(f"Unknown feature '{f}'. Valid: {sorted(FEATURE_DIM)}")

    print(f"Loading MIDI files from {args.data} ...")
    melodies, max_duration, max_onset, max_ioi = process_midi_folder(args.data, features)
    dataset = MelodyDataset(melodies, max_duration, max_onset, max_ioi, features)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    print(f"  {len(dataset)} melodies, {len(loader)} batches/epoch")

    model = MelodyTransformer(
        features=features,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        loss = train(model, loader, criterion, optimizer, args.device, args.clip_norm)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{start_epoch + args.epochs}  loss={loss:.4f}  lr={lr:.2e}")

        stem = args.out.replace(".pth", "")
        ckpt_path = f"{stem}_epoch{epoch + 1}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "features": features,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
            "max_duration": max_duration,
            "max_onset": max_onset,
            "max_ioi": max_ioi,
            "epoch": epoch + 1,
        }, ckpt_path)

    print(f"Training complete. Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
