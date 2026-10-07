import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.features import DEPRECATED, FEATURES, with_start, MelodyDataset, process_midi_folder, FEATURE_DIM
from training.split import split_corpus
from training import optim_config
from training.selection_eval import mean_ic
from models.transformer import MelodyTransformer, collate_fn


def train(model, loader, criterion, optimizer, device, clip_norm):
    model.train()
    total_loss = 0.0
    for inputs, targets, masks in tqdm(loader, leave=False):
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

        input_seq = with_start(inputs)
        target_seq = targets
        input_masks = target_masks = masks

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
    parser.add_argument("--features", default=",".join(FEATURES),
                        help="Comma-separated feature names")
    parser.add_argument("--d_model", type=int, default=320)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dim_feedforward", type=int, default=1280)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=optim_config.EPOCHS["transformer"])
    parser.add_argument("--tmax", type=int, default=optim_config.TMAX,
                        help="epochs the cosine decay is shaped for; training stops at --epochs")
    parser.add_argument("--lr", type=float, default=optim_config.LR["transformer"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for preprocessing and data loading")
    parser.add_argument("--cache_dir", default=None,
                        help="Cache extracted melodies here; reused across runs with the same data/features")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--finetune_from", default=None,
                        help="Load model WEIGHTS ONLY from this checkpoint and start a "
                             "fresh optimiser and schedule. Unlike --resume this does not "
                             "restore optimiser state or the epoch counter, which is what "
                             "you want when continuing on a different corpus.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep_epochs", action="store_true", help="keep one checkpoint per epoch instead of the last")
    parser.add_argument("--no_holdout", action="store_true",
                        help="train on every melody (by default, 400 + 1,000 are held out, training/split.py)")
    parser.add_argument("--eval_selection", action="store_true",
                        help="after each epoch, report mean IC on the selection slice (training/split.py)")
    parser.add_argument("--curve", default=None, help="append one JSON line per epoch here")
    parser.add_argument("--no_save", action="store_true", help="do not write checkpoints (tuning runs)")
    parser.add_argument("--patience", type=int, default=0,
                        help="with --curve: stop when the selection IC has not improved for this many epochs")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    features = [f.strip() for f in args.features.split(",")]
    for f in features:
        if f not in FEATURE_DIM:
            raise ValueError(f"Unknown feature '{f}'. Valid: {sorted(FEATURE_DIM)}")
    if set(features) & DEPRECATED:
        print(f"Warning: deprecated features: {sorted(set(features) & DEPRECATED)}")

    print(f"Loading MIDI files from {args.data} ...")
    # a subset of the full representation is read from the full representation's cache: the same
    # notes and values, only the encoded features differ, and no feature extraction is repeated
    load_features = FEATURES if set(features) <= set(FEATURES) else features
    melodies, max_duration, max_onset, max_ioi = process_midi_folder(
        args.data, load_features, num_workers=args.workers, cache_dir=args.cache_dir)
    if args.no_holdout:
        sel, test = [], []
    else:
        melodies, sel, test = split_corpus(melodies)
        print(f"  holding out {len(sel)} selection + {len(test)} test melodies")
    dataset = MelodyDataset(melodies, max_duration, max_onset, max_ioi, features)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.workers, pin_memory=True,
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
    optimizer, scheduler = optim_config.build(
        model, args.lr, tmax=(args.tmax or args.epochs))

    start_epoch = 0
    if args.finetune_from:
        ck = torch.load(args.finetune_from, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print(f"Fine-tuning from {args.finetune_from} (weights only)")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    best_ic, best_epoch, stop = float("inf"), 0, False
    for epoch in range(start_epoch, start_epoch + args.epochs):
        loss = train(model, loader, criterion, optimizer, args.device, args.clip_norm)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{start_epoch + args.epochs}  loss={loss:.4f}  lr={lr_now:.2e}")
        if (args.eval_selection or args.curve) and sel:
            sel_ic = mean_ic(model, sel, (max_duration, max_onset, max_ioi), features, collate_fn, args.device)
            print(f"  selection mean IC {sel_ic:.4f} bits")
            if args.curve:
                with open(args.curve, "a") as fh:
                    fh.write(json.dumps({"epoch": epoch + 1, "loss": loss, "lr": lr_now,
                                         "selection_ic": sel_ic}) + "\n")
            if args.patience:
                if sel_ic < best_ic:
                    best_ic, best_epoch = sel_ic, epoch + 1
                stop = epoch + 1 - best_epoch >= args.patience
        if args.no_save:
            continue

        ckpt_path = args.out.replace(".pth", f"_epoch{epoch + 1}.pth") if args.keep_epochs else args.out
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
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
        if stop:
            print(f"Early stop: best selection IC at epoch {best_epoch}")
            break

    print("Training complete." + ("" if args.no_save else f" Final checkpoint: {ckpt_path}"))


if __name__ == "__main__":
    main()
