"""Predictions of a network ensemble on the stimuli of the four datasets, in the format of evaluate.py.

    python evaluation/predict_stimuli.py --arch lstm --checkpoints checkpoints/lstm --out predictions/my_lstm

Writes <out>/<dataset>.json. Each network's distributions are averaged over the ensemble.
"""
import argparse
import json
import os
import re
import sys
from glob import glob

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from inference import load_ensemble, predict_single
from models.features import extract_melody_notes

STIM = os.path.join(ROOT, "data", "stimuli")
SETS = ["cuddy_lunney", "schellenberg", "manzara", "fogel"]


def key_of(path, dataset):
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.search(r"(\d+)$", stem).group(1) if dataset == "schellenberg" else stem


def predict_stimuli(arch, checkpoints, out, device="cpu", sets=SETS):
    ensemble = load_ensemble(arch, checkpoints, device=device)
    features = ensemble[0][0].features
    os.makedirs(out, exist_ok=True)
    for dataset in sets:
        res = {}
        for f in sorted(glob(os.path.join(STIM, dataset, "*.mid"))):
            pitches = [n["pitch"] for n in extract_melody_notes(f, features, min_notes=2)[0]]
            per = [predict_single(m, norm, f, device=device) for m, norm in ensemble]
            probs = np.mean([np.stack(r["probabilities"]) for r in per], axis=0)
            res[key_of(f, dataset)] = {"ics": [float(-np.log2(max(probs[i][p], 1e-300))) for i, p in enumerate(pitches)],
                                       "last_note_probs": np.mean([r["last_note_probs"] for r in per], axis=0).tolist()}
        json.dump(res, open(os.path.join(out, f"{dataset}.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["lstm", "transformer"], required=True)
    ap.add_argument("--checkpoints", required=True, help="a folder of .pth checkpoints (one or more)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    predict_stimuli(a.arch, a.checkpoints, a.out, a.device)
