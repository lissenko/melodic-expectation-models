import os
from glob import glob
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from models.features import extract_melody_notes, get_note_vec, get_input_size
from models.lstm import MelodyLSTM
from models.transformer import MelodyTransformer


ModelType = Literal["lstm", "transformer"]


def load_single(model_type: ModelType, checkpoint_path: str, device: str = "cpu"):
    cls = MelodyLSTM if model_type == "lstm" else MelodyTransformer
    return cls.from_checkpoint(checkpoint_path, device=device)


def load_ensemble(model_type: ModelType, checkpoints_dir: str, device: str = "cpu"):
    paths = sorted(glob(os.path.join(checkpoints_dir, "**/*.pth"), recursive=True)
                   or glob(os.path.join(checkpoints_dir, "*.pth")))
    if not paths:
        raise FileNotFoundError(f"No .pth files found in {checkpoints_dir}")
    print(f"Loading {len(paths)} {model_type} checkpoints from {checkpoints_dir}")
    return [load_single(model_type, p, device) for p in paths]


def predict_single(model, norm: dict, midi_path: str, device: str = "cpu"):
    melodies = extract_melody_notes(midi_path, model.features, min_notes=2)
    if not melodies:
        raise ValueError(f"No monophonic melody found in {midi_path}")
    melody = melodies[0]

    max_duration = norm["max_duration"]
    max_onset = norm["max_onset"]
    max_ioi = norm["max_ioi"]
    is_lstm = isinstance(model, MelodyLSTM)

    input_dim = get_input_size(model.features)
    current_seq = torch.zeros(1, input_dim, device=device)

    probabilities, ics, entropies, durations = [], [], [], []

    model.eval()
    with torch.no_grad():
        for note_rep in melody:
            seq = current_seq.unsqueeze(0)

            if is_lstm:
                mask = torch.ones(1, seq.size(1), device=device)
                logits, _ = model(seq, mask, hidden=None)
            else:
                mask = torch.ones(1, seq.size(1), dtype=torch.bool, device=device)
                logits = model(seq, mask)

            logits = logits[0, -1]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            true_pitch = note_rep["pitch"]
            p = float(probs[true_pitch])
            ic = -np.log2(p) if p > 0 else np.inf
            H = float(-(probs * np.log2(probs + 1e-12)).sum())

            probabilities.append(probs)
            ics.append(ic)
            entropies.append(H)
            durations.append(note_rep["duration"])

            new_note = get_note_vec(note_rep, max_duration, max_onset, max_ioi, model.features)
            current_seq = torch.cat([current_seq, new_note.unsqueeze(0).to(device)])

    total_dur = sum(durations)
    mdwics = sum(ic * d for ic, d in zip(ics, durations)) / total_dur if total_dur > 0 else float("nan")

    return {"probabilities": probabilities, "ics": ics, "entropies": entropies, "mdwics": mdwics}


def predict_ensemble(ensemble, midi_path: str, device: str = "cpu"):
    all_probs = None
    individual_ics = []

    for model, norm in ensemble:
        result = predict_single(model, norm, midi_path, device=device)
        probs = np.stack(result["probabilities"])
        individual_ics.append(result["ics"])
        all_probs = probs if all_probs is None else all_probs + probs

    mean_probs = all_probs / len(ensemble)

    melodies = extract_melody_notes(midi_path, ensemble[0][0].features, min_notes=2)
    melody = melodies[0]

    ics, entropies, durations = [], [], []
    for i, note_rep in enumerate(melody):
        probs = mean_probs[i]
        p = float(probs[note_rep["pitch"]])
        ics.append(-np.log2(p) if p > 0 else np.inf)
        entropies.append(float(-(probs * np.log2(probs + 1e-12)).sum()))
        durations.append(note_rep["duration"])

    total_dur = sum(durations)
    mdwics = sum(ic * d for ic, d in zip(ics, durations)) / total_dur if total_dur > 0 else float("nan")

    return {
        "probabilities": [mean_probs[i] for i in range(len(melody))],
        "ics": ics,
        "entropies": entropies,
        "mdwics": mdwics,
        "individual_ics": individual_ics,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict melodic expectation from a MIDI file")
    parser.add_argument("midi", help="Path to MIDI file")
    parser.add_argument("model_type", choices=["lstm", "transformer"], help="Architecture to use")
    parser.add_argument("checkpoints", help="Path to a single .pth file or a directory of checkpoints")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    if os.path.isdir(args.checkpoints):
        ensemble = load_ensemble(args.model_type, args.checkpoints, device=args.device)
        result = predict_ensemble(ensemble, args.midi, device=args.device)
        print(f"Ensemble of {len(ensemble)} models")
    else:
        model, norm = load_single(args.model_type, args.checkpoints, device=args.device)
        result = predict_single(model, norm, args.midi, device=args.device)

    print(f"\nMIDI: {args.midi}")
    print(f"Notes: {len(result['ics'])}")
    print(f"Mean IC:   {np.mean(result['ics']):.3f} bits")
    print(f"MDW-IC:    {result['mdwics']:.3f} bits")
    print(f"\nNote-level IC values:")
    for i, ic in enumerate(result["ics"]):
        print(f"  note {i + 1:3d}: IC = {ic:.3f}  entropy = {result['entropies'][i]:.3f}")
