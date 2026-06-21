import argparse
import json
import os
import re

import numpy as np


EXPERIMENTS = ["manzara", "cuddy_lunney", "schellenberg", "fogel"]
WESTERN_EXPERIMENTS = ["western"]
MODEL_TYPES = ["lstm", "transformer", "temperley"]


def midi_key_to_id(path: str, experiment: str) -> str:
    basename = os.path.basename(path)
    stem = os.path.splitext(basename)[0]

    if experiment == "schellenberg":
        m = re.search(r"(\d+)$", stem)
        return m.group(1) if m else stem

    return stem


def process_neural(data: dict, experiment: str, ics_only: bool = False) -> dict:
    out = {}
    for midi_path, checkpoints in data.items():
        mid_id = midi_key_to_id(midi_path, experiment)
        values = list(checkpoints.values())

        out[mid_id] = {
            "ics": np.mean([v["ics"] for v in values], axis=0).tolist(),
        }
        if not ics_only:
            out[mid_id]["entropies"] = np.mean([v["entropies"] for v in values], axis=0).tolist()
            out[mid_id]["mdwics"] = float(np.mean([v["mdwics"] for v in values]))
            out[mid_id]["last_note_probs"] = np.mean(
                [v["last_note_probs"] for v in values], axis=0
            ).tolist()
    return out


def process_temperley(data: dict, experiment: str, ics_only: bool = False) -> dict:
    out = {}
    for midi_path, values in data.items():
        mid_id = midi_key_to_id(midi_path, experiment)
        out[mid_id] = {"ics": values["ics"]}
        if not ics_only:
            out[mid_id]["last_note_probs"] = values.get("last_note_probs", [])
            if "probs" in values:
                out[mid_id]["probs"] = values["probs"]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw prediction JSONs to clean public-release format."
    )
    parser.add_argument(
        "--src_dir",
        required=True,
        help="Directory containing the raw prediction JSON files.",
    )
    parser.add_argument(
        "--lstm_prefix",
        default="reduced_model_lstm",
        help=(
            "Filename prefix for LSTM JSON files "
            "(default: 'reduced_model_lstm', yielding e.g. "
            "reduced_model_lstm_train_tegridy_test_manzara.json)."
        ),
    )
    args = parser.parse_args()

    dest_root = os.path.dirname(__file__)

    all_exps = [(exp, False) for exp in EXPERIMENTS] + [(exp, True) for exp in WESTERN_EXPERIMENTS]

    for model in MODEL_TYPES:
        dest_dir = os.path.join(dest_root, model)
        os.makedirs(dest_dir, exist_ok=True)

        for exp, ics_only in all_exps:
            prefix = args.lstm_prefix if model == "lstm" else model
            src_file = os.path.join(args.src_dir, f"{prefix}_train_tegridy_test_{exp}.json")
            if not os.path.exists(src_file):
                print(f"  MISSING: {src_file}")
                continue

            with open(src_file) as f:
                raw = json.load(f)

            if model == "temperley":
                clean = process_temperley(raw, exp, ics_only=ics_only)
            else:
                clean = process_neural(raw, exp, ics_only=ics_only)

            dest_file = os.path.join(dest_dir, f"{exp}.json")
            with open(dest_file, "w") as f:
                json.dump(clean, f, separators=(",", ":"))

            size_kb = os.path.getsize(dest_file) // 1024
            print(f"  {model}/{exp}.json  ({size_kb} KB,  {len(clean)} melodies)")


if __name__ == "__main__":
    main()
