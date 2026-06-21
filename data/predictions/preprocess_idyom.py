import argparse
import json
import math
import os

import numpy as np
import pandas as pd


DAT_FILES = {
    "manzara":      "manzara.dat",
    "cuddy_lunney": "cuddy_lunney.dat",
    "schellenberg": "schellenberg.dat",
    "fogel":        "fogel.dat",
    "western":      "western.dat",
}


def melody_name_to_id(name: str, experiment: str) -> str:
    name = name.strip('"')
    if experiment == "schellenberg":
        return name.replace("fragment", "")
    return name


def dat_to_json(path: str, experiment: str, ics_only: bool = False) -> dict:
    df = pd.read_csv(path, sep=" ")

    pitch_cols = {
        int(c.split(".")[-1]): c
        for c in df.columns
        if c.startswith("cpitch.") and c.split(".")[-1].isdigit()
    }

    out = {}
    for raw_name, group in df.groupby("melody.name", sort=False):
        mid_id = melody_name_to_id(str(raw_name), experiment)
        group = group.sort_values("note.id")

        out[mid_id] = {"ics": group["cpitch.ic"].tolist()}

        if not ics_only:
            out[mid_id]["entropies"] = group["cpitch.entropy"].tolist()
            last_row = group.iloc[-1]
            last_note_probs = [0.0] * 128
            for midi, col in pitch_cols.items():
                if midi < 128:
                    last_note_probs[midi] = float(last_row[col])
            out[mid_id]["last_note_probs"] = last_note_probs

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        default=os.path.join(os.path.dirname(__file__), "idyom", "raw"),
        help="Directory containing the raw IDyOM .dat files.",
    )
    args = parser.parse_args()

    dest_root = os.path.join(os.path.dirname(__file__), "idyom")
    os.makedirs(dest_root, exist_ok=True)

    for exp, rel_path in DAT_FILES.items():
        src_file = os.path.join(args.src_dir, rel_path)
        if not os.path.exists(src_file):
            print(f"  MISSING: {src_file}")
            continue

        ics_only = exp == "western"
        clean = dat_to_json(src_file, exp, ics_only=ics_only)

        dest_file = os.path.join(dest_root, f"{exp}.json")
        with open(dest_file, "w") as f:
            json.dump(clean, f, separators=(",", ":"))

        size_kb = os.path.getsize(dest_file) // 1024
        print(f"  idyom/{exp}.json  ({size_kb} KB,  {len(clean)} melodies)")


if __name__ == "__main__":
    main()
