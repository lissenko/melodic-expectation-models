"""Write the partition of the corpus as a list of file names per set: data/partition.json.

    python training/export_partition.py datasets/tegridy_midi_melodies

Each melody of the corpus is read as in training, exact duplicates are removed in the same order, and each
remaining melody is assigned to its set by training/split.py. Files that give no strictly monophonic melody,
and exact duplicates of an earlier file, are listed as excluded."""
import json
import os
import sys
from multiprocessing import Pool

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from models.features import FEATURES, _extract_first_melody, get_midi_files_from_dir
from training.split import _roles, melody_key

NAMES = {"fit": "training", "selection": "validation", "test": "test"}


def main(folder, out=os.path.join(ROOT, "data", "partition.json")):
    files = get_midi_files_from_dir(folder)
    with Pool(8) as pool:
        extracted = pool.map(_extract_first_melody, [(f, FEATURES, False) for f in files], chunksize=64)
    kept, seen, excluded = [], set(), {"no_monophonic_melody": [], "exact_duplicate": []}
    for f, m in zip(files, extracted):
        name = os.path.relpath(f, folder)
        if not m:
            excluded["no_monophonic_melody"].append(name)
            continue
        fp = "".join(f"{n['pitch']}{n['duration']:.2f}" for n in m[:10])
        if fp in seen:
            excluded["exact_duplicate"].append(name)
            continue
        seen.add(fp)
        kept.append((name, m))
    roles = _roles([m for _, m in kept])
    sets = {v: [] for v in NAMES.values()}
    for name, m in kept:
        sets[NAMES[roles[melody_key(m)]]].append(name)
    key = lambda n: int(''.join(c for c in n if c.isdigit()) or 0)
    result = {k: sorted(v, key=key) for k, v in sets.items()}
    result["excluded"] = {k: sorted(v, key=key) for k, v in excluded.items()}
    with open(out, "w") as fh:
        json.dump(result, fh)
    print({k: len(v) for k, v in sets.items()}, {k: len(v) for k, v in excluded.items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
