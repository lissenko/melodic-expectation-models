import csv
import os

_HERE = os.path.dirname(__file__)


# Manzara et al. (1992): uncertainty of each note in bits, averaged over participants (the average capital
# estimate; 15 participants for Chorale 151, the 6 finalists for Chorale 61), as plotted by Pearce (2005,
# Figures 8.7 and 8.8). Exact values read from the vector figures: data/extract_manzara.py.

# Chorale 151 (29 notes)
MANZARA_IC_SHORT = [
    2.9310, 2.2068, 2.3366, 3.1505, 2.1305, 3.0688, 1.0069, 2.0164,
    2.1832, 3.7889, 2.4907, 2.0689, 1.9278, 0.8295, 0.9973, 1.5839,
    1.0301, 3.3681, 4.4257, 2.0848, 2.4028, 2.7326, 5.6970, 1.6048,
    1.3656, 0.7311, 1.4004, 0.6498, 0.3626,
]

# Chorale 61 (57 notes)
MANZARA_IC_LONG = [
    4.3729, 4.6006, 1.9500, 2.4679, 1.1406, 1.1013, 0.3545, 3.1267,
    2.2098, 1.0632, 1.2650, 2.0890, 2.5630, 0.2779, 2.0129, 4.0581,
    2.0701, 0.8392, 1.2160, 1.8768, 2.2287, 1.1293, 1.5316, 0.7805,
    3.4340, 2.2797, 1.0911, 2.3036, 3.1757, 0.7414, 2.3365, 0.9085,
    1.0392, 1.9794, 3.2270, 3.3115, 1.4927, 2.6410, 0.1041, 1.3180,
    0.7572, 0.2589, 2.4482, 3.8045, 3.8228, 0.9346, 1.2746, 0.6885,
    0.5451, 1.9922, 1.0808, 0.4357, 1.3588, 0.4497, 0.3844, 0.2014,
    0.0740,
]


def load_cuddy_lunney(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(_HERE, "human", "cuddy_lunney.csv")

    data = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            context = f"{row['Interval_Type']}_{row['Direction']}"
            tone_idx = int(row["Continuation_Tone"])
            midi = 66 - 12 + (tone_idx - 1)   # F#4 ± 12 semitones
            rating = (float(row["Trained_M"]) + float(row["Untrained_M"])) / 2.0
            data.setdefault(context, {})[midi] = rating
    return data


def load_schellenberg(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(_HERE, "human", "schellenberg.csv")

    data = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frag = str(row["Fragment"])
            midi = int(row["MIDI_Number"])
            rating = float(row["Value"])
            data.setdefault(frag, {})[midi] = rating
    return data


def load_fogel(human_csv=None, tonic_csv=None):
    if human_csv is None:
        human_csv = os.path.join(_HERE, "human", "fogel_human_data.csv")
    if tonic_csv is None:
        tonic_csv = os.path.join(_HERE, "human", "fogel_tonic_mapping.csv")

    human_data = []
    with open(human_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sd = int(row["sd"])
            except (ValueError, KeyError):
                continue   # skip rows where participant didn't produce a sung note
            human_data.append({
                "subject": row["subject"],
                "code": row["code"],
                "note": row["note"],
                "sd": sd,
            })

    tonic_map = {}
    with open(tonic_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tonic_map[row["code"]] = int(row["tonic"])

    return human_data, tonic_map
