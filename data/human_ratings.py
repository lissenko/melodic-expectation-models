import csv
import os

_HERE = os.path.dirname(__file__)


# Chorale 151 (29 notes)
MANZARA_IC_SHORT = [
    2.933497536945813, 2.206896551724139, 2.3300492610837438,
    3.142857142857143, 2.1330049261083746, 3.0689655172413794,
    1.01231527093596, 2.0221674876847295, 2.182266009852217,
    3.783251231527094, 2.5024630541871926, 2.083743842364532,
    1.9359605911330053, 0.8399014778325116, 1.0,
    1.5788177339901486, 1.0369458128078817, 3.376847290640394,
    4.4236453201970445, 2.096059113300493, 2.403940886699508,
    2.724137931034483, 5.692118226600986, 1.6157635467980302,
    1.3694581280788176, 0.7413793103448283, 1.4064039408866993,
    0.6674876847290641, 0.5935960591133007,
]

# Chorale 61 (57 notes)
MANZARA_IC_LONG = [
    4.374149659863946, 4.591836734693878, 1.9591836734693886,
    2.455782312925171, 1.1428571428571432, 1.1020408163265314,
    0.36054421768707545, 3.1224489795918373, 2.2312925170068034,
    1.0680272108843543, 1.2653061224489806, 2.0816326530612255,
    2.551020408163266, 0.2925170068027221, 2.013605442176871,
    4.054421768707483, 2.224489795918368, 0.8435374149659873,
    1.204081632653062, 1.823129251700681, 2.2176870748299327,
    1.1360544217687085, 1.5238095238095246, 0.7823129251700687,
    3.4285714285714293, 2.3061224489795924, 1.0952380952380958,
    2.2721088435374157, 3.1700680272108848, 0.7414965986394568,
    2.3265306122448988, 0.904761904761906, 1.034013605442178,
    1.9455782312925174, 3.224489795918368, 3.3061224489795924,
    1.4897959183673475, 2.6326530612244903, 0.10204081632653139,
    1.312925170068028, 0.7619047619047628, 0.25850340136054495,
    2.3605442176870755, 3.7959183673469394, 3.8163265306122454,
    0.9387755102040822, 1.2653061224489806, 0.6938775510204094,
    0.5442176870748305, 1.9931972789115653, 1.1496598639455788,
    0.43537414965986443, 1.3537414965986398,
    0.44897959183673564, 0.387755102040817, 0.2040816326530619,
    0.07482993197278986,
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
