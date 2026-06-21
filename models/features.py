import os
import shutil
from glob import glob

import numpy as np
import pretty_midi
import torch
from music21 import analysis, note, stream
from torch.utils.data import Dataset
from tqdm import tqdm

SIXTY_FOURTH_DURATION = 0.0625
PHRASE_IOI_THRESHOLD = 1.5  # beats
UNDEFINED = None

FEATURE_DIM = {
    "pitch": 128,
    "duration": 1,
    "symbolic_duration": 21,
    "interval": 73,        # intervals from -36 to +36
    "onset": 1,
    "contour": 3,
    "pitch_class": 12,
    "beat_position": 1,
    "ioi": 1,
    "scale_degree": 7,
    "key_membership": 1,
    "is_repeated_pitch": 1,
    "register": 3,
    "phrase": 2,
    "cpintfip": 73,        # cumulative pitch interval from first pitch
}


def get_input_size(features):
    return sum(FEATURE_DIM[f] for f in features)


def get_midi_files_from_dir(path):
    files = []
    for filename in glob(path + "/**", recursive=True):
        if filename[filename.rfind("."):] in [".midi", ".mid"]:
            files.append(filename)
    return files


def is_strictly_monophonic(notes):
    if len(notes) < 1:
        return False
    notes = sorted(notes, key=lambda x: x.start)
    for i in range(len(notes) - 1):
        if notes[i].end > notes[i + 1].start:
            return False
    return True


def infer_key_from_notes(notes):
    s = stream.Stream()
    for n in notes:
        m21_note = note.Note()
        m21_note.pitch.midi = n.pitch
        s.insert(n.start, m21_note)
    key = s.analyze("key")
    return key.tonic.pitchClass, key.mode


def pitch_class_to_scale_degree(pitch_class, tonic_pc, mode):
    scale = [0, 2, 4, 5, 7, 9, 11] if mode == "major" else [0, 2, 3, 5, 7, 8, 10]
    rel_pc = (pitch_class - tonic_pc) % 12
    scale_degree = scale.index(rel_pc) if rel_pc in scale else UNDEFINED
    key_membership = int(rel_pc in scale)
    return scale_degree, key_membership


def get_register(pitch):
    if pitch < 48:
        return 0
    elif pitch < 72:
        return 1
    return 2


def get_note_type(duration_ratio):
    base_durations = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    all_durations = []
    for base in base_durations:
        all_durations.extend([base, base * 1.5, base * 1.75])
    return min(range(len(all_durations)), key=lambda i: abs(all_durations[i] - duration_ratio))


def _get_melody_representation(notes, features, filter_sf, beat_duration):
    if filter_sf:
        for n in notes:
            dur_ratio = (n.end - n.start) / beat_duration
            if dur_ratio <= SIXTY_FOURTH_DURATION:
                return None, False

    tonic_pc, mode = None, None
    if "scale_degree" in features or "key_membership" in features:
        tonic_pc, mode = infer_key_from_notes(notes)

    first_pitch = notes[0].pitch
    melody = []

    for i, n in enumerate(notes):
        rep = {}
        pitch = n.pitch
        rep["pitch"] = pitch
        dur = n.end - n.start
        rep["duration"] = dur
        onset = n.start

        if "symbolic_duration" in features:
            rep["symbolic_duration"] = get_note_type(dur / beat_duration)
        if "onset" in features:
            rep["onset"] = onset
        if "interval" in features:
            rep["interval"] = (pitch - notes[i - 1].pitch) if i > 0 else UNDEFINED
        if "contour" in features:
            rep["contour"] = int(np.sign(pitch - notes[i - 1].pitch)) if i > 0 else UNDEFINED

        pitch_class = pitch % 12
        if "pitch_class" in features:
            rep["pitch_class"] = pitch_class
        if "beat_position" in features:
            rep["beat_position"] = (onset % beat_duration) / beat_duration

        ioi = (n.start - notes[i - 1].start) if i > 0 else 0
        if "ioi" in features:
            rep["ioi"] = ioi

        if "scale_degree" in features or "key_membership" in features:
            sd, km = pitch_class_to_scale_degree(pitch_class, tonic_pc, mode)
            if "scale_degree" in features:
                rep["scale_degree"] = sd
            if "key_membership" in features:
                rep["key_membership"] = km

        if "register" in features:
            rep["register"] = get_register(pitch)
        if "phrase" in features:
            rep["phrase"] = 1 if (i == 0 or ioi >= PHRASE_IOI_THRESHOLD * beat_duration) else 0
        if "cpintfip" in features:
            rep["cpintfip"] = pitch - first_pitch
        if "is_repeated_pitch" in features:
            rep["is_repeated_pitch"] = int(i > 0 and pitch == notes[i - 1].pitch)

        melody.append(rep)

    return melody, True


def extract_melody_notes(midi_path, features, filter_sf=False, min_notes=2):
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return []

    tempo = pm.get_tempo_changes()[1][0]
    beat_duration = 60.0 / tempo
    melodies = []

    for instrument in pm.instruments:
        if not instrument.notes:
            continue
        notes = sorted(instrument.notes, key=lambda x: x.start)
        if is_strictly_monophonic(notes):
            melody, ok = _get_melody_representation(notes, features, filter_sf, beat_duration)
            if ok and len(melody) >= min_notes:
                melodies.append(melody)

    return melodies


def process_midi_folder(dataset_path, features, filter_sf=False):
    melodies, seen, all_durations, all_onsets, all_iois = [], set(), set(), set(), set()

    for file_path in tqdm(get_midi_files_from_dir(dataset_path), desc="Processing MIDI files"):
        extracted = extract_melody_notes(file_path, features, filter_sf)
        if not extracted:
            continue
        melody = extracted[0]
        fp = "".join(f"{n['pitch']}{n['duration']:.2f}" for n in melody[:10])
        if fp in seen:
            continue
        seen.add(fp)
        melodies.append(melody)
        for n in melody:
            if "duration" in features:
                all_durations.add(n["duration"])
            if "onset" in features:
                all_onsets.add(n["onset"])
            if "ioi" in features:
                all_iois.add(n["ioi"])

    print(f"Extracted {len(melodies)} melodies")
    return (
        melodies,
        max(all_durations) if all_durations else None,
        max(all_onsets) if all_onsets else None,
        max(all_iois) if all_iois else None,
    )


def get_feature_encoded_vector(feature, vals):
    if feature == "pitch":
        v = torch.zeros(FEATURE_DIM["pitch"])
        v[vals[0]] = 1.0
    elif feature == "duration":
        v = torch.tensor([min(vals[0] / vals[1], 1.0)])
    elif feature == "symbolic_duration":
        v = torch.zeros(FEATURE_DIM["symbolic_duration"])
        v[vals[0]] = 1.0
    elif feature == "interval":
        v = torch.zeros(FEATURE_DIM["interval"])
        if vals[0] is not UNDEFINED:
            idx = vals[0] + 36
            if 0 <= idx < FEATURE_DIM["interval"]:
                v[idx] = 1.0
    elif feature == "onset":
        v = torch.tensor([min(vals[0] / vals[1], 1.0)])
    elif feature == "contour":
        v = torch.zeros(FEATURE_DIM["contour"])
        if vals[0] is not UNDEFINED:
            v[vals[0] + 1] = 1.0
    elif feature == "pitch_class":
        v = torch.zeros(12)
        v[vals[0]] = 1.0
    elif feature == "beat_position":
        v = torch.tensor([vals[0]])
    elif feature == "ioi":
        v = torch.zeros(FEATURE_DIM["ioi"])
        if vals[0] is not UNDEFINED:
            v = torch.tensor([min(vals[0] / vals[1], 1.0)])
    elif feature == "scale_degree":
        v = torch.zeros(FEATURE_DIM["scale_degree"])
        if vals[0] is not UNDEFINED:
            v[vals[0]] = 1.0
    elif feature == "key_membership":
        v = torch.tensor([float(vals[0])])
    elif feature == "is_repeated_pitch":
        v = torch.tensor([float(vals[0])])
    elif feature == "register":
        v = torch.zeros(FEATURE_DIM["register"])
        v[vals[0]] = 1.0
    elif feature == "phrase":
        v = torch.zeros(FEATURE_DIM["phrase"])
        v[1 if vals[0] == 1 else 0] = 1.0
    elif feature == "cpintfip":
        v = torch.zeros(FEATURE_DIM["cpintfip"])
        idx = vals[0] + 36
        if 0 <= idx < FEATURE_DIM["cpintfip"]:
            v[idx] = 1.0
    else:
        raise ValueError(f"Unknown feature: {feature}")
    return v.float()


def get_note_vec(note_rep, max_duration, max_onset, max_ioi, features):
    # Features are encoded in a fixed order to guarantee consistent input vectors.
    ordering = [
        ("pitch", lambda: get_feature_encoded_vector("pitch", [note_rep["pitch"]])),
        ("duration", lambda: get_feature_encoded_vector("duration", [note_rep["duration"], max_duration])),
        ("symbolic_duration", lambda: get_feature_encoded_vector("symbolic_duration", [note_rep["symbolic_duration"]])),
        ("interval", lambda: get_feature_encoded_vector("interval", [note_rep["interval"]])),
        ("onset", lambda: get_feature_encoded_vector("onset", [note_rep["onset"], max_onset])),
        ("contour", lambda: get_feature_encoded_vector("contour", [note_rep["contour"]])),
        ("pitch_class", lambda: get_feature_encoded_vector("pitch_class", [note_rep["pitch_class"]])),
        ("beat_position", lambda: get_feature_encoded_vector("beat_position", [note_rep["beat_position"]])),
        ("ioi", lambda: get_feature_encoded_vector("ioi", [note_rep["ioi"], max_ioi])),
        ("scale_degree", lambda: get_feature_encoded_vector("scale_degree", [note_rep["scale_degree"]])),
        ("key_membership", lambda: get_feature_encoded_vector("key_membership", [note_rep["key_membership"]])),
        ("is_repeated_pitch", lambda: get_feature_encoded_vector("is_repeated_pitch", [note_rep["is_repeated_pitch"]])),
        ("register", lambda: get_feature_encoded_vector("register", [note_rep["register"]])),
        ("phrase", lambda: get_feature_encoded_vector("phrase", [note_rep["phrase"]])),
        ("cpintfip", lambda: get_feature_encoded_vector("cpintfip", [note_rep["cpintfip"]])),
    ]
    parts = [enc() for feat, enc in ordering if feat in features]
    return torch.cat(parts)


class MelodyDataset(Dataset):
    def __init__(self, melodies, max_duration, max_onset, max_ioi, features,
                 dataset_path=None, cache_dir=None):
        self.melodies = melodies
        self.max_duration = max_duration
        self.max_onset = max_onset
        self.max_ioi = max_ioi
        self.features = features
        self.cache_dir = None

        if cache_dir and dataset_path:
            dataset_name = os.path.basename(os.path.normpath(dataset_path))
            self.cache_dir = os.path.join(cache_dir, dataset_name)
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.melodies)

    def __getitem__(self, idx):
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"melody_{idx}.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path)
                return cached["input_sequence"], cached["pitch_sequence"], cached["length"]

        melody = self.melodies[idx]
        input_sequence = [
            get_note_vec(n, self.max_duration, self.max_onset, self.max_ioi, self.features)
            for n in melody
        ]
        pitch_sequence = [n["pitch"] for n in melody]

        input_tensor = torch.stack(input_sequence)
        pitch_tensor = torch.tensor(pitch_sequence, dtype=torch.long)
        seq_len = len(melody)

        if self.cache_dir:
            torch.save({"input_sequence": input_tensor, "pitch_sequence": pitch_tensor, "length": seq_len}, cache_path)

        return input_tensor, pitch_tensor, seq_len
