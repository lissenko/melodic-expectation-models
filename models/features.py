import bisect
import hashlib
import json
import math
import os
from fractions import Fraction
import pickle
import shutil
from glob import glob
from multiprocessing import Pool

import numpy as np
import pretty_midi
import torch
from music21 import analysis, note, stream
from torch.utils.data import Dataset
from tqdm import tqdm

FEATURE_ENCODING_VERSION = 3  # bump when any feature's computation changes

SIXTY_FOURTH_DURATION = 0.0625
PHRASE_IOI_THRESHOLD = 1.5  # beats
UNDEFINED = None

FEATURE_DIM = {
    "pitch": 128,
    "duration": 1,
    "symbolic_duration": 21,
    "interval": 255,       # -127..+127, every MIDI interval (IDyOM cpint is unbounded)
    "onset": 1,
    "contour": 3,
    "pitch_class": 12,
    "beat_position": 1,
    "ioi": 1,
    "scale_degree": 7,
    "key_membership": 1,
    "is_repeated_pitch": 1,
    "tessitura": 3,
    "phrase": 2,
    "cpintfip": 255,       # -127..+127, interval from the first pitch
    "cpintfref": 12,       # chromatic interval from the tonic, IDyOM's cpintfref
    "bioi_ratio": 13,      # IDyOM's bioi-ratio, rounded (bioi-ratio-q)
    "bioi_contour": 3,     # shorter / same / longer, from the rounded ratio (bioi-contour-q)
    "dur": 128,            # IDyOM dur, in 32 ms corpus steps (0-127)
    "bioi": 128,           # IDyOM bioi, in 32 ms corpus steps (0-127)
}

# Every feature is computed as IDyOM computes the matching viewpoint
# (viewpoints/melody/*.lisp), timing included: IDyOM times events in MIDI ticks
# rescaled to its timebase and rounded (midi2db.lisp), never in seconds.
IDYOM_TIMEBASE = 500     # IDyOM's timebase in the paper

# Rhythm is IDyOM's bioi-ratio (inter-onset interval over the previous one),
# rounded to the nearest category in log2 space; codes 0 and 12 are below 1/4
# and above 4. A ratio is independent of tempo and articulation, so it means the
# same on the millisecond-timed corpus and on the notated stimuli.
# idyom/viewpoints.lisp defines bioi-ratio-q with the same bounds.
IOI_RATIO_CENTRES = [1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 1, 4 / 3, 3 / 2, 2, 3, 4]
_c = [math.log2(r) for r in IOI_RATIO_CENTRES]
IOI_RATIO_BOUNDS = ([_c[0] - (_c[1] - _c[0]) / 2]
                    + [(a + b) / 2 for a, b in zip(_c, _c[1:])]
                    + [_c[-1] + (_c[-1] - _c[-2]) / 2])

# Pitch features: one encoding per perceptual abstraction, each with an exact
# IDyOM viewpoint (cpitch, cpitch-class, cpint, contour, cpintfip, cpintfref, tessitura).
# DEPRECATED: features of an earlier version, not used by the networks of the paper.
DEPRECATED = {"duration", "symbolic_duration", "onset", "beat_position", "ioi", "scale_degree",
              "key_membership", "is_repeated_pitch", "phrase"}
PITCH_FEATURES = ["pitch", "pitch_class", "interval", "contour", "cpintfip", "cpintfref", "tessitura"]
RHYTHM_FEATURES = ["dur", "bioi", "bioi_ratio", "bioi_contour"]
# The corpus stores time in 32 ms steps (Clean Melodies: delta time and duration
# tokens x 32 ms, at most 127 steps), i.e. 4 IDyOM units at timebase 500; the
# stimuli are re-encoded on the same grid (training/encode_stimuli.py).
STEP_UNITS = IDYOM_TIMEBASE // 125
RATIO_ONE = IOI_RATIO_CENTRES.index(1) + 1   # bioi_ratio code of "same"
FEATURES = PITCH_FEATURES + RHYTHM_FEATURES   # the features of the paper, the default


def idyom_onset(ticks, ppq):
    # midi2db: (round (* (/ timebase 4) (/ ticks ppqn))); both rounds are half-to-even
    return round(Fraction(IDYOM_TIMEBASE, 4) * Fraction(ticks, ppq))


def bioi_ratio_code(prev_bioi, bioi):
    if prev_bioi <= 0 or bioi <= 0:
        return UNDEFINED
    return bisect.bisect_right(IOI_RATIO_BOUNDS, math.log2(bioi / prev_bioi))


def time_steps(units):
    return min(round(units / STEP_UNITS), FEATURE_DIM["dur"] - 1)


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


def get_tessitura(pitch):
    # Boundaries follow IDyOM's tessitura viewpoint (Pearce 2005, p. 206).
    if pitch < 66:
        return 0
    elif pitch > 74:
        return 2
    return 1


def get_note_type(duration_ratio):
    base_durations = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    all_durations = []
    for base in base_durations:
        all_durations.extend([base, base * 1.5, base * 1.75])
    return min(range(len(all_durations)), key=lambda i: abs(all_durations[i] - duration_ratio))


STIMULUS_KEYS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "stimulus_keys.json")
_KEY_SETS = {"fogel": "fogel", "schellenberg": "schellenberg", "manzara": "manzara"}
_known_keys = None


def known_key(midi_path):
    """Notated key of a stimulus, from data/stimulus_keys.json, or None (corpus, Cuddy & Lunney)."""
    global _known_keys
    if _known_keys is None:
        _known_keys = json.load(open(STIMULUS_KEYS)) if os.path.exists(STIMULUS_KEYS) else {}
    folder = os.path.basename(os.path.dirname(os.path.abspath(midi_path)))
    name = os.path.splitext(os.path.basename(midi_path))[0]
    entry = _known_keys.get(_KEY_SETS.get(folder, ""), {}).get(name)
    return tuple(entry) if entry else None


def _get_melody_representation(notes, features, filter_sf, beat_duration, onsets=None, offsets=None, key=None):
    if filter_sf:
        for n in notes:
            dur_ratio = (n.end - n.start) / beat_duration
            if dur_ratio <= SIXTY_FOURTH_DURATION:
                return None, False

    # Key is treated as given, as a notated key signature is in IDyOM's corpora: the notated key
    # for stimuli that have one (known_key), otherwise estimated once from the whole melody.
    tonic_pc, mode = None, None
    if {"scale_degree", "key_membership", "cpintfref"} & set(features):
        tonic_pc, mode = key if key else infer_key_from_notes(notes)

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

        if "cpintfref" in features:
            rep["cpintfref"] = (pitch_class - tonic_pc) % 12
        if "tessitura" in features:
            rep["tessitura"] = get_tessitura(pitch)
        if "phrase" in features:
            rep["phrase"] = 1 if (i == 0 or ioi >= PHRASE_IOI_THRESHOLD * beat_duration) else 0
        if "cpintfip" in features:
            rep["cpintfip"] = pitch - first_pitch if i > 0 else UNDEFINED
        ratio = (bioi_ratio_code(onsets[i - 1] - onsets[i - 2], onsets[i] - onsets[i - 1])
                 if onsets is not None and i >= 2 else UNDEFINED)
        if "bioi_ratio" in features:
            rep["bioi_ratio"] = ratio
        if "bioi_contour" in features:
            rep["bioi_contour"] = UNDEFINED if ratio is UNDEFINED else (ratio > RATIO_ONE) - (ratio < RATIO_ONE)
        if "dur" in features:
            rep["dur"] = time_steps(offsets[i] - onsets[i])
        if "bioi" in features:
            # IDyOM's bioi is 0 for the first event (midi2db.lisp)
            rep["bioi"] = time_steps(onsets[i] - onsets[i - 1]) if i > 0 else 0
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
            onsets = [idyom_onset(pm.time_to_tick(n.start), pm.resolution) for n in notes]
            offsets = [idyom_onset(pm.time_to_tick(n.end), pm.resolution) for n in notes]
            melody, ok = _get_melody_representation(notes, features, filter_sf, beat_duration, onsets, offsets,
                                                    key=known_key(midi_path))
            if ok and len(melody) >= min_notes:
                melodies.append(melody)

    return melodies


def _extract_first_melody(args):
    file_path, features, filter_sf = args
    extracted = extract_melody_notes(file_path, features, filter_sf)
    return extracted[0] if extracted else None


def _cache_path(cache_dir, dataset_path, features, filter_sf, files):
    sig = "\n".join(sorted(files))
    key = repr((os.path.abspath(dataset_path), sorted(features), filter_sf, len(files),
                hashlib.md5(sig.encode()).hexdigest(), FEATURE_ENCODING_VERSION))
    digest = hashlib.md5(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"melodies_{digest}.pkl")


def process_midi_folder(dataset_path, features, filter_sf=False, num_workers=1, cache_dir=None):
    files = get_midi_files_from_dir(dataset_path)

    cache_file = _cache_path(cache_dir, dataset_path, features, filter_sf, files) if cache_dir else None
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached melodies from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    tasks = [(f, features, filter_sf) for f in files]

    if num_workers and num_workers > 1:
        with Pool(num_workers) as pool:
            extracted_melodies = list(tqdm(
                pool.imap(_extract_first_melody, tasks, chunksize=64),
                total=len(tasks), desc="Processing MIDI files",
            ))
    else:
        extracted_melodies = [
            _extract_first_melody(t)
            for t in tqdm(tasks, desc="Processing MIDI files")
        ]

    melodies, seen, all_durations, all_onsets, all_iois = [], set(), set(), set(), set()

    for melody in extracted_melodies:
        if not melody:
            continue
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
    result = (
        melodies,
        max(all_durations) if all_durations else None,
        max(all_onsets) if all_onsets else None,
        max(all_iois) if all_iois else None,
    )

    if cache_file:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Cached melodies to {cache_file}")

    return result


def with_start(inputs):
    """Inputs for predicting every note: a zero start step (no previous note), then all notes but the last.
    Inference feeds the same start step (inference.py), so training and prediction see identical sequences."""
    return torch.cat([inputs.new_zeros(inputs.size(0), 1, inputs.size(2)), inputs[:, :-1]], dim=1)


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
            v[vals[0] + 127] = 1.0
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
    elif feature == "tessitura":
        v = torch.zeros(FEATURE_DIM["tessitura"])
        v[vals[0]] = 1.0
    elif feature == "phrase":
        v = torch.zeros(FEATURE_DIM["phrase"])
        v[1 if vals[0] == 1 else 0] = 1.0
    elif feature == "cpintfref":
        v = torch.zeros(FEATURE_DIM["cpintfref"])
        v[vals[0]] = 1.0
    elif feature in ("dur", "bioi"):
        v = torch.zeros(FEATURE_DIM[feature])
        v[vals[0]] = 1.0
    elif feature == "bioi_contour":
        v = torch.zeros(FEATURE_DIM["bioi_contour"])
        if vals[0] is not UNDEFINED:
            v[vals[0] + 1] = 1.0
    elif feature == "bioi_ratio":
        v = torch.zeros(FEATURE_DIM["bioi_ratio"])
        if vals[0] is not UNDEFINED:
            v[vals[0]] = 1.0
    elif feature == "cpintfip":
        v = torch.zeros(FEATURE_DIM["cpintfip"])
        if vals[0] is not UNDEFINED:
            v[vals[0] + 127] = 1.0
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
        ("tessitura", lambda: get_feature_encoded_vector("tessitura", [note_rep["tessitura"]])),
        ("phrase", lambda: get_feature_encoded_vector("phrase", [note_rep["phrase"]])),
        ("cpintfip", lambda: get_feature_encoded_vector("cpintfip", [note_rep["cpintfip"]])),
        ("cpintfref", lambda: get_feature_encoded_vector("cpintfref", [note_rep["cpintfref"]])),
        ("dur", lambda: get_feature_encoded_vector("dur", [note_rep["dur"]])),
        ("bioi", lambda: get_feature_encoded_vector("bioi", [note_rep["bioi"]])),
        ("bioi_ratio", lambda: get_feature_encoded_vector("bioi_ratio", [note_rep["bioi_ratio"]])),
        ("bioi_contour", lambda: get_feature_encoded_vector("bioi_contour", [note_rep["bioi_contour"]])),
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
