import numpy as np
from scipy.stats import norm

import music21

from models.features import extract_melody_notes, process_midi_folder


def estimate_parameters(midi_dir):
    extracted, _, _, _ = process_midi_folder(midi_dir, ["pitch"])
    all_melodies = [[n["pitch"] for n in mel] for mel in extracted if mel]
    return _compute_parameters(all_melodies)


def _compute_parameters(all_melodies):
    central_pitches, first_notes = [], []
    major_pc = np.zeros(12)
    minor_pc = np.zeros(12)
    major_count = minor_count = 0

    for melody in all_melodies:
        if len(melody) < 1:
            continue
        first_notes.append(melody[0])
        central_pitches.append(int(round(np.mean(melody))))

        if len(melody) < 4:
            continue
        s = music21.stream.Stream()
        for p in melody:
            s.append(music21.note.Note(p))
        try:
            key = s.analyze("key")
        except Exception:
            continue
        tonic = key.tonic.midi % 12
        for p in melody:
            sd = (p - tonic) % 12
            if key.mode == "major":
                major_pc[sd] += 1
                major_count += 1
            else:
                minor_pc[sd] += 1
                minor_count += 1

    central_pitch_mean = float(np.mean(central_pitches))
    central_pitch_var = float(np.var(central_pitches, ddof=0))

    # variance: variance of each melody's first note around its own central pitch
    vr = float(np.mean([(fn - c) ** 2 for fn, c in zip(first_notes, central_pitches)]))

    # Proximity variance: estimated from notes following the central pitch
    deviations = []
    for melody, c in zip(all_melodies, central_pitches):
        for i in range(1, len(melody)):
            if melody[i - 1] == c:
                deviations.append((melody[i] - c) ** 2)
    vc = float(np.mean(deviations)) if deviations else 7.2
    vp = float((vr * vc) / (vr - vc)) if vr > vc > 0 else 7.2

    total = major_count + minor_count
    prob_major = major_count / total if total > 0 else 0.5

    major_profile = major_pc / major_pc.sum() if major_pc.sum() > 0 else np.ones(12) / 12
    minor_profile = minor_pc / minor_pc.sum() if minor_pc.sum() > 0 else np.ones(12) / 12

    return {
        "central_pitch_mean": central_pitch_mean,
        "central_pitch_var": central_pitch_var,
        "vr": vr,
        "vp": vp,
        "prob_major": prob_major,
        "major_profile": major_profile,
        "minor_profile": minor_profile,
    }


def _key_profile(tonic_pc, is_major, major_profile, minor_profile, n=128):
    profile = major_profile if is_major else minor_profile
    kp = np.array([profile[(p - tonic_pc) % 12] for p in range(n)])
    return kp / kp.sum()


def _range_profile(central_pitch, vr, n=128):
    p = norm.pdf(np.arange(n), loc=central_pitch, scale=np.sqrt(vr))
    return p / p.sum()


def _proximity_profile(prev_pitch, vp, n=128):
    p = norm.pdf(np.arange(n), loc=prev_pitch, scale=np.sqrt(vp))
    return p / p.sum()


def _central_pitch_profile(mean, var, n=128):
    p = norm.pdf(np.arange(n), loc=mean, scale=np.sqrt(var))
    return p / p.sum()


def infer_key(melody, params):
    major_profile = params["major_profile"]
    minor_profile = params["minor_profile"]
    cp_profile = _central_pitch_profile(params["central_pitch_mean"], params["central_pitch_var"])
    prob_major = params["prob_major"]

    best_key, best_log = None, -np.inf
    for tonic in range(12):
        for is_major, prior in [(True, prob_major / 12), (False, (1 - prob_major) / 12)]:
            kp = _key_profile(tonic, is_major, major_profile, minor_profile)
            log_k = -np.inf
            for c in range(128):
                cp_prob = cp_profile[c]
                if cp_prob < 1e-10:
                    continue
                rp_c = _range_profile(c, params["vr"])
                log_mel = 0.0
                for i, pitch in enumerate(melody):
                    if i == 0:
                        combined = rp_c * kp
                        combined /= combined.sum()
                    else:
                        prox = _proximity_profile(melody[i - 1], params["vp"])
                        combined = rp_c * prox * kp
                        combined /= combined.sum()
                    p = combined[pitch]
                    log_mel += np.log(p) if p > 0 else -1000
                log_joint = np.log(prior) + np.log(cp_prob) + log_mel
                log_k = np.logaddexp(log_k, log_joint)
            if log_k > best_log:
                best_log, best_key = log_k, (tonic, is_major)

    return best_key


def predict(midi_path, params, n=128):
    melodies = extract_melody_notes(midi_path, ["pitch"])
    if not melodies:
        raise ValueError(f"No monophonic melody found in {midi_path}")
    melody = [note["pitch"] for note in melodies[0]]

    prob_major = params["prob_major"]
    cp_profile = _central_pitch_profile(params["central_pitch_mean"], params["central_pitch_var"])

    # Restrict central pitch candidates to a plausible range around the melody
    lo = max(0, min(melody) - 20)
    hi = min(n, max(melody) + 21)
    c_list = list(range(lo, hi))

    keys = [(t, True, prob_major / 12) for t in range(12)] + \
           [(t, False, (1 - prob_major) / 12) for t in range(12)]

    kp_arr = np.stack([_key_profile(t, im, params["major_profile"], params["minor_profile"], n)
                       for t, im, _ in keys])
    rp_arr = np.stack([_range_profile(c, params["vr"], n) for c in c_list])

    log_prior_k = np.array([np.log(pk) for _, _, pk in keys])
    cp_vals = np.array([cp_profile[c] for c in c_list])
    log_prior_c = np.where(cp_vals > 1e-12, np.log(np.maximum(cp_vals, 1e-300)), -np.inf)
    log_joint = log_prior_k[:, np.newaxis] + log_prior_c[np.newaxis, :]  # (24, num_c)

    probs, ics, entropies = [], [], []

    for step, pitch in enumerate(melody):
        if step == 0:
            combined = rp_arr[np.newaxis] * kp_arr[:, np.newaxis]  # (24, num_c, n)
        else:
            prox = _proximity_profile(melody[step - 1], params["vp"], n)
            combined = rp_arr[np.newaxis] * prox * kp_arr[:, np.newaxis]
        combined /= combined.sum(axis=2, keepdims=True).clip(1e-300)

        w_max = log_joint[np.isfinite(log_joint)].max() if np.any(np.isfinite(log_joint)) else 0.0
        w = np.where(np.isfinite(log_joint), np.exp(log_joint - w_max), 0.0)  # (24, num_c)
        dist = (w[:, :, np.newaxis] * combined).sum(axis=(0, 1))               # (n,)
        s = dist.sum()
        if s > 0:
            dist /= s

        p_val = float(dist[pitch])
        ics.append(-np.log2(p_val) if p_val > 0 else np.inf)
        entropies.append(float(-(dist * np.log2(dist + 1e-12)).sum()))
        probs.append(dist.copy())

        # Update log-joint with log P(observed pitch | k, c, prev)
        log_p_pitch = np.log(np.maximum(combined[:, :, pitch], 1e-300))
        log_joint = np.where(np.isfinite(log_joint), log_joint + log_p_pitch, -np.inf)

    return {"probabilities": probs, "ics": ics, "entropies": entropies}
