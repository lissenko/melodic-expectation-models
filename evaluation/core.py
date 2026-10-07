"""Fit to the four behavioral datasets, unique and common variance, and model comparisons.

Predictions use the IDyOM export layout for every model: per stimulus, "ics" of the context notes
and "last_note_probs" (128 values) after the context.
"""
import collections
import csv
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from data.human_ratings import MANZARA_IC_LONG, MANZARA_IC_SHORT, load_cuddy_lunney, load_schellenberg

DATASETS = ["cuddy_lunney", "schellenberg", "manzara", "fogel"]
DIATONIC = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4, 9: 5, 11: 6}   # semitones above the tonic -> degree 1..7
REGISTER = 12                                           # +-12 semitones around the last context note
RNG_SEED = 20261005


# ---------------------------------------------------------------- human data

def fogel_human():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "human", "fogel_human_data.csv"))))
    tonic = {r["code"]: int(r["tonic"]) for r in csv.DictReader(open(os.path.join(ROOT, "data", "human", "fogel_tonic_mapping.csv")))}
    counts = collections.defaultdict(lambda: np.zeros(7))
    by_subject = collections.defaultdict(dict)
    for r in rows:
        if r["sd"] in ("FALSE", ""):
            continue
        deg = DIATONIC.get(int(r["sd"]) - 1)
        if deg is None:
            continue
        counts[r["code"]][deg] += 1
        by_subject[r["code"]][r["subject"]] = deg
    return tonic, dict(counts), dict(by_subject)


def last_context_pitch():
    import pretty_midi
    out = {}
    for f in os.listdir(os.path.join(ROOT, "data", "stimuli", "fogel")):
        notes = sorted(pretty_midi.PrettyMIDI(os.path.join(ROOT, "data", "stimuli", "fogel", f)).instruments[0].notes,
                       key=lambda n: n.start)
        out[f[:-4]] = notes[-1].pitch
    return out


_HUMAN = {}


def human():
    if not _HUMAN:
        tonic, counts, subj = fogel_human()
        _HUMAN.update(cuddy_lunney=load_cuddy_lunney(), schellenberg=load_schellenberg(),
                      manzara={"short": MANZARA_IC_SHORT, "long": MANZARA_IC_LONG},
                      fogel_tonic=tonic, fogel_counts=counts, fogel_subjects=subj, fogel_last=last_context_pitch())
    return _HUMAN


# ---------------------------------------------------------------- predictors

def degree_probs(p128, tonic, last, window=REGISTER):
    h = np.zeros(7)
    lo, hi = (last - window, last + window) if window else (0, 127)
    for m in range(max(lo, 0), min(hi, 127) + 1):
        d = DIATONIC.get((m - tonic) % 12)
        if d is not None:
            h[d] += p128[m]
    return h / h.sum()


def observations(pred, dataset, window=REGISTER):
    """(x, y, cluster) with x oriented so that a better model correlates positively with y."""
    H = human()
    x, y, g = [], [], []
    if dataset in ("cuddy_lunney", "schellenberg"):
        for ctx, ratings in sorted(H[dataset].items()):
            if ctx not in pred:
                continue
            for p, rating in sorted(ratings.items()):
                x.append(float(np.log2(max(pred[ctx]["last_note_probs"][p], 1e-300))))   # -IC
                y.append(rating); g.append(ctx)
    elif dataset == "manzara":
        for key in ("short", "long"):
            ics, hum = pred[key]["ics"], H["manzara"][key]
            n = min(len(ics), len(hum))
            x += list(ics[:n]); y += list(hum[:n]); g += [f"{key}:{i}" for i in range(n)]
    elif dataset == "fogel":
        for code, c in sorted(H["fogel_counts"].items()):
            if code not in pred:
                continue
            dp = degree_probs(pred[code]["last_note_probs"], H["fogel_tonic"][code], H["fogel_last"][code], window)
            x += list(dp); y += list(c / c.sum()); g += [code] * 7
    return np.array(x, float), np.array(y, float), np.array(g)


def fogel_contexts(prefix=None):
    return [c for c in sorted(human()["fogel_counts"]) if prefix is None or c.startswith(prefix)]


# ---------------------------------------------------------------- statistics

def r2_multi(X, y):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c) for c in X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))


def commonality(Xs, y):
    """All 2^k - 1 commonality components for k predictors, solved exactly from the subset R2s."""
    k = len(Xs)
    subsets = [s for r in range(1, k + 1) for s in itertools.combinations(range(k), r)]
    r2 = {s: r2_multi([Xs[i] for i in s], y) for s in subsets}
    A = np.array([[1.0 if set(S) & set(Y) else 0.0 for S in subsets] for Y in subsets])
    comp = np.linalg.solve(A, np.array([r2[s] for s in subsets]))
    return dict(zip(subsets, comp)), r2[tuple(range(k))]


def resample_index(g, rng, by_cluster=True):
    if not by_cluster:
        return rng.integers(0, len(g), len(g))
    clusters = np.unique(g)
    pick = rng.choice(clusters, len(clusters), replace=True)
    return np.concatenate([np.flatnonzero(g == c) for c in pick])


def by_cluster(dataset):
    return dataset != "manzara"          # two chorales: resample notes, stated in the Method


def boot_ci(stat, x_list, y, g, dataset, n=2000):
    rng = np.random.default_rng(RNG_SEED)
    vals = []
    for _ in range(n):
        idx = resample_index(g, rng, by_cluster(dataset))
        try:
            vals.append(stat([x[idx] for x in x_list], y[idx]))
        except np.linalg.LinAlgError:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fit(pred, dataset, window=REGISTER, ci=True):
    x, y, g = observations(pred, dataset, window)
    r2 = r2_multi([x], y)
    out = {"r2": float(r2), "n": int(len(y)), "r": float(np.corrcoef(x, y)[0, 1])}
    if ci:
        out["ci"] = boot_ci(lambda X, yy: r2_multi(X, yy), [x], y, g, dataset)
    return out


def pair_partition(pred_a, pred_b, dataset, ci=True):
    xa, y, g = observations(pred_a, dataset)
    xb, _, _ = observations(pred_b, dataset)
    comp, total = commonality([xa, xb], y)
    out = {"unique_a": float(comp[(0,)]), "unique_b": float(comp[(1,)]), "shared": float(comp[(0, 1)]),
           "total": float(total)}
    if ci:
        out["unique_a_ci"] = boot_ci(lambda X, yy: commonality(X, yy)[0][(0,)], [xa, xb], y, g, dataset)
        out["unique_b_ci"] = boot_ci(lambda X, yy: commonality(X, yy)[0][(1,)], [xa, xb], y, g, dataset)
    return out


def dependent_corr(ry1, ry2, r12, n):
    """Steiger's (1980) z and Zou's (2007) 95% CI for ry1 - ry2, overlapping dependent correlations."""
    rbar = (ry1 + ry2) / 2
    c = (r12 * (1 - 2 * rbar ** 2) - 0.5 * rbar ** 2 * (1 - 2 * rbar ** 2 - r12 ** 2)) / (1 - rbar ** 2) ** 2
    z = (np.arctanh(ry1) - np.arctanh(ry2)) * np.sqrt((n - 3) / (2 - 2 * c))
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    def zci(r):
        h = 1.96 / np.sqrt(n - 3)
        return np.tanh(np.arctanh(r) - h), np.tanh(np.arctanh(r) + h)
    l1, u1 = zci(ry1); l2, u2 = zci(ry2)
    cc = ((r12 - 0.5 * ry1 * ry2) * (1 - ry1 ** 2 - ry2 ** 2 - r12 ** 2) + r12 ** 3) / ((1 - ry1 ** 2) * (1 - ry2 ** 2))
    lo = ry1 - ry2 - np.sqrt((ry1 - l1) ** 2 + (u2 - ry2) ** 2 - 2 * cc * (ry1 - l1) * (u2 - ry2))
    hi = ry1 - ry2 + np.sqrt((u1 - ry1) ** 2 + (ry2 - l2) ** 2 - 2 * cc * (u1 - ry1) * (ry2 - l2))
    return float(z), float(p), (float(lo), float(hi))


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def compare_models(preds, names, dataset):
    obs = {n: observations(p, dataset) for n, p in zip(names, preds)}
    y = next(iter(obs.values()))[1]
    out = []
    for a, b in itertools.combinations(names, 2):
        xa, xb = obs[a][0], obs[b][0]
        ra, rb, rab = np.corrcoef(xa, y)[0, 1], np.corrcoef(xb, y)[0, 1], np.corrcoef(xa, xb)[0, 1]
        z, p, ci = dependent_corr(ra, rb, rab, len(y))
        out.append({"a": a, "b": b, "ra": float(ra), "rb": float(rb), "z": z, "p": p, "ci": ci})
    for row, pa in zip(out, holm(np.array([r["p"] for r in out]))):
        row["p_holm"] = float(pa)
    return out


def permutation_p(pred, dataset, n=1000):
    x, y, g = observations(pred, dataset)
    obs_r2 = r2_multi([x], y)
    rng = np.random.default_rng(RNG_SEED)
    groups = [np.flatnonzero(g == c) for c in np.unique(g)] if dataset != "manzara" else \
             [np.flatnonzero(np.char.startswith(g.astype(str), k)) for k in ("short", "long")]
    count = 0
    for _ in range(n):
        xp = x.copy()
        for idx in groups:
            xp[idx] = x[rng.permutation(idx)]
        count += r2_multi([xp], y) >= obs_r2
    return (count + 1) / (n + 1)


def load_preds(folder):
    """Predictions of one model: <folder>/<dataset>.json, for the datasets present."""
    return {d: json.load(open(os.path.join(folder, f"{d}.json"))) for d in DATASETS
            if os.path.exists(os.path.join(folder, f"{d}.json"))}
