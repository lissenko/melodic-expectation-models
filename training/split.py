"""The corpus partition, defined in one place: fit / selection / test.

Every model and every stage calls this, so the same music is used for the same
purpose everywhere:
  fit        training (networks, IDyOM long-term model, Temperley parameters)
  selection  every configuration choice (epochs, learning rate, IDyOM settings)
  test       every reported cross-entropy

The corpus carries no song identity, and a third of its melodies have near
duplicates elsewhere in it (other arrangements or excerpts of the same tune).
Splitting melody by melody put 15% of held-out melodies more than half inside
the training set as exact 12-note pitch sequences. Melodies are therefore
grouped into families first, and whole families are assigned to one side.

A family is a connected component of the graph linking two melodies when they
share at least LINK_FRAC of their 12-note interval sequences, counting only
sequences that occur in at most MAX_DF melodies (common figures such as scales
are ordinary musical regularity, not duplication). Intervals make the link
transposition-invariant.

Everything is decided by melody content, never by file order, so the partition
is reproducible from the released code and stable across machines.
"""
import collections
import hashlib
import os
import pickle

NGRAM = 12
MAX_DF = 50
LINK_FRAC = 0.25
SEL_N = 400
TEST_N = 1000
SPLIT_SEED = 24680
_KEY_NOTES = 16          # process_midi_folder already deduplicates on the first 10
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")


def melody_key(melody):
    return "|".join(f"{n['pitch']}:{n['duration']:.4f}" for n in melody[:_KEY_NOTES])


def _ngrams(melody):
    p = [n["pitch"] for n in melody]
    s = [b - a for a, b in zip(p, p[1:])]
    # hash of an int tuple is deterministic in CPython (PYTHONHASHSEED only affects str/bytes)
    return {hash(tuple(s[i:i + NGRAM])) for i in range(len(s) - NGRAM + 1)}


def _families(melodies):
    grams = [_ngrams(m) for m in melodies]
    df = collections.Counter(g for gs in grams for g in gs)
    postings = collections.defaultdict(list)
    for i, gs in enumerate(grams):
        for g in gs:
            if 1 < df[g] <= MAX_DF:
                postings[g].append(i)

    parent = list(range(len(melodies)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, gs in enumerate(grams):
        shared = collections.Counter(j for g in gs if g in postings for j in postings[g] if j != i)
        for j, k in shared.items():
            if k >= LINK_FRAC * len(gs):
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(melodies))]


def _assign(melodies):
    keys = [melody_key(m) for m in melodies]
    members = collections.defaultdict(list)
    for i, f in enumerate(_families(melodies)):
        members[f].append(i)
    order = sorted(members.values(), key=lambda ms: hashlib.md5(
        f"{SPLIT_SEED}|{min(keys[i] for i in ms)}".encode()).digest())

    role = {}
    for name, quota in (("selection", SEL_N), ("test", TEST_N)):
        taken = 0
        for ms in order:
            if taken == quota:
                break
            if keys[ms[0]] in role or len(ms) > quota - taken:
                continue
            for i in ms:
                role[keys[i]] = name
            taken += len(ms)
    return {k: role.get(k, "fit") for k in keys}


def _roles(melodies):
    keys = sorted(melody_key(m) for m in melodies)
    sig = repr((keys, NGRAM, MAX_DF, LINK_FRAC, SEL_N, TEST_N, SPLIT_SEED))
    path = os.path.join(CACHE_DIR, f"split_{hashlib.md5(sig.encode()).hexdigest()[:16]}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    roles = _assign(melodies)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(roles, f)
    return roles


def split_corpus(melodies):
    """Return (fit, selection, test). Depends only on melody content."""
    roles = _roles(melodies)
    out = {"fit": [], "selection": [], "test": []}
    for m in melodies:
        out[roles[melody_key(m)]].append(m)
    return out["fit"], out["selection"], out["test"]
