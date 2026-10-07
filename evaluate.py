"""Fit of one or more models to the four behavioral datasets.

    python evaluate.py results/predictions/idyom results/predictions/transformer

For each dataset: the variance in participants' responses explained by each model (R2, with a 95%
bootstrap confidence interval) and, for every pair of models, the variance unique to each and common
to both, with Steiger's test of the difference between their correlations (Holm-corrected).
"""
import argparse
import itertools
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from evaluation import core

NAMES = {"cuddy_lunney": "Cuddy & Lunney (1995)", "schellenberg": "Schellenberg (1996)",
         "manzara": "Manzara et al. (1992)", "fogel": "Fogel et al. (2015)"}


def evaluate(folders, ci=True):
    models = {os.path.basename(os.path.normpath(f)): core.load_preds(f) for f in folders}
    out = {}
    for d in core.DATASETS:
        have = [m for m in models if d in models[m]]
        if not have:
            continue
        res = {"fit": {m: core.fit(models[m][d], d, ci=ci) for m in have}, "pairs": []}
        if len(have) > 1:
            tests = {(t["a"], t["b"]): t for t in core.compare_models([models[m][d] for m in have], have, d)}
            for a, b in itertools.combinations(have, 2):
                part = core.pair_partition(models[a][d], models[b][d], d, ci=ci)
                res["pairs"].append(dict(part, a=a, b=b, z=tests[(a, b)]["z"], p_holm=tests[(a, b)]["p_holm"]))
        out[d] = res
    return out


def show(out):
    f = lambda v: f"{v:.2f}"
    for d, res in out.items():
        print(f"\n{NAMES[d]}, {next(iter(res['fit'].values()))['n']} observations")
        for m, r in res["fit"].items():
            ci = f" [{f(r['ci'][0])}, {f(r['ci'][1])}]" if "ci" in r else ""
            print(f"  {m:<16} R2 = {f(r['r2'])}{ci}")
        for p in res["pairs"]:
            print(f"  {p['a']} and {p['b']}: unique to {p['a']} {f(p['unique_a'])}, unique to {p['b']} {f(p['unique_b'])}, "
                  f"common {f(p['shared'])}, total {f(p['total'])}; z = {f(p['z'])}, Holm p = {p['p_holm']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="one folder of predictions per model (<dataset>.json)")
    ap.add_argument("--no_ci", action="store_true", help="skip the bootstrap confidence intervals")
    ap.add_argument("--json", help="also write the results to this file")
    a = ap.parse_args()
    out = evaluate(a.folders, ci=not a.no_ci)
    show(out)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=float)
