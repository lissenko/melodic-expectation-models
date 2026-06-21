import argparse
import json
import os

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import spearmanr, pearsonr, norm as _norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from data.human_ratings import (
    MANZARA_IC_SHORT,
    MANZARA_IC_LONG,
    load_cuddy_lunney,
    load_schellenberg,
    load_fogel,
)

_HERE = os.path.dirname(__file__)
PRED_DIR = os.path.join(_HERE, "data", "predictions")

def _load_pred(model: str, experiment: str) -> dict:
    path = os.path.join(PRED_DIR, model, f"{experiment}.json")
    with open(path) as f:
        return json.load(f)


def _regression(x, y):
    x, y = np.array(x), np.array(y)
    reg = LinearRegression().fit(x.reshape(-1, 1), y)
    r2 = r2_score(y, reg.predict(x.reshape(-1, 1)))
    n, k = len(y), 1
    f = (r2 / k) / ((1 - r2) / (n - k - 1)) if r2 < 1 else np.inf
    fp = 1 - stats.f.cdf(f, k, n - k - 1)
    rho, _ = spearmanr(x, y)
    return dict(n=n, beta=reg.coef_[0], r2=r2, F=f, F_p=fp, rho=rho)


_REG_HEADER = f"{'model':<13}{'beta':>9}{'rho':>8}{'R2':>8}{'F':>9}{'p':>11}{'n':>6}"


def _reg_row(name, m):
    print(f"{name:<13}{m['beta']:>9.4f}{m['rho']:>8.3f}{m['r2']:>8.3f}"
          f"{m['F']:>9.1f}{m['F_p']:>11.2e}{m['n']:>6}")
    return {"model": name, "beta": float(m["beta"]), "rho": float(m["rho"]),
            "R2": float(m["r2"]), "F": float(m["F"]), "p": float(m["F_p"]), "n": int(m["n"])}


def exp_manzara():
    print("Experiment 1: Manzara et al. (1992)")

    human_short = np.array(MANZARA_IC_SHORT)
    human_long  = np.array(MANZARA_IC_LONG)
    human = np.concatenate([human_short, human_long])

    print(f"{'model':<13}{'rho':>8}{'p':>11}{'R':>8}{'R2':>8}{'n':>6}")

    rows = []
    for model in ("temperley", "idyom", "lstm", "transformer"):
        pred = _load_pred(model, "manzara")
        ic_short = np.array(pred["short"]["ics"])
        ic_long  = np.array(pred["long"]["ics"])

        min_s = min(len(human_short), len(ic_short))
        min_l = min(len(human_long),  len(ic_long))

        model_ics = np.concatenate([ic_short[:min_s], ic_long[:min_l]])
        human_ics = np.concatenate([human_short[:min_s], human_long[:min_l]])

        rho, rho_p = spearmanr(model_ics, human_ics)
        r, r_p     = pearsonr(model_ics, human_ics)
        n = len(model_ics)
        r2 = r ** 2

        print(f"{model:<13}{rho:>8.3f}{rho_p:>11.2e}{r:>8.3f}{r2:>8.3f}{n:>6}")
        rows.append({"model": model, "rho": float(rho), "p": float(rho_p),
                     "R": float(r), "R2": float(r2), "n": int(n)})

    return {"rows": rows}


def exp_cuddy_lunney():
    print("Experiment 3: Cuddy & Lunney (1995)")

    human = load_cuddy_lunney()

    print(_REG_HEADER)

    rows = []
    for model in ("temperley", "idyom", "lstm", "transformer"):
        pred = _load_pred(model, "cuddy_lunney")

        all_ratings, all_ranks = [], []
        for context, ratings_dict in human.items():
            if context not in pred:
                continue
            rated_pitches = sorted(ratings_dict.keys())
            probs = np.array([pred[context]["last_note_probs"][p] for p in rated_pitches])
            probs /= probs.sum()
            ranks = stats.rankdata(probs)   # high prob -> high rank

            all_ranks.extend(ranks)
            all_ratings.extend([ratings_dict[p] for p in rated_pitches])

        m = _regression(all_ranks, all_ratings)
        rows.append(_reg_row(model, m))

    return {"rows": rows}


def exp_schellenberg():
    print("Experiment 4: Schellenberg (1996)")

    human = load_schellenberg()

    print(_REG_HEADER)

    rows = []
    for model in ("temperley", "idyom", "lstm", "transformer"):
        pred = _load_pred(model, "schellenberg")

        all_ratings, all_ranks = [], []
        for frag_id, ratings_dict in sorted(human.items()):
            if frag_id not in pred:
                continue
            rated_pitches = sorted(ratings_dict.keys())
            probs = np.array([pred[frag_id]["last_note_probs"][p] for p in rated_pitches])
            probs /= probs.sum()
            ranks = stats.rankdata(probs)

            all_ranks.extend(ranks)
            all_ratings.extend([ratings_dict[p] for p in rated_pitches])

        m = _regression(all_ranks, all_ratings)
        rows.append(_reg_row(model, m))

    return {"rows": rows}


# Experiment 5

_SD_CYCLE = ["sd1", "ook", "sd3", "ook", "sd5", "sd6", "ook",
             "sd8", "ook", "sd10", "ook", "sd12"]
_DIATONIC = ["sd1", "sd3", "sd5", "sd6", "sd8", "sd10", "sd12"]  # 7 positions

_SD_TO_DIATONIC = {
    sd: (_DIATONIC.index(_SD_CYCLE[sd - 1]) if _SD_CYCLE[sd - 1] != "ook" else -1)
    for sd in range(1, 13)
}


def _scale_degree_label(midi: int, tonic_pc: int) -> str:
    return _SD_CYCLE[(midi - tonic_pc) % 12]


def _sd_logprobs(pred_dict: dict, code: str, tonic_map: dict, eps: float = 1e-10) -> np.ndarray:
    tonic = tonic_map[code]
    p128 = pred_dict[code]["last_note_probs"]
    hist = np.zeros(7)
    for midi in range(47, 84):
        label = _scale_degree_label(midi, tonic)
        if label != "ook":
            hist[_DIATONIC.index(label)] += p128[midi]
    total = hist.sum()
    if total > 0:
        hist /= total
    return np.log(np.maximum(hist, eps))


def _fit_clogit(chosen: np.ndarray, x: np.ndarray):
    n, J, K = x.shape

    def neg_ll_grad(beta):
        u = x @ beta
        log_denom = logsumexp(u, axis=1)
        ll = (u[np.arange(n), chosen] - log_denom).sum()
        p = np.exp(u - log_denom[:, None])
        x_exp = (p[:, :, None] * x).sum(axis=1)
        grad = (x[np.arange(n), chosen] - x_exp).sum(axis=0)
        return -ll, -grad

    res = minimize(neg_ll_grad, np.zeros(K), jac=True, method="BFGS",
                   options={"gtol": 1e-8})
    beta = res.x

    u = x @ beta
    p = np.exp(u - logsumexp(u, axis=1)[:, None])
    x_exp = (p[:, :, None] * x).sum(axis=1)
    dx = x - x_exp[:, None, :]
    H = (p[:, :, None, None] * dx[:, :, :, None] * dx[:, :, None, :]).sum(axis=(0, 1))

    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pval = 2 * (1 - _norm.cdf(np.abs(z)))
    return beta, se, z, pval


def exp_fogel():
    print("Experiment 5: Fogel et al. (2015)")

    human_data, tonic_map = load_fogel()
    _models = ("temperley", "idyom", "lstm", "transformer")
    preds = {m: _load_pred(m, "fogel") for m in _models}

    def build_obs(nc_only=False):
        chosen, logp = [], {m: [] for m in _models}
        for row in human_data:
            code = row["code"]
            if nc_only and code[:2] != "NC":
                continue
            di = _SD_TO_DIATONIC.get(row["sd"], -1)
            if di < 0:
                continue
            if any(code not in preds[m] for m in _models):
                continue
            chosen.append(di)
            for m in _models:
                logp[m].append(_sd_logprobs(preds[m], code, tonic_map))
        return np.array(chosen), {m: np.array(v) for m, v in logp.items()}

    chosen_all, lp_all = build_obs(nc_only=False)
    chosen_nc,  lp_nc  = build_obs(nc_only=True)

    def _p_str(p):
        return "< .001" if p < 0.001 else f"= {p:.3f}"

    def _clogit_row(model, beta, se, z, pval):
        print(f"{_MODEL_LABELS[model]:<13}{beta:>7.2f}{se:>7.2f}{z:>8.2f}{_p_str(pval):>9}")
        return {"model": model, "beta": float(beta), "SE": float(se),
                "z": float(z), "p": float(pval)}

    def print_table(chosen, lp, title):
        n = len(chosen)
        print(f"\n{title} (n={n})")
        print(f"{'model':<13}{'beta':>7}{'SE':>7}{'z':>8}{'p':>9}")
        rows = []
        for m in _models:
            x = lp[m][:, :, None]
            beta, se, z, pval = _fit_clogit(chosen, x)
            rows.append(_clogit_row(m, beta[0], se[0], z[0], pval[0]))
        return {"n": n, "rows": rows}

    def print_combined(chosen, lp, title):
        n = len(chosen)
        print(f"\n{title} (n={n})")
        print(f"{'model':<13}{'beta':>7}{'SE':>7}{'z':>8}{'p':>9}")
        x = np.stack([lp[m] for m in _models], axis=2)
        betas, ses, zs, pvals = _fit_clogit(chosen, x)
        rows = [_clogit_row(m, betas[i], ses[i], zs[i], pvals[i])
                for i, m in enumerate(_models)]
        return {"n": n, "rows": rows}

    return {
        "individual_ac_nc": print_table(chosen_all, lp_all, "individual models, AC + NC"),
        "individual_nc":    print_table(chosen_nc, lp_nc, "individual models, NC only"),
        "combined_ac_nc":   print_combined(chosen_all, lp_all, "combined model, AC + NC"),
        "combined_nc":      print_combined(chosen_nc, lp_nc, "combined model, NC only"),
    }


# Inter-model correlation

_MODELS = ("temperley", "idyom", "lstm", "transformer")
_MODEL_LABELS = {
    "temperley": "Temperley",
    "idyom": "IDyOM",
    "lstm": "LSTM",
    "transformer": "Transformer",
}


def exp_model_correlation():
    print("Inter-model correlation (Bach chorales)")

    preds = {m: _load_pred(m, "western") for m in _MODELS}
    melodies = sorted(preds["lstm"].keys())

    pairs = [
        (a, b) for i, a in enumerate(_MODELS) for b in _MODELS[i + 1:]
    ]

    print(f"{'pair':<26}{'mean_rho':>9}   {'95% CI':<18}{'n':>6}")

    rows = []
    for a, b in pairs:
        rhos = []
        for mel in melodies:
            if mel not in preds[a] or mel not in preds[b]:
                continue
            ics_a = np.array(preds[a][mel]["ics"])
            ics_b = np.array(preds[b][mel]["ics"])
            min_len = min(len(ics_a), len(ics_b))
            if min_len < 3:
                continue
            rho, _ = spearmanr(ics_a[:min_len], ics_b[:min_len])
            rhos.append(rho)

        rhos = np.array(rhos)
        n = len(rhos)
        mean = rhos.mean()
        se = rhos.std(ddof=1) / np.sqrt(n)
        lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
        label = f"{_MODEL_LABELS[a]}-{_MODEL_LABELS[b]}"
        ci = f"[{lo:.3f}, {hi:.3f}]"
        print(f"{label:<26}{mean:>9.3f}   {ci:<18}{n:>6}")
        rows.append({"pair": label, "model_a": a, "model_b": b,
                     "mean_rho": float(mean), "ci_low": float(lo),
                     "ci_high": float(hi), "n": int(n)})

    return {"rows": rows}

EXPERIMENTS = {
    "manzara": exp_manzara,
    "cuddy_lunney": exp_cuddy_lunney,
    "schellenberg": exp_schellenberg,
    "fogel": exp_fogel,
    "model_correlation": exp_model_correlation,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce paper validation experiments")
    parser.add_argument(
        "--experiment", "-e",
        choices=list(EXPERIMENTS) + ["all"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--out", "-o",
        default="evaluation_results.json",
        help="JSON file to write results to (default: evaluation_results.json)",
    )
    args = parser.parse_args()

    if args.experiment == "all":
        to_run = list(EXPERIMENTS.items())
    else:
        to_run = [(args.experiment, EXPERIMENTS[args.experiment])]

    results = {}
    for i, (name, fn) in enumerate(to_run):
        if i:
            print()
        results[name] = fn()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nresults written to {args.out}")
