"""Reconstruction experiments: denoising + imputation, levels 2-3.

This is the script that produces every number in the paper's results table and in both
panels of the experiments figure. For each (dataset, level) it

  * standardises the signal, builds the interaction-order filtration and all candidate
    regularizers (Sec. "Filters and Signal Reconstruction");
  * runs denoising at sigma in {0.25, 0.5, 1.0} and imputation at 30/50/70% missing,
    20 trials each (10 when N_p > 1000);
  * selects (candidate, alpha, gamma) by SURE for denoising and by a 25% validation
    split inside the observed entries for imputation;
  * records per-trial NRMSE, the band-resolved errors at the headline setting, and the
    interaction-order energy profile with its permutation null.

Hyperparameter grids (these ARE the settings used for the published numbers):
    alpha  = logspace(-3, 3, 7)        gamma = logspace(-3, 1, 3)
    R_ord shapes: beta_k = k^4, cut_2, cut_3        (eq. 12)
    cohesion / polarization for every q < p; oriented Tikhonov over the six Hodge
    Laplacians (lower/upper/full x low/high-pass), plus 20 re-orientations of the
    full low-pass variant as the orientation-invariance check.
Every R is rescaled to unit spectral radius, so one alpha grid means the same thing
for every method.

TIE-BREAK METRIC, an extra hyperparameter for IMPUTATION only:
    Gamma = I   ->  gamma ||x||^2                           (eq. 13 as written)
    Gamma = G   ->  gamma x'Gx,  G = Q^T ((QQ^T)^+)^2 Q,  Q = Q_{0,p}
For x = Q^T c in V_1 the second gives ||c||^2 with the minimum-norm c, and G vanishes
on V_1^perp so alpha*R still does the band shrinkage. It is selected per trial on the
same 25% split as alpha and gamma, and the choice is recorded in the `tie` column.
Denoising keeps Gamma = I: with M = I there is no unidentified subspace for the
tie-break to resolve, and SURE's closed form needs the filter diagonal in R's
eigenbasis, which gamma*G breaks.

Writes results/recon_<dataset>.csv (per trial), recon_bands_<dataset>.csv and
energy_<dataset>.csv.

Run: python scripts/02_reconstruction.py <dataset> [<dataset> ...]
     with no argument it runs all four, sequentially.
Runtimes (single core, one dataset per process): NDC-substances ~1 min,
acm-coauth ~12 min, fp-landscape ~14 min, tags-math ~25 min.
"""
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
PROC = os.path.join(REPO, "data", "processed")
RES = os.path.join(REPO, "results")
os.makedirs(RES, exist_ok=True)

from src.complex import SimplicialComplex                    # noqa: E402
from src.bands import Filtration, permutation_null           # noqa: E402
import src.recon as rc                                       # noqa: E402
import src.regularizers as rg                                # noqa: E402

ALPHAS = np.logspace(-3, 3, 7)
GAMMAS = np.logspace(-3, 1, 3)
SIGMAS = (0.25, 0.5, 1.0)
MISS = (0.30, 0.50, 0.70)
TRIALS, TRIALS_LARGE, N_LARGE = 20, 10, 1000
LEVELS = {"fp-landscape": (2, 3), "tags-math": (2, 3),
          "NDC-substances": (2, 3), "acm-coauth": (2, 3)}   


def load(ds_id):
    z = np.load(os.path.join(PROC, f"{ds_id}.npz"), allow_pickle=True)
    md = int(z["maxdim"])
    simp = [[tuple(int(v) for v in s) for s in z[f"simplices_{k}"]] for k in range(md + 1)]
    cx = SimplicialComplex.__new__(SimplicialComplex)
    cx.maxdim = md
    cx.simplices = simp
    cx.index = [{s: i for i, s in enumerate(simp[k])} for k in range(md + 1)]
    cx.N = [len(simp[k]) for k in range(md + 1)]
    sig = {k: z[f"sig_{k}"] for k in range(1, md + 1) if f"sig_{k}" in z.files}
    return cx, sig, json.loads(str(z["meta_json"]))


def tie_metric(cx, p):
    """G = Q^T ((QQ^T)^+)^2 Q, unit spectral radius: the ||c||^2 tie-break on V_1."""
    Q = cx.Q(0, p)
    P = np.linalg.pinv(Q @ Q.T, rcond=1e-10)
    G = Q.T @ (P @ P) @ Q
    G = (G + G.T) / 2.0
    l = float(np.linalg.eigvalsh(G)[-1])
    return G / l if l > 1e-12 else G


def candidates(cx, p, filt, rng, n_orient=20):
    n = cx.N[p]
    g = {}
    for q in range(p):
        g[f"cohesion q={q}"] = {f"cohesion q={q}": rg.cohesion(cx, p, q)}
        g[f"polarization q={q}"] = {f"polarization q={q}": rg.polarization(cx, p, q)}
    g["ord-fixed"] = {"ord-fixed": rg.order_reg(filt, np.arange(p + 2, dtype=float))}
    nb = p + 2
    shapes = {"pow4.0": np.arange(nb, dtype=float) ** 4.0}
    for k0 in (2, 3):
        if k0 < nb:
            b = np.zeros(nb); b[k0:] = np.inf
            shapes[f"cut{k0}"] = b
    g["ord-tuned"] = {k: rg.order_reg(filt, b) for k, b in shapes.items()}
    g["ridge"] = {"ridge": np.eye(n)}
    for kind in ("lower", "upper", "full"):
        for hp in (False, True):
            R = rg.oriented(cx, p, kind, None, hp)
            if np.abs(R).max() == 0:
                continue
            g[f"oriented {kind} {'high' if hp else 'low'}"] = {kind: R}
    lam = rg.lam_max(cx.L_sg(p, "full"))
    reo = {f"orient{i}": rg.oriented(cx, p, "full", cx.random_signs(rng), False, lam=lam)
           for i in range(n_orient)}
    reo = {k: v for k, v in reo.items() if np.abs(v).max() > 0}
    if reo:
        g["oriented full low (reorient)"] = reo
    return g


def solve_tie(mask, y, R, alpha, gamma, G):
    n = len(y)
    A = alpha * np.asarray(R, float) + (gamma * G if G is not None else 0.0)
    A = (A + A.T) / 2.0
    if G is None:
        A[np.diag_indices(n)] += mask + max(float(gamma), 1e-10)
    else:
        A[np.diag_indices(n)] += mask + 1e-10
    b = mask * y
    try:
        return cho_solve(cho_factor(A, lower=True, check_finite=False), b,
                         check_finite=False)
    except Exception:
        return np.linalg.lstsq(A, b, rcond=1e-10)[0]


def sure_pick(cands, y, sigma, eig, key):
    n = len(y)
    best = (np.inf, None, None, None, None)
    for nm in cands:
        w, V = eig[(key, nm)]
        c = V.T @ y
        for a in ALPHAS:
            for gm in GAMMAS:
                gl = 1.0 / (1.0 + gm + a * w)
                s = (float(np.sum((c * (gl - 1.0)) ** 2)) - n * sigma ** 2
                     + 2.0 * sigma ** 2 * float(np.sum(gl)))
                if s < best[0]:
                    best = (s, nm, a, gm, V @ (c * gl))
    return best[4], best[1], float(best[2]), float(best[3])


def impute_pick(cands, y, obs, seed, G, val_frac=0.25):
    """Selects candidate, alpha, gamma AND the tie-break metric on a 25% split."""
    n = len(y)
    rng = np.random.default_rng(5000 + seed)
    perm = rng.permutation(obs)
    k = int(round(val_frac * len(obs)))
    val, tr = np.sort(perm[:k]), np.sort(perm[k:])
    m_tr = np.zeros(n); m_tr[tr] = 1.0
    best = (np.inf, None, ALPHAS[0], GAMMAS[0], "I")
    for nm, R in cands.items():
        for a in ALPHAS:
            for gm in GAMMAS:
                for tag, Gm in (("I", None), ("G", G)):
                    x = solve_tie(m_tr, y, R, a, gm, Gm)
                    e = float(np.mean((x[val] - y[val]) ** 2))
                    if e < best[0]:
                        best = (e, nm, a, gm, tag)
    m = np.zeros(n); m[obs] = 1.0
    _, nm, a, gm, tag = best
    return (solve_tie(m, y, cands[nm], a, gm, None if tag == "I" else G),
            nm, float(a), float(gm), tag)


def run(ds_id):
    cx, sig, meta = load(ds_id)
    rows, brows, erows = [], [], []
    for p in LEVELS[ds_id]:
        if p > cx.maxdim or cx.N[p] < 40 or p not in sig:
            continue
        s0 = np.asarray(sig[p], float)
        if not np.all(np.isfinite(s0)) or s0.std() == 0:
            print(f"[skip] {ds_id} p={p}: degenerate signal", flush=True)
            continue
        s = (s0 - s0.mean()) / s0.std()
        n = cx.N[p]
        nt = TRIALS_LARGE if n > N_LARGE else TRIALS
        filt = Filtration(cx, p)
        G = tie_metric(cx, p)
        groups = candidates(cx, p, filt, np.random.default_rng(11))
        ls_designs = {f"ls{k}": rg.order_ls_design(cx, p, k) for k in range(1, p + 1)}
        t0 = time.time()

        def record(task, setting, trial, method, pick, xh, score, a=np.nan, g=np.nan,
                   tie="I"):
            rows.append(dict(dataset=ds_id, level=p, task=task, setting=setting,
                             trial=trial, method=method, pick=pick, alpha=a, gamma=g,
                             tie=tie, N=n, nrmse=rc.rmse(xh[score], s[score])))
            if (task == "denoise" and setting == 0.5) or (task == "impute" and setting == 0.5):
                for k, e in enumerate(rc.band_errors(filt, xh, s)):
                    brows.append(dict(dataset=ds_id, level=p, task=task, trial=trial,
                                      method=method, band=k, e_k=e))

        eig = {}
        for gname, cands in groups.items():
            for nm, R in cands.items():
                if (gname, nm) in eig:
                    continue
                w, V = np.linalg.eigh((R + R.T) / 2.0)
                eig[(gname, nm)] = (np.clip(w, 0.0, None), V)
        for sigma in SIGMAS:
            for trial in range(nt):
                y, obs, _ = rc.make_trial(s, sigma, 0.0, trial)
                score = np.arange(n)
                sel_a = sel_g = None
                for gname, cands in groups.items():
                    if "reorient" in gname:
                        if sel_a is None:
                            continue
                        for nm in cands:
                            w, VV = eig[(gname, nm)]
                            record("denoise", sigma, trial, gname, nm,
                                   VV @ ((VV.T @ y) / (1.0 + sel_g + sel_a * w)),
                                   score, sel_a, sel_g)
                        continue
                    xh, pick, a, g = sure_pick(cands, y, sigma, eig, gname)
                    if gname == "oriented full low":
                        sel_a, sel_g = a, g
                    record("denoise", sigma, trial, gname, pick, xh, score, a, g)
                for lname, D in ls_designs.items():
                    record("denoise", sigma, trial, lname, lname,
                           rc.ls_fit(D, y, np.arange(n)), score)
                record("denoise", sigma, trial, "predict-mean", "predict-mean",
                       np.full(n, y.mean()), score)
            print(f"[{ds_id:15s} p={p} denoise s={sigma}] done", flush=True)
        del eig
        for miss in MISS:
            for trial in range(nt):
                y, obs, msk = rc.make_trial(s, 0.0, miss, trial)
                sel_a = sel_g = None
                for gname, cands in groups.items():
                    if "reorient" in gname:
                        if sel_a is None:
                            continue
                        m = np.zeros(n); m[obs] = 1.0
                        for nm, R in cands.items():
                            record("impute", miss, trial, gname, nm,
                                   solve_tie(m, y, R, sel_a, sel_g, None), msk,
                                   sel_a, sel_g)
                        continue
                    xh, pick, a, g, tie = impute_pick(cands, y, obs, trial, G)
                    if gname == "oriented full low":
                        sel_a, sel_g = a, g
                    record("impute", miss, trial, gname, pick, xh, msk, a, g, tie)
                for lname, D in ls_designs.items():
                    record("impute", miss, trial, lname, lname, rc.ls_fit(D, y, obs), msk)
                record("impute", miss, trial, "neighbour-mean", "neighbour-mean",
                       rc.neighbour_mean(cx, p, p - 1, y, obs), msk)
                record("impute", miss, trial, "predict-mean", "predict-mean",
                       np.full(n, y[obs].mean()), msk)
            print(f"[{ds_id:15s} p={p} impute m={miss}] done", flush=True)
        pi = np.asarray(filt.pi(s), float)
        dfr = np.asarray(filt.dim_frac(), float)
        null = permutation_null(s, filt, n=500)
        mu, sd = null.mean(0), null.std(0)
        for k in range(len(pi)):
            erows.append(dict(dataset=ds_id, level=p, k=k,
                              dim=int(round(dfr[k] * n)), dim_frac=dfr[k], pi=pi[k],
                              enrichment=(pi[k] / dfr[k]) if dfr[k] > 0 else np.nan,
                              z=((pi[k] - mu[k]) / sd[k]) if sd[k] > 0 else np.nan))
        print(f"[ok  ] {ds_id} p={p}  N={n}  {(time.time()-t0)/60:.1f} min", flush=True)
    for name, data in ((f"recon_{ds_id}.csv", rows), (f"recon_bands_{ds_id}.csv", brows),
                       (f"energy_{ds_id}.csv", erows)):
        if data:
            with open(os.path.join(RES, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader(); w.writerows(data)
    print(f"[done] {ds_id}", flush=True)


if __name__ == "__main__":
    for d in (sys.argv[1:] or list(LEVELS)):
        run(d)
