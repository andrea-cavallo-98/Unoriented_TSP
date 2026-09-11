"""Permutation null for the ACM keyword signal, at the configuration used in the paper.

Null: permute each author's TF-IDF vector independently over terms. Every author keeps
their exact value distribution; only which terms they share with whom is destroyed. A
real signal must beat this. The statistic compared against the null is the per-band
Wiener denoising gain implied by the interaction-order energy profile, which needs no
reconstruction run.

Reported in the paper as z = +7.5 (p=2), +17.5 (p=3).

Writes results/acm_null.csv.

Run: python scripts/04_acm_null.py        # ~1 min
"""
import csv
import math
import os
import sys
from collections import Counter

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
RES = os.path.join(REPO, "results")
from src.complex import SimplicialComplex          # noqa: E402
from src.bands import Filtration                   # noqa: E402

SIG = 0.5
FLOOR = SIG / math.sqrt(1 + SIG ** 2)
MAX_SIZE, BUDGET, N_NULL = 8, 800, 20


def oracle_gain(pi, dfr):
    s = np.asarray(pi, float); n = SIG ** 2 * np.asarray(dfr, float)
    ok = (s + n) > 0
    return 100 * (FLOOR - math.sqrt(float(np.sum(s[ok] * n[ok] / (s[ok] + n[ok]))))) / FLOOR


def main():
    import scipy.io as sio
    import scipy.sparse as sp
    m = sio.loadmat(os.path.join(REPO, "data", "raw", "acm", "ACM.mat"))
    PvsA, TvsP = m["PvsA"].tocsr(), m["TvsP"].tocsc()
    df = np.asarray((TvsP > 0).sum(axis=1)).ravel().astype(float)
    idf = np.log((TvsP.shape[1] + 1.0) / (df + 1.0))
    F = (sp.diags(idf) @ TvsP @ PvsA).tocsc()
    ind, ptr = PvsA.indices, PvsA.indptr
    hyps = [tuple(sorted(int(x) for x in ind[ptr[p]:ptr[p + 1]]))
            for p in range(PvsA.shape[0])]
    hyps = [h for h in hyps if 2 <= len(h) <= MAX_SIZE]
    cnt = Counter()
    for h in hyps:
        cnt.update(h)
    keep = {a for a, _ in cnt.most_common(BUDGET)}
    hs = [h for h in hyps if all(a in keep for a in h)]
    relab = {a: i for i, a in enumerate(sorted(keep))}
    cx = SimplicialComplex([tuple(sorted(relab[a] for a in h)) for h in hs], maxdim=3)
    V = F[:, np.array(sorted(keep), int)].toarray()
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)

    def mom(Vm, k):
        out = np.zeros(cx.N[k])
        for i, s in enumerate(cx.simplices[k]):
            pr = Vm[:, s[0]].copy()
            for a in s[1:]:
                pr *= Vm[:, a]
            out[i] = pr.sum()
        return out

    rng = np.random.default_rng(0)
    perms = [np.array([rng.permutation(V.shape[0]) for _ in range(V.shape[1])]).T
             for _ in range(N_NULL)]
    rows = []
    print(f"ACM v2 config: max_size={MAX_SIZE} budget={BUDGET} "
          f"N=({cx.N[0]},{cx.N[1]},{cx.N[2]},{cx.N[3]})\n")
    print(f"{'level':>6s}{'N_p':>7s}{'real':>9s}{'null':>9s}{'sd':>7s}{'z':>9s}  verdict")
    for p in (2, 3):
        if cx.N[p] < 60:
            continue
        f = Filtration(cx, p)

        def g(x):
            xs = (x - x.mean()) / x.std()
            return oracle_gain(np.asarray(f.pi(xs), float),
                               np.asarray(f.dim_frac(), float))
        real = g(mom(V, p))
        gs = []
        for P in perms:
            x = mom(np.take_along_axis(V, P, axis=0), p)
            if x.std() > 0:
                gs.append(g(x))
        mu, sd = float(np.mean(gs)), float(np.std(gs))
        z = (real - mu) / sd if sd > 0 else float('nan')
        verdict = "REAL" if z > 3 else ("artefact" if z < 0 else "inconclusive")
        print(f"{p:6d}{cx.N[p]:7d}{real:8.1f}%{mu:8.1f}%{sd:7.1f}{z:+9.1f}  {verdict}")
        rows.append(dict(dataset="acm-coauth", level=p, N_p=cx.N[p], real_gain=real,
                         null_gain_mean=mu, null_gain_sd=sd, z=z, n_null=len(gs),
                         verdict=verdict))
    with open(os.path.join(RES, "acm_null.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nwrote results/acm_null.csv")


if __name__ == "__main__":
    main()
