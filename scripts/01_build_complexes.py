"""Build the four simplicial complexes and their signals into data/processed/.

Selection rule, applied identically to every event dataset and declared in the paper:
  1. max_size = the smallest hyperedge-size cap for which the pure top band at p=3 is
     non-degenerate (dim W_4 > 0).
  2. node budget = the one maximising N_2/N_0 subject to N_2 in [200, 1200], i.e. the
     densest complex that is still solvable at 3 settings x 20 trials.
The resulting (max_size, budget) pairs are hard-coded below; they are the outcome of a
screening sweep that is not needed to reproduce the paper.

Datasets
  fp-landscape    13-site GFP landscape (Poelwijk et al. 2019). Every subset of the 13
                  mutated sites was assayed, so the complex is the FULL simplex and no
                  cap or budget applies. Signal = measured brightness of each mutation
                  combination (mean of the two reported trials); the two trials are the
                  replicate pair used for the split-half reliability.
  tags-math       Benson et al. 2018 tags-math-sx. Signal = exact-k event counts, log1p.
  NDC-substances  Benson et al. 2018 NDC-substances. Signal = exact-k counts, log1p.
  acm-coauth      ACM co-authorship (PvsA) with a TF-IDF k-way term co-moment signal
                  (TvsP) -- structure and signal come from independent matrices.

"Exact-k" means the level-k signal counts events whose participant set is EXACTLY that
k-simplex; a simplex present only as a face of a larger observed set gets a real zero,
so nothing at level k is ever computed from level < k.

Writes data/processed/<dataset>.npz and results/dataset_table.csv.

Run: python scripts/01_build_complexes.py
"""
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
RAW = os.path.join(REPO, "data", "raw")
OUTD = os.path.join(REPO, "data", "processed")
RES = os.path.join(REPO, "results")
for d in (OUTD, RES):
    os.makedirs(d, exist_ok=True)

from src.complex import SimplicialComplex, read_benson, restrict, top_nodes  # noqa: E402
from src.bands import Filtration                                             # noqa: E402

CONFIG = {                      # dataset -> (stem, max_size, budget)
    "tags-math":      ("tags-math-sx", 6, 25),
    "NDC-substances": ("NDC-substances", 6, 55),
}
ACM = dict(max_size=8, budget=800, restrict="whole")
ACM_MAT = os.path.join(RAW, "acm", "ACM.mat")
FP_XLSX = os.path.join(RAW, "fp_MOESM8_ESM.xlsx")
MAXDIM = 3


def half_corr(xa, xb):
    """Split-half reliability of a count signal: correlation of the two log1p halves
    over the simplices active in at least one of them."""
    m = (np.asarray(xa) > 0) | (np.asarray(xb) > 0)
    if m.sum() < 10:
        return float('nan')
    a, b = np.log1p(np.asarray(xa)[m]), np.log1p(np.asarray(xb)[m])
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def build_benson(name, stem, max_size, budget):
    hyps, times = read_benson(os.path.join(RAW, stem), stem, max_size=max_size)
    sizes = set(range(2, max_size + 1))
    keep = top_nodes(hyps, sizes, budget)
    hs, ts = restrict(hyps, times, keep, sizes)
    cx = SimplicialComplex(hs, maxdim=MAXDIM)
    sig, halves = {}, {}
    for k in range(1, MAXDIM + 1):
        sig[k] = np.log1p(cx.exact_counts(hs, k))
    if ts.size and ts.max() > ts.min():
        cut = np.median(ts)
        A = [h for h, t in zip(hs, ts) if t <= cut]
        B = [h for h, t in zip(hs, ts) if t > cut]
        for k in range(1, MAXDIM + 1):
            halves[k] = (cx.exact_counts(A, k), cx.exact_counts(B, k))
    meta = dict(source=f"Benson et al. 2018, {stem}", signal="exact-k event counts",
                transform="log1p", max_size=max_size, budget=budget)
    return cx, sig, halves, meta


def build_acm():
    """Structure from PvsA (paper x author), signal from TvsP (term x paper).

    A paper with k+1 authors contributes the k-simplex of its author set. The level-k
    signal on a simplex is the k-way co-moment sum_t prod_{i in sigma} vhat_i[t] of the
    unit-normalised TF-IDF term vectors of its authors, i.e. how much vocabulary the
    whole group shares. The two matrices are independent, so the complex is not built
    from the signal.
    """
    import scipy.io as sio
    import scipy.sparse as sp
    from collections import Counter
    m = sio.loadmat(ACM_MAT)
    PvsA, TvsP = m["PvsA"].tocsr(), m["TvsP"].tocsc()
    df = np.asarray((TvsP > 0).sum(axis=1)).ravel().astype(float)
    idf = np.log((TvsP.shape[1] + 1.0) / (df + 1.0))
    F = (sp.diags(idf) @ TvsP @ PvsA).tocsc()
    ind, ptr = PvsA.indices, PvsA.indptr
    hyps = [tuple(sorted(int(x) for x in ind[ptr[p]:ptr[p + 1]]))
            for p in range(PvsA.shape[0])]
    hyps = [h for h in hyps if 2 <= len(h) <= ACM["max_size"]]
    cnt = Counter()
    for h in hyps:
        cnt.update(h)
    keep = {a for a, _ in cnt.most_common(ACM["budget"])}
    hs = [h for h in hyps if all(a in keep for a in h)]
    relab = {a: i for i, a in enumerate(sorted(keep))}
    hs2 = [tuple(sorted(relab[a] for a in h)) for h in hs]
    cx = SimplicialComplex(hs2, maxdim=MAXDIM)
    cols = np.array(sorted(keep), int)
    V = F[:, cols].toarray()
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)

    def mom(Vm, k):
        out = np.zeros(cx.N[k])
        for i, s in enumerate(cx.simplices[k]):
            pr = Vm[:, s[0]].copy()
            for a in s[1:]:
                pr *= Vm[:, a]
            out[i] = pr.sum()
        return out

    sig = {k: mom(V, k) for k in range(1, MAXDIM + 1)}
    rng = np.random.default_rng(0)
    tp = rng.permutation(V.shape[0]); th = V.shape[0] // 2
    halves = {k: (mom(V[tp[:th]], k), mom(V[tp[th:]], k)) for k in range(1, MAXDIM + 1)}
    meta = dict(source="ACM.mat co-authorship (PvsA) + TF-IDF terms (TvsP)",
                signal="k-way TF-IDF co-moment sum_t prod_i vhat_i[t]",
                transform="raw", **ACM)
    return cx, sig, halves, meta


def build_fp():
    """Poelwijk et al. 2019: 13-site GFP landscape.

    A genotype carrying the mutation set S of size k is the (k-1)-simplex S; the signal
    is that genotype's measured brightness. Every subset was assayed, so the complex is
    the full simplex on 13 vertices truncated at MAXDIM. Sheet `genodata` of the
    supplementary workbook: column 0 is the 13-character 0/1 genotype string, columns 2
    and 4 are the brightness of the two independent trials.
    """
    import openpyxl
    wb = openpyxl.load_workbook(FP_XLSX, read_only=True, data_only=True)
    ws = wb["genodata"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    gen, t1, t2 = [], [], []
    for r in rows[2:]:                                  # two header rows
        g = str(r[0]).strip().strip("'")
        if len(g) != 13 or set(g) - {"0", "1"}:
            continue
        try:
            a, b = float(r[2]), float(r[4])
        except (TypeError, ValueError):
            continue
        gen.append(g); t1.append(a); t2.append(b)
    gen = np.array(gen); t1 = np.array(t1); t2 = np.array(t2)
    bright = 0.5 * (t1 + t2)
    cx = SimplicialComplex([tuple(range(13))], maxdim=MAXDIM)
    sig, halves = {}, {}
    for k in range(1, MAXDIM + 1):
        m, m1, m2 = {}, {}, {}
        for g, v, v1, v2 in zip(gen, bright, t1, t2):
            S = tuple(i for i, c in enumerate(g) if c == "1")
            if len(S) == k + 1:
                m[S] = v; m1[S] = v1; m2[S] = v2
        sig[k] = cx.value_signal(m, k, default=np.nan)
        halves[k] = (cx.value_signal(m1, k, np.nan), cx.value_signal(m2, k, np.nan))
    meta = dict(source="Poelwijk, Socolich & Ranganathan, Nat. Commun. 10:4213 (2019), "
                       "Supplementary Data (41467_2019_12130_MOESM8)",
                signal="measured brightness", transform="raw",
                n_genotypes=int(len(gen)), full_simplex=True, n_sites=13,
                replicates="trial1 / trial2", max_size=None, budget=None)
    return cx, sig, halves, meta


def fp_half_corr(xa, xb):
    """fp-landscape replicates are measurements, not counts: plain correlation of the
    two trials over the genotypes where both are finite."""
    m = np.isfinite(xa) & np.isfinite(xb)
    if m.sum() < 10:
        return float('nan')
    return float(np.corrcoef(np.asarray(xa)[m], np.asarray(xb)[m])[0, 1])


def main():
    table = []
    jobs = [(k, lambda k=k, v=v: build_benson(k, *v)) for k, v in CONFIG.items()]
    jobs.append(("acm-coauth", build_acm))
    jobs.append(("fp-landscape", build_fp))
    for ds_id, fn in jobs:
        cx, sig, halves, meta = fn()
        d = {"maxdim": cx.maxdim}
        for k in range(cx.maxdim + 1):
            d[f"simplices_{k}"] = (np.array(cx.simplices[k], dtype=np.int64)
                                   if cx.N[k] else np.zeros((0, k + 1), np.int64))
            d[f"N_{k}"] = cx.N[k]
        for k, v in sig.items():
            d[f"sig_{k}"] = v
        for k, (a, b) in halves.items():
            d[f"half1_{k}"], d[f"half2_{k}"] = a, b
        d["meta_json"] = json.dumps(meta, default=str)
        np.savez_compressed(os.path.join(OUTD, f"{ds_id}.npz"), **d)
        row = dict(dataset=ds_id, **{f"N{k}": cx.N[k] for k in range(4)},
                   max_size=meta.get("max_size") or "-",
                   budget=meta.get("budget") or "-",
                   signal=meta["signal"], transform=meta["transform"])
        rel = fp_half_corr if ds_id == "fp-landscape" else half_corr
        for p in (1, 2, 3):
            if p > cx.maxdim or cx.N[p] < 40:
                continue
            f = Filtration(cx, p)
            dims = [int(round(v * cx.N[p])) for v in f.dim_frac()]
            row[f"p{p}_dimW"] = str(dims)
            row[f"p{p}_topdim"] = dims[-1]
            if p in halves:
                row[f"p{p}_half"] = round(rel(*halves[p]), 3)
        table.append(row)
        print(f"[ok  ] {ds_id:16s} N={[cx.N[k] for k in range(4)]}  "
              f"topdim p2/p3 = {row.get('p2_topdim')}/{row.get('p3_topdim')}", flush=True)
    keys = sorted({k for r in table for k in r})
    with open(os.path.join(RES, "dataset_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(table)
    print("\nwrote results/dataset_table.csv")


if __name__ == "__main__":
    main()
