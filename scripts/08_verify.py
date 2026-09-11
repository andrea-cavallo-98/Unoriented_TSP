"""Re-derive every number the paper reports from results/*.csv and check it.

Run this after a re-run of the pipeline. It checks the 8 table rows (8 numbers each,
mean only -- the standard deviations are printed by 06_table_experiments.py) plus the
supporting claims made in the experiments section, and prints one PASS/FAIL line.

Tolerance is +-0.001 on the table, i.e. the last digit the paper prints. A FAIL is a
genuine discrepancy, not a rounding artefact: the pipeline is deterministic, and across
two runs on the same machine the per-trial NRMSE agreed to 2e-8.

Run: python scripts/08_verify.py
"""
import csv
import os
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(REPO, "results")

# ---------------------------------------------------------------- expected values
# Table 1, sigma = 0.5 / 50% missing:
#   (dataset, p) -> denoise (ridge, oriented, cohesion, order),
#                   impute  (neighbour-mean, oriented, cohesion, order)
TABLE = {
    ("fp-landscape", "2"): ((0.446, 0.446, 0.427, 0.298), (0.708, 0.880, 0.728, 0.279)),
    ("fp-landscape", "3"): ((0.445, 0.445, 0.411, 0.273), (0.678, 0.963, 0.699, 0.277)),
    ("acm-coauth", "2"):   ((0.446, 0.437, 0.441, 0.390), (0.642, 0.898, 0.822, 0.620)),
    ("acm-coauth", "3"):   ((0.444, 0.446, 0.440, 0.365), (0.675, 0.878, 0.785, 0.632)),
    ("tags-math", "2"):    ((0.447, 0.445, 0.443, 0.395), (0.899, 0.954, 0.877, 0.645)),
    ("tags-math", "3"):    ((0.446, 0.445, 0.442, 0.431), (0.950, 0.985, 0.898, 0.839)),
    ("NDC-substances", "2"): ((0.449, 0.448, 0.446, 0.423), (0.997, 0.981, 0.953, 0.906)),
    ("NDC-substances", "3"): ((0.446, 0.446, 0.443, 0.420), (1.012, 1.010, 0.958, 0.864)),
}
SIZES = {                      # N_0..N_3, then dim W_{p+1} at p=2 and p=3
    "fp-landscape":   ([13, 78, 286, 715], 208, 429),
    "tags-math":      ([25, 297, 1093, 846], 797, 64),
    "NDC-substances": ([51, 175, 204, 167], 75, 26),
    "acm-coauth":     ([542, 1064, 689, 290], 65, 5),
}
TIE_G = {"acm-coauth": 66, "NDC-substances": 38, "tags-math": 37, "fp-landscape": 7}
SIGNLESS = ("cohesion", "polarization", "ord-fixed", "ord-tuned")
TOL, TOL_PCT = 0.0015, 1.0
fails = []


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def rows(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        print(f"[FAIL] missing {name} -- run the pipeline first")
        sys.exit(1)
    return list(csv.DictReader(open(p)))


def check(label, got, want, tol=TOL):
    ok = got == got and abs(got - want) <= tol
    if not ok:
        fails.append(label)
    print(f"  {'ok ' if ok else 'FAIL'}  {label:52s} {got:9.3f}  (paper {want:.3f})")


# ------------------------------------------------------------------- dataset table
print("--- dataset sizes and top-band dimensions ---")
dt = {r["dataset"]: r for r in rows("dataset_table.csv")}
for ds, (N, w2, w3) in SIZES.items():
    r = dt.get(ds)
    if r is None:
        fails.append(f"{ds} missing from dataset_table"); print(f"  FAIL  {ds} missing")
        continue
    got = [int(r[f"N{k}"]) for k in range(4)]
    ok = got == N and int(r["p2_topdim"]) == w2 and int(r["p3_topdim"]) == w3
    if not ok:
        fails.append(f"{ds} sizes")
    print(f"  {'ok ' if ok else 'FAIL'}  {ds:16s} N={got} topdim={r['p2_topdim']}/"
          f"{r['p3_topdim']}  (paper N={N} topdim={w2}/{w3})")

# ------------------------------------------------------------------------- table 1
print("\n--- Table 1: NRMSE at sigma=0.5 / 50% missing ---")
H = {(r["dataset"], r["level"], r["task"]): r for r in rows("recon_headline.csv")}
for (ds, lv), (dn, im) in TABLE.items():
    for task, want, cols in (("denoise", dn, ("ridge", "oriented", "cohesion", "ord-tuned")),
                             ("impute", im, ("neighbour-mean", "oriented", "cohesion",
                                             "ord-tuned"))):
        r = H.get((ds, lv, task))
        if r is None:
            fails.append(f"{ds} p={lv} {task} missing")
            print(f"  FAIL  {ds} p={lv} {task} missing"); continue
        for col, w in zip(cols, want):
            check(f"{ds} p={lv} {task:7s} {col}", fl(r[col]), w)

# ------------------------------------------------- gains over the oriented estimator
print("\n--- gain of the order regularizer over oriented (paper: 3.2-38.6 / 7.7-71.2) ---")
for task, lo, hi in (("denoise", 3.2, 38.6), ("impute", 7.7, 71.2)):
    g = [100 * (fl(r["oriented"]) - fl(r["ord-tuned"])) / fl(r["oriented"])
         for r in H.values() if r["task"] == task and r["level"] in ("2", "3")]
    check(f"{task} min gain %", min(g), lo, TOL_PCT)
    check(f"{task} max gain %", max(g), hi, TOL_PCT)

# ------------------------------------------------------------------ energy profile
print("\n--- energy profile (paper: fp-landscape p=2, 74% of energy in 4% of dims) ---")
en = defaultdict(dict)
for r in rows("energy_profile.csv"):
    en[(r["dataset"], r["level"])][int(r["k"])] = r
e = en[("fp-landscape", "2")][1]
check("fp-landscape p=2  pi_1", fl(e["pi"]), 0.74, 0.005)
check("fp-landscape p=2  dim fraction of W_1", fl(e["dim_frac"]), 0.04, 0.005)

# --------------------------------------------------------------- pure top band (b)
print("\n--- pure top band after denoising (figure panel b) ---")
bd = {(r["dataset"], r["level"], r["task"], r["band"], r["family"]): fl(r["e_k"])
      for r in rows("recon_band_table.csv")}
for fam, want in (("ord-tuned", 0.51), ("ridge", 0.71), ("oriented", 0.76)):
    check(f"tags-math p=2  e_3  {fam}", bd[("tags-math", "2", "denoise", "3", fam)], want,
          0.005)
fp = [bd[("fp-landscape", "2", "denoise", "3", f)] for f in ("cohesion", "ridge",
                                                             "oriented")]
check("fp-landscape p=2  e_3  order", bd[("fp-landscape", "2", "denoise", "3",
                                          "ord-tuned")], 0.97, 0.005)
check("fp-landscape p=2  e_3  others, min", min(fp), 3.6, 0.05)
check("fp-landscape p=2  e_3  others, max", max(fp), 5.4, 0.05)

# ------------------------------------------------------- tie-break metric selection
print("\n--- Gamma = G selection, imputation, p in {2,3}, signless families ---")
for ds, want in TIE_G.items():
    rs = [r for r in rows(f"recon_{ds}.csv")
          if r["task"] == "impute" and r["level"] in ("2", "3")
          and any(r["method"] == m or r["method"].startswith(m + " ") for m in SIGNLESS)]
    pct = 100 * sum(1 for r in rs if r["tie"] == "G") / max(len(rs), 1)
    check(f"{ds} frac G %", pct, want, 0.5)

# ------------------------------------------------------------ acm permutation null
print("\n--- acm-coauth permutation null (paper: z = +7.5 at p=2, +17.5 at p=3) ---")
if os.path.exists(os.path.join(RES, "acm_null.csv")):
    z = {r["level"]: fl(r["z"]) for r in rows("acm_null.csv")}
    check("acm-coauth p=2 z", z.get("2", float("nan")), 7.5, 0.1)
    check("acm-coauth p=3 z", z.get("3", float("nan")), 17.5, 0.1)
else:
    print("  skipped: results/acm_null.csv not present (run scripts/04_acm_null.py)")

print("\n" + ("PASS: every reported number re-derived from results/" if not fails
              else f"FAIL: {len(fails)} discrepancies\n  " + "\n  ".join(fails)))
sys.exit(1 if fails else 0)
