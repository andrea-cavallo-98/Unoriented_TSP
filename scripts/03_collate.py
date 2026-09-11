"""Collate results/recon_*.csv into the tables the paper needs.

Writes results/recon_headline.csv (the sigma=0.5 / 50%-missing setting reported in the
table), recon_all_settings.csv (all nine settings), recon_band_table.csv (band-resolved
errors, panel (b) of the figure), energy_profile.csv (panel (a)) and tie_table.csv.

A "family" is reported at its best variant per cell: cohesion/polarization over q,
oriented over {lower, upper, full} x {low, high}, ord-tuned over its beta shapes. The
variant index is not a fitted hyperparameter for cohesion and oriented, which is
generous to those baselines.

Run: python scripts/03_collate.py
"""
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(REPO, "results")
DATASETS = ["fp-landscape", "tags-math", "NDC-substances", "acm-coauth"]
FAMILIES = {
    "ridge": ["ridge"],
    "predict-mean": ["predict-mean"],
    "neighbour-mean": ["neighbour-mean"],
    "ls1": ["ls1"], "ls2": ["ls2"], "ls3": ["ls3"],
    "cohesion": [f"cohesion q={q}" for q in range(3)],
    "polarization": [f"polarization q={q}" for q in range(3)],
    "ord-fixed": ["ord-fixed"],
    "ord-tuned": ["ord-tuned"],
    "oriented": [f"oriented {k} {h}" for k in ("lower", "upper", "full")
                 for h in ("low", "high")],
    "oriented-reorient": ["oriented full low (reorient)"],
}
SIGNLESS = ["cohesion", "polarization", "ord-fixed", "ord-tuned"]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def main():
    raw = defaultdict(lambda: defaultdict(dict))     # (ds,lv,task,setting) -> meth -> {trial: v}
    ties = defaultdict(lambda: defaultdict(int))
    for ds in DATASETS:
        p = os.path.join(RES, f"recon_{ds}.csv")
        if not os.path.exists(p):
            print(f"[warn] missing {p}")
            continue
        for r in csv.DictReader(open(p)):
            k = (r["dataset"], r["level"], r["task"], f(r["setting"]))
            raw[k][r["method"]][r["trial"]] = f(r["nrmse"])
            if r["task"] == "impute":
                ties[(r["dataset"], r["level"], r["method"])][r.get("tie", "I")] += 1

    def fam(k, name):
        t = raw[k]
        vals = [(st.mean(t[m].values()), m) for m in FAMILIES[name] if m in t and t[m]]
        return min(vals) if vals else (float('nan'), None)

    head, allset = [], []
    for k in sorted(raw, key=lambda k: (k[0], k[1], k[2], k[3])):
        ds, lv, task, setting = k
        row = dict(dataset=ds, level=lv, task=task, setting=setting,
                   N=int(f(next(iter(raw[k].values())) and 0) or 0) or "")
        for name in FAMILIES:
            v, pick = fam(k, name)
            row[name] = round(v, 4) if v == v else ""
            row[name + "_pick"] = pick or ""
        sl = min((f(row[m]) for m in SIGNLESS if row[m] != ""), default=float('nan'))
        row["signless_best"] = round(sl, 4) if sl == sl else ""
        floor = (setting / math.sqrt(1 + setting ** 2)) if task == "denoise" else 1.0
        row["floor"] = round(floor, 4)
        best = min((f(row[m]) for m in FAMILIES if row[m] != ""), default=float('nan'))
        row["best"] = round(best, 4) if best == best else ""
        row["best_vs_floor_pct"] = round(100 * (floor - best) / floor, 2) if best == best else ""
        orr = f(row["oriented"]) if row["oriented"] != "" else float('nan')
        row["signless_vs_oriented_pct"] = (round(100 * (orr - sl) / orr, 2)
                                           if orr == orr and sl == sl else "")
        allset.append(row)
        if (task == "denoise" and setting == 0.5) or (task == "impute" and setting == 0.5):
            head.append(row)

    def dump(name, rows):
        if not rows:
            return
        keys = sorted({kk for r in rows for kk in r})
        keys = ["dataset", "level", "task", "setting"] + [k for k in keys
                                                          if k not in ("dataset", "level", "task", "setting")]
        with open(os.path.join(RES, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        print(f"wrote results/{name} ({len(rows)} rows)")

    dump("recon_headline.csv", head)
    dump("recon_all_settings.csv", allset)

    # band table at the headline setting
    brows = []
    for ds in DATASETS:
        p = os.path.join(RES, f"recon_bands_{ds}.csv")
        if not os.path.exists(p):
            continue
        acc = defaultdict(list)
        for r in csv.DictReader(open(p)):
            for name, members in FAMILIES.items():
                if r["method"] in members:
                    acc[(r["dataset"], r["level"], r["task"], r["band"], name)].append(f(r["e_k"]))
        for (d, lv, task, band, name), v in sorted(acc.items()):
            v = [x for x in v if x == x]
            if v:
                brows.append(dict(dataset=d, level=lv, task=task, band=band,
                                  family=name, e_k=round(st.mean(v), 4), n=len(v)))
    dump("recon_band_table.csv", brows)

    # energy profiles
    erows = []
    for ds in DATASETS:
        p = os.path.join(RES, f"energy_{ds}.csv")
        if os.path.exists(p):
            erows += list(csv.DictReader(open(p)))
    dump("energy_profile.csv", erows)

    # tie-break usage
    trows = []
    for (ds, lv, m), c in sorted(ties.items()):
        tot = sum(c.values())
        trows.append(dict(dataset=ds, level=lv, method=m, n=tot,
                          n_I=c.get("I", 0), n_G=c.get("G", 0),
                          frac_G=round(c.get("G", 0) / tot, 3) if tot else ""))
    dump("tie_table.csv", trows)


if __name__ == "__main__":
    sys.exit(main())
