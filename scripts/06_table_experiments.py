"""Emit the LaTeX body of the results table, straight from results/.

Columns per task block: the task's naive baseline (ridge for denoising, neighbour-mean
for imputation), oriented Tikhonov, cohesion, order -- the last two being the proposed
regularizers. Each entry is mean +- SD over trials of the family's best variant.

Paste the output between \midrule and ottomrule of the table in the paper.

Run: python scripts/06_table_experiments.py      # prints to stdout
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(REPO, "results")

FAM = {
    "ridge": ["ridge"],
    "neighbour-mean": ["neighbour-mean"],
    "oriented": [f"oriented {k} {h}" for k in ("lower", "upper", "full")
                 for h in ("low", "high")],
    "cohesion": [f"cohesion q={q}" for q in range(3)],
    "ord-tuned": ["ord-tuned"],
}
ROWS = [("fp-landscape", "2"), ("fp-landscape", "3"),
        ("acm-coauth", "2"), ("acm-coauth", "3"),
        ("tags-math", "2"), ("tags-math", "3"),
        ("NDC-substances", "2"), ("NDC-substances", "3")]
LABEL = {"fp-landscape": "fp-landscape", "acm-coauth": "acm-coauth",
         "tags-math": "tags-math", "NDC-substances": "NDC-subst."}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def load():
    per = defaultdict(lambda: defaultdict(dict))
    for ds in {d for d, _ in ROWS}:
        for r in csv.DictReader(open(os.path.join(RES, f"recon_{ds}.csv"))):
            if f(r["setting"]) != 0.5 or r["level"] not in ("2", "3"):
                continue
            per[(r["dataset"], r["level"], r["task"])][r["method"]][r["trial"]] = \
                f(r["nrmse"])
    return per


def cell(per, key, name):
    t = per[key]
    cands = [(st.mean(list(t[m].values())), m) for m in FAM[name] if m in t and t[m]]
    if not cands:
        return None
    mu, best = min(cands)
    v = list(t[best].values())
    return mu, (st.stdev(v) if len(v) > 1 else 0.0)


def fmt(mu, sd, bold):
    body = r"%.3f_{\pm%.3f}" % (mu, sd)
    body = body.replace("\\pm0.", "\\pm.")
    return (r"$\mathbf{%s}$" % body) if bold else ("$%s$" % body)


def main():
    per = load()
    out = []
    for d, lv in ROWS:
        vals, bolds = [], []
        for task, base in (("denoise", "ridge"), ("impute", "neighbour-mean")):
            block = [(nm, cell(per, (d, lv, task), nm))
                     for nm in (base, "oriented", "cohesion", "ord-tuned")]
            best = min(v[0] for _, v in block if v)
            for nm, v in block:
                vals.append(v)
                bolds.append(v is not None and abs(v[0] - best) < 1e-12)
        cells = [fmt(v[0], v[1], b) if v else "--" for v, b in zip(vals, bolds)]
        out.append(r"\texttt{%s} & $%s$ & %s" % (LABEL[d], lv,
                                                 " & ".join(cells[:4])) +
                   "\n" + " " * 22 + "& %s \\\\" % " & ".join(cells[4:]))
    print("\n".join(out))
    # which family wins each cell, for the caption / text
    print("\n%% wins:", file=sys.stderr)
    for d, lv in ROWS:
        for task, base in (("denoise", "ridge"), ("impute", "neighbour-mean")):
            block = {nm: cell(per, (d, lv, task), nm)
                     for nm in (base, "oriented", "cohesion", "ord-tuned")}
            w = min((v[0], k) for k, v in block.items() if v)
            print(f"%%   {d:16s} p={lv} {task:8s} {w[1]} {w[0]:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
