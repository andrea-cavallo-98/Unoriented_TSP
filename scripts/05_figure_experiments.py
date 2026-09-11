"""The experiments figure (Fig. 3 of the paper). Writes figures/fig_experiments.{pdf,png}.

(a) interaction-order profile at p=2 and p=3, energy against dimension share
(b) error in the pure top band after denoising, for the cells whose top band carries
    energy (acm-coauth's is empty: pi_top = 0.0005 and 0.0000, so e_top is not
    meaningful there and the paper says so in the caption)

Every number is read from results/, so the figure cannot drift from the tables.

Run: python scripts/05_figure_experiments.py      # instant
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(REPO, "results")
FIG = os.path.join(REPO, "figures")
os.makedirs(FIG, exist_ok=True)

BAND_C = ["#cde2fb", "#86b6ef", "#2a78d6", "#104281", "#0d366b"]
M_C = {"ord-tuned": "#2a78d6", "ridge": "#eb6834", "oriented": "#1baf7a"}
INK, INK2, MUTED, GRID, GHOST = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 6.4,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "axes.edgecolor": GHOST, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

DS = [("fp-landscape", "fp-land."), ("acm-coauth", "acm"),
      ("tags-math", "tags"), ("NDC-substances", "NDC-sub")]
CELLS = [(d, l) for d, _ in DS for l in ("2", "3")]
SHORT = dict(DS)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


en = {}
for r in csv.DictReader(open(os.path.join(RES, "energy_profile.csv"))):
    en.setdefault((r["dataset"], r["level"]), {})[int(r["k"])] = r
bd = {(r["dataset"], r["level"], r["task"], r["band"], r["family"]): f(r["e_k"])
      for r in csv.DictReader(open(os.path.join(RES, "recon_band_table.csv")))}

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(7.0, 1.85),
    gridspec_kw=dict(width_ratios=[1.95, 0.78], wspace=0.20))
fig.subplots_adjust(left=0.118, right=0.995, top=0.880, bottom=0.235)

# ------------------------------------------------------------------ (a) profile
H_PI, H_DIM, GAP = 0.38, 0.17, 0.045
LBL_MIN = 0.075                       # narrower segments are left unlabelled
ylab, ypos = [], []
for i, (d, l) in enumerate(CELLS):
    y = len(CELLS) - 1 - i
    ks = sorted(en[(d, l)])
    pi = np.array([f(en[(d, l)][k]["pi"]) for k in ks])
    dm = np.array([f(en[(d, l)][k]["dim_frac"]) for k in ks])
    pi, dm = pi / pi.sum(), dm / dm.sum()
    for arr, y0, h, al in ((pi, y + GAP, H_PI, 1.0),
                           (dm, y - GAP - H_DIM, H_DIM, 0.42)):
        left = 0.0
        for k in range(len(arr)):
            c = BAND_C[min(k, len(BAND_C) - 1)]
            axA.barh(y0 + h / 2, arr[k], left=left, height=h, align="center",
                     color=c, alpha=al, linewidth=0.6, edgecolor="white", zorder=3)
            if al == 1.0 and arr[k] > LBL_MIN:
                axA.text(left + arr[k] / 2, y0 + h / 2, "$\\mathcal{W}_%d$" % k,
                         ha="center", va="center", fontsize=5.8, zorder=5,
                         color="white" if k >= 2 else INK)
            left += arr[k]
    ypos.append(y)
    ylab.append("%s $p{=}%s$" % (SHORT[d], l))

ytop = len(CELLS) - 1
axA.text(1.015, ytop - GAP - H_DIM / 2, "dim.", va="center", ha="left",
         fontsize=6.4, color=MUTED, style="italic")
axA.set_yticks(ypos)
axA.set_yticklabels(ylab, fontsize=7.0, color=INK2)
axA.set_xlim(0, 1.085)
axA.set_ylim(-0.30, len(CELLS) - 0.24)
axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axA.set_xticklabels(["0", ".25", ".5", ".75", "1"])
axA.set_xlabel("share of energy $\\pi_k$  /  of dimensions", labelpad=2.0)
axA.set_title("(a) interaction-order profile", loc="left", pad=4.0, color=INK)
for s in ("top", "right", "left"):
    axA.spines[s].set_visible(False)
axA.spines["bottom"].set_bounds(0, 1.0)          # no stub past the last tick
axA.tick_params(axis="y", length=0)

# -------------------------------------------------- (b) error in the pure top band
BC = [(d, l) for d, l in CELLS
      if f(en[(d, l)][int(l) + 1]["pi"]) >= 0.01]          # ACM's top band is empty
w = 0.26
for j, m in enumerate(["ord-tuned", "ridge", "oriented"]):
    v = [bd.get((d, l, "denoise", str(int(l) + 1), m), np.nan) for d, l in BC]
    axB.bar(np.arange(len(BC)) + (j - 1) * w, v, width=w - 0.035, color=M_C[m],
            linewidth=0, zorder=3,
            label={"ord-tuned": "order (ours)"}.get(m, m))
axB.axhline(1.0, color=INK2, lw=0.7, ls=(0, (2.6, 1.8)), zorder=4)
axB.text(len(BC) - 0.55, 1.09, "no recovery", fontsize=6.2, color=INK2,
         ha="right", va="bottom")
axB.set_yscale("log")
axB.set_ylim(0.28, 16)
axB.set_yticks([0.5, 1, 2, 4, 8]); axB.set_yticklabels(["0.5", "1", "2", "4", "8"])
axB.set_xticks(range(len(BC)))
axB.set_xticklabels(["%s\n%s" % (SHORT[d].split('-')[0][:5], l) for d, l in BC],
                    fontsize=6.8, linespacing=1.25, color=INK2)
axB.set_xlabel("level $p$", labelpad=2.0)
axB.set_ylabel("$e_{p+1}$", labelpad=1.0)
axB.set_title("(b) pure top band", loc="left", pad=4.0, color=INK)
axB.grid(axis="y", color=GRID, lw=0.4, zorder=0); axB.set_axisbelow(True)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
axB.tick_params(axis="x", length=0)
axB.legend(loc="upper right", frameon=False, handlelength=0.8, handleheight=0.7,
           handletextpad=0.28, labelspacing=0.18, borderpad=0.0, labelcolor=INK2)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, "fig_experiments." + ext), dpi=400)
plt.close(fig)
print("saved figures/fig_experiments.pdf / .png  (2 panels, 7.0 x 1.85 in)")
