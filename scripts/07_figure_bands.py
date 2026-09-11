"""Figure 2 of the paper: a triangle signal and its interaction-order decomposition.

The complex is the octahedron (6 nodes, 12 edges, 8 triangles), the smallest complex
whose four bands are all non-trivial: dim(W_0,...,W_3) = (1, 3, 3, 1). It is drawn in
its planar embedding, in which seven of the eight triangles tile the outer triangle and
the eighth (the outer face 0-1-2) is the surrounding band.

The signal is built as a node-additive part (a strength per vertex), a pair bonus on one
edge, and a genuine three-way term (+c on one trio, -c on a neighbouring trio). The
three panels on the right are its order-<=1, order-2 and order-3 components; they sum
back to the signal exactly.

Illustrative only: no dataset is involved.

Writes figures/fig_bands_illustration.{pdf,png}.

Run: python scripts/07_figure_bands.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Polygon, Circle

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from src.complex import SimplicialComplex
from src.bands import Filtration

FIG = os.path.join(REPO, "figures")
os.makedirs(FIG, exist_ok=True)

TRIS = [(0, 1, 2), (0, 1, 5), (0, 4, 5), (0, 2, 4),
        (1, 2, 3), (1, 3, 5), (2, 3, 4), (3, 4, 5)]
cx = SimplicialComplex(TRIS, maxdim=2)
filt = Filtration(cx, 2)

# ---- planar embedding: outer triangle 0,1,2; inner triangle 4,5,3 ----------
# R_IN trades the room in the three corner faces (which grows with it) against
# the room in the three edge faces (which shrinks); 0.26 is where the smallest
# face is still large enough to hold a 6.8 pt label without spilling sideways.
R_IN = 0.26
POS = {}
for v, a in zip([0, 1, 2], np.deg2rad([90, 210, 330])):
    POS[v] = np.array([np.cos(a), np.sin(a)])
for v, a in zip([4, 5, 3], np.deg2rad([30, 150, 270])):
    POS[v] = R_IN * np.array([np.cos(a), np.sin(a)])
OUTER, EXPAND = (0, 1, 2), 1.30

# ---- signal ---------------------------------------------------------------
f = 0.7 * np.array([1.6, 1.1, 1.1, 0.35, 0.35, 0.7])      # vertex strengths
PAIR, B = (0, 1), 1.5                                      # pair bonus
TPLUS, TMINUS, C = (3, 4, 5), (0, 4, 5), 3.5               # three-way terms
x = np.array([f[u] + f[v] + f[w]
              + (B if set(PAIR) <= {u, v, w} else 0.0)
              + (C if (u, v, w) == TPLUS else 0.0)
              - (C if (u, v, w) == TMINUS else 0.0)
              for (u, v, w) in cx.simplices[2]])
parts = [filt.project_V(x, 1), filt.project(x, 2), filt.project(x, 3)]
pi = filt.pi(x)
assert np.abs(sum(parts) - x).max() < 1e-10

vmax = max(np.abs(x).max(), *(np.abs(p).max() for p in parts))
norm = plt.Normalize(-vmax, vmax)
cmap = plt.get_cmap("RdBu_r")


def incenter(P, clear=0.12):
    """Label position that stays away from the vertices even for thin triangles: the
    incenter, pushed off the nearest vertex if it would sit on the node marker."""
    a = np.linalg.norm(P[1] - P[2]); b = np.linalg.norm(P[0] - P[2])
    c = np.linalg.norm(P[0] - P[1])
    z = (a * P[0] + b * P[1] + c * P[2]) / (a + b + c)
    d = np.linalg.norm(P - z, axis=1)
    i = int(np.argmin(d))
    if d[i] < clear:
        u = z - P[i]
        z = P[i] + u / max(np.linalg.norm(u), 1e-9) * clear
    return z


def ring(ax, tri, colour):
    P = np.array([POS[v] for v in tri]); ctr = P.mean(0)
    out = ctr + (P - ctr) * EXPAND
    verts = np.vstack([out, out[:1], P[::-1], P[-1:]])
    codes = ([Path.MOVETO] + [Path.LINETO] * 2 + [Path.CLOSEPOLY]
             + [Path.MOVETO] + [Path.LINETO] * 2 + [Path.CLOSEPOLY])
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=colour, edgecolor="0.3",
                           lw=0.5, zorder=0))


def draw(ax, vals, title):
    """One panel. The label sits at the incenter of each face, and in the middle of the
    bottom strip of the ring for the outer face."""
    for j, t in enumerate(cx.simplices[2]):
        col = cmap(norm(vals[j]))
        if t == OUTER:
            ring(ax, t, col)
            lp = np.array([0.0, -0.5 * (1.0 + EXPAND) / 2.0 - 0.005])
        else:
            P = np.array([POS[v] for v in t])
            ax.add_patch(Polygon(P, facecolor=col, edgecolor="0.3", lw=0.5, zorder=1))
            lp = incenter(P)
        ax.text(lp[0], lp[1], f"{vals[j]:.1f}", ha="center", va="center",
                fontsize=6.8, zorder=4,
                color="white" if abs(vals[j]) > 0.60 * vmax else "0.10")
    for e in cx.simplices[1]:
        p, q = POS[e[0]], POS[e[1]]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="0.2", lw=0.6, zorder=2)
    for p in POS.values():
        ax.add_patch(Circle(p, 0.034, facecolor="0.15", edgecolor="none", zorder=3))
    ax.set_title(title, fontsize=7.2, pad=1.5, linespacing=1.15)
    ax.set_xlim(-1.19, 1.19); ax.set_ylim(-0.675, 1.315)
    ax.set_aspect("equal"); ax.axis("off")


fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.60))
draw(axes[0], x, "triangle signal  $\\mathbf{x}$\n"
                r"$100\%$ of $\|\mathbf{x}\|^2$")
draw(axes[1], parts[0], "order $\\leq\\!1$:  $\\mathbf{P}_{\\mathcal{V}_1}\\mathbf{x}$\n"
                        rf"node-additive,  ${100*(pi[0]+pi[1]):.0f}\%$")
draw(axes[2], parts[1], "order $2$:  $\\mathbf{P}_2\\mathbf{x}$\n"
                        rf"pair effects,  ${100*pi[2]:.0f}\%$")
draw(axes[3], parts[2], "order $3$:  $\\mathbf{P}_3\\mathbf{x}$\n"
                        rf"pure three-way,  ${100*pi[3]:.0f}\%$")

fig.subplots_adjust(left=0.004, right=0.936, top=0.79, bottom=0.004, wspace=0.008)
cax = fig.add_axes([0.947, 0.05, 0.013, 0.68])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.ax.tick_params(labelsize=6.0, length=2, pad=1.5)
fig.savefig(os.path.join(FIG, "fig_bands_illustration.pdf"))
fig.savefig(os.path.join(FIG, "fig_bands_illustration.png"), dpi=300)
print("dim W =", filt.dimW, " pi =", np.round(pi, 4))
print("saved figures/fig_bands_illustration.{pdf,png}")
