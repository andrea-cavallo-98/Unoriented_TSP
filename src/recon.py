"""Trials, baselines and error metrics for the reconstruction experiments.

The Tikhonov problem itself,

    min_x ||M(y - x)||^2 + alpha x^T R x + gamma x^T Gamma x
        =>  x = (M + alpha R + gamma Gamma)^{-1} M y                     (eq. 13)

is solved in scripts/02_reconstruction.py, which also owns the alpha and gamma grids;
what lives here is everything around it: the shared noise/mask draw for one trial, the
two non-Tikhonov baselines (additive least squares and neighbour-mean imputation), and
the error metrics including the band-resolved one.
"""
from __future__ import annotations

import numpy as np


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def ls_fit(design, y, obs, ridge_eps=1e-6):
    """Least squares for the order-<=k additive model on the observed entries, with a
    small ridge for stability; predicts everywhere."""
    A = np.asarray(design)[np.asarray(obs)]
    b = y[np.asarray(obs)]
    G = A.T @ A + ridge_eps * np.eye(A.shape[1])
    c = np.linalg.solve(G, A.T @ b)
    return design @ c


def neighbour_mean(cx, p, q, y, obs, iters=50, tol=1e-8):
    """Iterated co-face neighbour-mean imputation (imputation baseline)."""
    A = cx.L_un(p, q).copy()
    np.fill_diagonal(A, 0.0)
    A = (A > 0).astype(float)
    deg = A.sum(1)
    x = np.zeros_like(y)
    obs = np.asarray(obs)
    x[obs] = y[obs]
    fill = y[obs].mean()
    miss = np.setdiff1d(np.arange(len(y)), obs)
    x[miss] = fill
    for _ in range(iters):
        num = A @ x
        new = np.where(deg > 0, num / np.maximum(deg, 1), fill)
        upd = x.copy(); upd[miss] = new[miss]
        if np.max(np.abs(upd - x)) < tol:
            x = upd; break
        x = upd
    return x


def band_errors(filt, xhat, s):
    """e_k = ||P_k (xhat - s)||^2 / ||P_k s||^2 for every band."""
    r = xhat - s
    out = []
    for k, W in enumerate(filt.Uw):
        if W.shape[1] == 0:
            out.append(np.nan); continue
        num = float(np.sum((W.T @ r) ** 2))
        den = float(np.sum((W.T @ s) ** 2))
        out.append(num / den if den > 1e-14 else np.nan)
    return out


def make_trial(s, sigma_rel, miss_frac, seed):
    """Shared noise realisation and mask for one trial, identical across methods."""
    rng = np.random.default_rng(seed)
    n = len(s)
    y = s + rng.normal(0.0, sigma_rel * s.std(), n) if sigma_rel else s.copy()
    perm = rng.permutation(n)
    n_miss = int(round(miss_frac * n))
    miss = perm[:n_miss]; obs = perm[n_miss:]
    return y, np.sort(obs), np.sort(miss)
