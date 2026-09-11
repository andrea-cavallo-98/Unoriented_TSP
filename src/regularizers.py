"""The regularizer matrices R of eq. (13).

Proposed in the paper: `cohesion` (R = lambda_1 I - L^un_{p,q}), `polarization`
(R = L^un_{p,q}) and `order_reg` (R_ord = sum_k beta_k P_k, eq. 12).
Baselines: `oriented` (Hodge Tikhonov, low- and high-pass, and the re-oriented
variants) and `order_ls_design` (the additive order-<=k least-squares fit).
The remaining baselines need no matrix: ridge is the identity and neighbour-mean and
predict-mean are in src/recon.py.

Every builder returns a PSD matrix already rescaled to unit spectral radius, so that a
single alpha grid means the same thing for every method.
"""
from __future__ import annotations

import numpy as np


def _sym(M):
    return (np.asarray(M, float) + np.asarray(M, float).T) / 2.0


def lam_max(M):
    return float(np.linalg.eigvalsh(_sym(M))[-1])


def normalize(R):
    R = _sym(R)
    l = lam_max(R)
    return R / l if l > 1e-12 else R


# ------------------------------------------------------------- proposed (paper)
def cohesion(cx, p, q):
    """R = lambda_1 I - L^un_{p,q}: soft low-pass in omega = lambda_1 - lambda."""
    L = cx.L_un(p, q)
    return normalize(lam_max(L) * np.eye(L.shape[0]) - L)


def polarization(cx, p, q):
    """R = L^un_{p,q}: high-pass; passes every order above q+1 with a fixed factor."""
    return normalize(cx.L_un(p, q))


def order_reg(filt, beta):
    """R_ord = sum_k beta_k P_k (eq. 12). `beta` is a length-(p+2) non-negative vector.
    Infinite entries are replaced by a large finite penalty (hard band limit)."""
    beta = np.asarray(beta, float).copy()
    finite = beta[np.isfinite(beta)]
    big = 1e6 * (finite.max() if finite.size and finite.max() > 0 else 1.0)
    beta[~np.isfinite(beta)] = big
    R = sum(b * filt.P(k) for k, b in enumerate(beta) if b > 0)
    if np.isscalar(R):
        R = np.zeros((filt.N, filt.N))
    return normalize(R)


# ------------------------------------------------------------------- baselines
def oriented(cx, p, kind="full", signs=None, highpass=False, lam=None):
    """Hodge Tikhonov: 'lower' = B_p^T B_p, 'upper' = B_{p+1} B_{p+1}^T, 'full' = sum.
    `highpass=True` returns lambda_1 I - R, the oriented analogue of `cohesion`.

    `lam` supplies lambda_max when it is already known. Re-orienting acts by
    L -> D_eps L D_eps, which is a similarity, so every re-orientation of the same
    operator has the same spectrum and the lexicographic lambda_max can be reused --
    this is what makes the 20-re-orientation runs affordable."""
    R = cx.L_sg(p, kind, signs)
    if np.abs(R).max() == 0:
        return R
    l = float(lam) if lam is not None else lam_max(R)
    if highpass:
        # lambda_1 I - R has a different spectral radius, so it is always recomputed
        R = l * np.eye(R.shape[0]) - R
        l = lam_max(R)
    return _sym(R) / (l if l > 1e-12 else 1.0)


def order_ls_design(cx, p, k):
    """Design matrix for the order-<=k additive least-squares baseline (`ls1`, `ls2`):
    columns are the lifts of the (k-1)-simplex indicators, i.e. Q(k-1, p)^T."""
    return cx.Q(k - 1, p).T
