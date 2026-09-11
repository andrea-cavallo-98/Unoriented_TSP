"""Interaction-order filtration: V_k, bands W_k, projectors and energy profiles.

V_k = range(Q(k-1, p)^T) for k = 0..p+1, with Q(-1,p) = 1^T and Q(p,p) = I, so
V_0 = span(1) and V_{p+1} = R^{N_p}. W_k = V_k \\ominus V_{k-1}.

Bases are built by pivoted QR on Q(k-1,p)^T (never by an explicit inverse) because
Q(k-1,p) is routinely rank deficient: nodes in no p-simplex, and "twin" nodes that
appear in exactly the same p-simplices, both cost rank.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def range_basis(M, rcond=1e-10, rank=None, abs_tol=None):
    """Orthonormal basis for range(M) by pivoted QR with a numerical rank cut.

    `rank` forces the number of columns (pivoting orders them by decreasing pivot, so
    the first `rank` are the well-determined directions). `abs_tol` cuts on the absolute
    pivot instead of relative to the largest one, which matters when M is numerically
    zero: then the relative test keeps spurious noise directions.
    """
    if M.size == 0 or M.shape[1] == 0:
        return np.zeros((M.shape[0], 0))
    Qm, R, _ = qr(M, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    if rank is not None:
        r = int(min(max(rank, 0), Qm.shape[1]))
    elif abs_tol is not None:
        r = int(np.sum(d > abs_tol)) if d.size else 0
    else:
        r = int(np.sum(d > rcond * max(d[0], 1e-300))) if d.size else 0
    return Qm[:, :r]


class Filtration:
    """The nested bases and band projectors at one level p."""

    def __init__(self, cx, p, rcond=1e-10):
        self.p = p
        self.N = cx.N[p]
        self.U = []                      # orthonormal basis of V_k
        self.rank_Q = []                 # rank of Q(k-1, p)
        for k in range(p + 2):
            Q = cx.Q(k - 1, p)
            U = range_basis(Q.T, rcond)
            self.U.append(U)
            self.rank_Q.append(U.shape[1])
        self.dimV = [U.shape[1] for U in self.U]
        self.dimW = [self.dimV[0]] + [self.dimV[k] - self.dimV[k - 1]
                                      for k in range(1, p + 2)]
        # band bases: orthonormal basis of W_k. The dimension is known from the ranks
        # of the inclusion matrices, so it is imposed rather than re-estimated -- when
        # W_k is trivial the residual is pure round-off and a relative rank cut would
        # invent a direction.
        self.Uw = [self.U[0]]
        for k in range(1, p + 2):
            R = self.U[k] - self.U[k - 1] @ (self.U[k - 1].T @ self.U[k])
            self.Uw.append(range_basis(R, rcond, rank=self.dimW[k]))

    # ------------------------------------------------------------- projectors
    def P(self, k):
        W = self.Uw[k]
        return W @ W.T if W.shape[1] else np.zeros((self.N, self.N))

    def project(self, x, k):
        W = self.Uw[k]
        return W @ (W.T @ x) if W.shape[1] else np.zeros_like(x)

    def project_V(self, x, k):
        U = self.U[k]
        return U @ (U.T @ x) if U.shape[1] else np.zeros_like(x)

    # ------------------------------------------------------------------ energy
    def pi(self, x):
        """(pi_0, ..., pi_{p+1}) energy fractions of x in the bands."""
        x = np.asarray(x, float)
        n2 = float(x @ x)
        e = np.array([float(np.sum((W.T @ x) ** 2)) if W.shape[1] else 0.0
                      for W in self.Uw])
        return e / n2 if n2 > 0 else e

    def dim_frac(self):
        return np.array(self.dimW, float) / self.N


def permutation_null(x, filt, n=500, seed=0, block=125):
    """Distribution of pi_k under shuffling the signal over simplices (vectorised)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    nrm = float(x @ x)
    Uall = np.hstack([W for W in filt.Uw if W.shape[1]])
    lab = np.concatenate([np.full(W.shape[1], k)
                          for k, W in enumerate(filt.Uw) if W.shape[1]])
    out = np.empty((n, len(filt.Uw)))
    done = 0
    while done < n:
        m = min(block, n - done)
        X = np.stack([rng.permutation(x) for _ in range(m)], axis=1)
        C = (Uall.T @ X) ** 2
        for k in range(len(filt.Uw)):
            sel = lab == k
            out[done:done + m, k] = (C[sel].sum(0) / nrm) if sel.any() else 0.0
        done += m
    return out
