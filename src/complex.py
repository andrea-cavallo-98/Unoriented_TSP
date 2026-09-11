"""Simplicial complexes from observed hyperedges, with signless and oriented operators.

Conventions fixed here and used everywhere:

* A *hyperedge* is an observed event's participant set. The complex is the downward
  closure of all hyperedges of size <= maxdim+1.
* **Exact-k signals**: the signal on a simplex sigma counts events whose participant set
  is *exactly* sigma. A simplex present only as a face of larger observed sets carries a
  real zero. This is rule 1 of the experiment spec: a triangle signal is never computed
  from the edge or node signals.
* `Q(q, p)` is the unoriented inclusion matrix, rows = q-simplices, columns =
  p-simplices, entry 1 iff the q-simplex is a face of the p-simplex. `Q(-1, p) = 1^T`
  and `Q(p, p) = I`.
* `B(p)` is the oriented boundary with the lexicographic orientation: for
  sigma = (v_0 < ... < v_p), the face sigma \\ {v_i} gets sign (-1)^i. This satisfies
  B(p) @ B(p+1) == 0 exactly.
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


class SimplicialComplex:
    def __init__(self, hyperedges, maxdim=2, nodes=None):
        """hyperedges: iterable of node-id tuples. maxdim: top simplex dimension."""
        self.maxdim = maxdim
        faces = [set() for _ in range(maxdim + 1)]
        for he in hyperedges:
            he = tuple(sorted(set(he)))
            if len(he) < 2:
                # a singleton still contributes its vertex
                faces[0].update(he)
                continue
            for k in range(0, maxdim + 1):
                if len(he) >= k + 1:
                    faces[k].update(itertools.combinations(he, k + 1))
        faces[0] = {v if isinstance(v, tuple) else (v,) for v in faces[0]}
        if nodes is not None:
            faces[0] |= {(v,) for v in nodes}
        self.simplices = [sorted(faces[k]) for k in range(maxdim + 1)]
        self.index = [{s: i for i, s in enumerate(self.simplices[k])}
                      for k in range(maxdim + 1)]
        self.N = [len(self.simplices[k]) for k in range(maxdim + 1)]

    # ------------------------------------------------------------------ signals
    def exact_counts(self, hyperedges, k, weights=None):
        """Number (or summed weight) of events whose participant set is EXACTLY each
        k-simplex. Events not matching any k-simplex are ignored."""
        c = np.zeros(self.N[k])
        idx = self.index[k]
        for j, he in enumerate(hyperedges):
            s = tuple(sorted(set(he)))
            if len(s) != k + 1:
                continue
            i = idx.get(s)
            if i is not None:
                c[i] += 1.0 if weights is None else float(weights[j])
        return c

    def value_signal(self, mapping, k, default=np.nan):
        """Signal from a dict {simplex tuple: measured value} (for measured, not counted,
        data such as brightness or growth). Missing simplices get `default`."""
        v = np.full(self.N[k], default, float)
        for s, val in mapping.items():
            i = self.index[k].get(tuple(sorted(s)))
            if i is not None:
                v[i] = val
        return v

    # ------------------------------------------------------- incidence matrices
    def Q(self, q, p):
        """Unoriented inclusion matrix, shape (N_q, N_p); q may be -1 or equal to p."""
        if q == p:
            return np.eye(self.N[p])
        if q == -1:
            return np.ones((1, self.N[p]))
        if q > p:
            raise ValueError("Q(q, p) requires q < p")
        M = np.zeros((self.N[q], self.N[p]))
        iq = self.index[q]
        for j, s in enumerate(self.simplices[p]):
            for f in itertools.combinations(s, q + 1):
                M[iq[f], j] = 1.0
        return M

    def L_un(self, p, q):
        """Signless Laplacian acting on level p through level q (eq. 1 of the paper)."""
        if q < p:
            A = self.Q(q, p)
            return A.T @ A
        A = self.Q(p, q)
        return A @ A.T

    # ---------------------------------------------------------- oriented / Hodge
    def B(self, p, signs=None):
        """Oriented boundary B_p: R^{N_p} -> R^{N_{p-1}}. `signs` is an optional list of
        per-level +-1 vectors implementing a re-orientation, applied as row/column
        scaling of the cached lexicographic boundary."""
        cache = getattr(self, "_B_cache", None)
        if cache is None:
            cache = self._B_cache = {}
        if p not in cache:
            M = np.zeros((self.N[p - 1], self.N[p]))
            ip = self.index[p - 1]
            for j, s in enumerate(self.simplices[p]):
                for i in range(len(s)):
                    f = s[:i] + s[i + 1:]
                    M[ip[f], j] = (-1.0) ** i
            cache[p] = M
        M = cache[p]
        if signs is None:
            return M
        return signs[p - 1][:, None] * M * signs[p][None, :]

    def L_sg(self, p, kind="full", signs=None):
        """Oriented Laplacian at level p: 'lower' = B_p^T B_p, 'upper' = B_{p+1}B_{p+1}^T,
        'full' = their sum. Missing levels contribute zero."""
        lo = np.zeros((self.N[p], self.N[p]))
        up = np.zeros((self.N[p], self.N[p]))
        if p >= 1:
            Bp = self.B(p, signs)
            lo = Bp.T @ Bp
        if p + 1 <= self.maxdim:
            Bn = self.B(p + 1, signs)
            up = Bn @ Bn.T
        return {"lower": lo, "upper": up, "full": lo + up}[kind]

    def random_signs(self, rng):
        return [rng.choice([-1.0, 1.0], size=n) for n in self.N]


# ---------------------------------------------------------------------- helpers
def _fast_ints(path):
    import pandas as pd
    return pd.read_csv(path, header=None, dtype=np.int64,
                       engine="c").iloc[:, 0].to_numpy()


def read_benson(path, name, max_size=None):
    """Read <name>-nverts.txt / -simplices.txt / -times.txt into (hyperedges, times).

    `max_size` keeps only hyperedges with at most that many vertices, which is applied
    while slicing so the large coauthorship files never materialise as Python tuples.
    """
    import os
    nv = _fast_ints(os.path.join(path, f"{name}-nverts.txt"))
    sp = _fast_ints(os.path.join(path, f"{name}-simplices.txt"))
    tf = os.path.join(path, f"{name}-times.txt")
    tm = _fast_ints(tf) if os.path.exists(tf) else np.zeros(len(nv), np.int64)
    starts = np.concatenate([[0], np.cumsum(nv)[:-1]])
    out, times = [], []
    for n, s, t in zip(nv, starts, tm):
        if max_size is not None and n > max_size:
            continue
        out.append(tuple(sp[s:s + n]))
        times.append(t)
    return out, np.asarray(times)


def top_nodes(hyperedges, sizes, n_keep):
    """The n_keep nodes with the most participations in hyperedges of the given sizes."""
    c = Counter()
    for he in hyperedges:
        if len(set(he)) in sizes:
            c.update(set(he))
    return set(v for v, _ in c.most_common(n_keep))


def restrict(hyperedges, times, keep, sizes):
    """Keep hyperedges whose (deduplicated) node set lies inside `keep` and has an
    allowed size. Returns the filtered hyperedges and their timestamps."""
    hs, ts = [], []
    for he, t in zip(hyperedges, times):
        s = tuple(sorted(set(he)))
        if len(s) in sizes and all(v in keep for v in s):
            hs.append(s); ts.append(t)
    return hs, np.asarray(ts)
