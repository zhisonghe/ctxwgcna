"""Steps 2-4: per-context similarity -> Fisher z -> top-n contexts per gene pair -> summary z.

For every gene pair (i, j) and context c with n_c observations:
    z_c = arctanh(s_c(i, j))            Fisher z
    t_c = z_c * sqrt(n_c - 3)           standardized: comparable across contexts of different size
Contexts are ranked per pair by t_c ('positive' selection, for signed networks) or |t_c|
('absolute', for unsigned networks); the top-n are kept together with their context ids.
The summary is the (n_c - 3)-weighted mean of the selected z_c (the pooled Fisher z estimate),
optionally corrected for the selection bias of picking the best n of K contexts.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .similarity import fisher_z, similarity_matrix

SELECTIONS = ("positive", "absolute")
CALIBRATIONS = ("none", "analytic", "permutation")


@dataclass
class ContextSupport:
    genes: pd.Index
    contexts: list  # context names; values in top_idx index into this list
    n_obs: np.ndarray  # observations per context
    top_n: int
    selection: str
    weighting: str
    method: str
    top_idx: np.ndarray  # (G, G, top_n) int16, sorted by decreasing ranking statistic
    top_t: np.ndarray  # (G, G, top_n) float32, ranking statistic (t_c, or |t_c|)
    z_raw: np.ndarray  # (G, G) summary Fisher z before calibration
    z: np.ndarray  # (G, G) summary Fisher z after calibration
    calibrate: str
    bias: object  # float or (G, G) array subtracted from z_raw
    null_sd: float  # spread of the null summary (for reference)
    t_all: np.ndarray = None  # (K, G, G) float32 signed standardized z per context, if stored

    @property
    def r(self):
        """Summary similarity on the correlation scale."""
        return np.tanh(self.z)

    def _directional(self, t):
        return np.abs(t) if self.selection == "absolute" else t

    def context_t(self, idx=None):
        """(K, m, m) per-context standardized z (signed), optionally restricted to gene indices idx."""
        if self.t_all is None:
            raise ValueError("per-context statistics were not stored (store_context_t=False or too large)")
        return self.t_all if idx is None else self.t_all[:, idx][:, :, idx]

    def context_weights(self, idx=None):
        """(K, m, m) soft context profile per edge: positive part of t (|t| for unsigned), normalised
        over contexts to sum to 1. Flat for globally co-expressed pairs, peaked for local ones."""
        w = np.maximum(self._directional(self.context_t(idx)), 0.0)
        tot = w.sum(axis=0, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(tot > 0, w / tot, 0.0)
        return w

    def pair_profile(self, gene_a, gene_b):
        """Full per-context support of one gene pair (all contexts, not just the top-n)."""
        i, j = self.genes.get_loc(gene_a), self.genes.get_loc(gene_b)
        t = self.context_t()[:, i, j].astype(np.float64)
        s = np.sqrt(np.maximum(self.n_obs - 3, 1))
        w = self.context_weights([i, j])[:, 0, 1]
        topn = set(self.top_idx[i, j])
        return pd.DataFrame({"n_obs": self.n_obs, "t": t, "similarity": np.tanh(t / s), "weight": w,
                             "in_top_n": [k in topn for k in range(len(self.contexts))]},
                            index=pd.Index(self.contexts, name="context")).sort_values("t", ascending=False)

    def indicator(self, k, min_t=None):
        """Boolean (G, G): context k is among the top-n supporting contexts of the pair."""
        hit = self.top_idx == k
        if min_t is not None:
            hit &= self.top_t >= min_t
        return hit.any(axis=-1)

    def pair(self, gene_a, gene_b):
        """Supporting contexts of one gene pair."""
        i, j = self.genes.get_loc(gene_a), self.genes.get_loc(gene_b)
        idx = self.top_idx[i, j]
        s = np.sqrt(np.maximum(self.n_obs[idx] - 3, 1))
        return pd.DataFrame({
            "context": [self.contexts[k] for k in idx],
            "n_obs": self.n_obs[idx],
            "t": self.top_t[i, j],
            "z": self.top_t[i, j] / s,
            "similarity": np.tanh(self.top_t[i, j] / s),
        })

    def context_matrix(self, rank=0):
        """(G, G) DataFrame with the name of the rank-th best supporting context per pair."""
        names = np.asarray(self.contexts, dtype=object)
        return pd.DataFrame(names[self.top_idx[..., rank]], index=self.genes, columns=self.genes)


def _top_n_stream(blocks, method, top_n, selection, chunk_bytes, rng=None, store=None):
    """Keep the top-n contexts per pair while streaming over contexts in memory-bounded chunks.

    If rng is given, each gene's observations are permuted independently within each context
    (null: no gene-gene association, gene marginals preserved).
    If store is a (K, G, G) array, the signed standardized z of every context is written into it.
    """
    K = len(blocks)
    G = blocks[0].X.shape[1]
    top_key = np.full((G, G, top_n), -np.inf, dtype=np.float32)
    top_idx = np.full((G, G, top_n), -1, dtype=np.int16)
    per_chunk = max(1, int(chunk_bytes // (G * G * 4)))

    for start in range(0, K, per_chunk):
        ks = list(range(start, min(K, start + per_chunk)))
        keys = np.empty((G, G, len(ks)), dtype=np.float32)
        for j, k in enumerate(ks):
            X = blocks[k].X
            if rng is not None:
                X = rng.permuted(X, axis=0)
            t = fisher_z(similarity_matrix(X, method)) * np.sqrt(max(X.shape[0] - 3, 1))
            if store is not None:
                store[k] = t
            keys[..., j] = np.abs(t) if selection == "absolute" else t
        cand_key = np.concatenate([top_key, keys], axis=-1)
        cand_idx = np.concatenate(
            [top_idx, np.broadcast_to(np.asarray(ks, dtype=np.int16), (G, G, len(ks)))], axis=-1)
        part = np.argpartition(-cand_key, top_n - 1, axis=-1)[..., :top_n]
        top_key = np.take_along_axis(cand_key, part, axis=-1)
        top_idx = np.take_along_axis(cand_idx, part, axis=-1)

    order = np.argsort(-top_key, axis=-1, kind="stable")
    return np.take_along_axis(top_key, order, axis=-1), np.take_along_axis(top_idx, order, axis=-1)


def _summarize(top_key, top_idx, n_obs, weighting):
    s = np.sqrt(np.maximum(n_obs - 3, 1)).astype(np.float64)[top_idx]
    zsel = top_key / s
    w = s ** 2 if weighting == "n" else np.ones_like(s)
    return (w * zsel).sum(axis=-1) / w.sum(axis=-1)


def _analytic_null(n_obs, top_n, selection, weighting, n_sim, rng):
    """Null summary assuming independent contexts and t_c ~ N(0, 1) (valid for centred methods)."""
    K = len(n_obs)
    t = rng.standard_normal((n_sim, K))
    key = np.abs(t) if selection == "absolute" else t
    idx = np.argpartition(-key, top_n - 1, axis=1)[:, :top_n]
    zsum = _summarize(np.take_along_axis(key, idx, axis=1), idx, n_obs, weighting)
    return float(zsum.mean()), float(zsum.std())


def compute_support(blocks, method="pearson", top_n=3, selection="positive", weighting="n",
                    calibrate="analytic", n_perm=10, n_sim=200_000, chunk_mb=128, seed=0, genes=None,
                    store_context_t="auto", store_max_mb=2048):
    """Compute the context-support summary for all gene pairs.

    :param blocks: list of ContextBlock (see data.make_context_blocks).
    :param method: 'pearson' | 'spearman' | 'bicor' | 'cosine'.
    :param top_n: number of best contexts summarized per pair (n >= 2 recommended).
    :param selection: 'positive' (rank by signed t; for signed / signed hybrid networks) or
        'absolute' (rank by |t|; for unsigned networks).
    :param weighting: 'n' (weight selected z by n_c - 3, the pooled estimate) or 'equal'.
    :param calibrate: 'none'; 'analytic' (subtract expected null summary of a best-n-of-K
        selection, assuming independent contexts and zero null mean, i.e. centred methods);
        'permutation' (permute each gene within each context, n_perm times; per-pair null for
        'cosine', pooled over pairs otherwise).
    :param store_context_t: keep the full (K, G, G) per-context statistics (float32) for soft context
        weights and module splitting. 'auto' stores them if they need <= store_max_mb.
    """
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}")
    if calibrate not in CALIBRATIONS:
        raise ValueError(f"calibrate must be one of {CALIBRATIONS}")
    if weighting not in ("n", "equal"):
        raise ValueError("weighting must be 'n' or 'equal'")
    K = len(blocks)
    if not 1 <= top_n <= K:
        raise ValueError(f"top_n must be between 1 and the number of contexts ({K})")
    if K > np.iinfo(np.int16).max:
        raise ValueError("too many contexts for int16 context ids")

    n_obs = np.array([b.n_obs for b in blocks])
    chunk_bytes = chunk_mb * 2 ** 20
    rng = np.random.default_rng(seed)

    G = blocks[0].X.shape[1]
    store_mb = K * G * G * 4 / 2 ** 20
    if store_context_t == "auto":
        store_context_t = store_mb <= store_max_mb
        if not store_context_t:
            print(f"Not storing per-context statistics ({store_mb:.0f} MB > store_max_mb={store_max_mb}); "
                  "they will be recomputed for module genes when needed.")
    t_all = np.empty((K, G, G), dtype=np.float32) if store_context_t else None

    top_key, top_idx = _top_n_stream(blocks, method, top_n, selection, chunk_bytes, store=t_all)
    z_raw = _summarize(top_key, top_idx, n_obs, weighting)

    if calibrate == "none":
        bias, null_sd = 0.0, float("nan")
    elif calibrate == "analytic":
        if method == "cosine":
            print("Warning: analytic calibration assumes a zero null mean, which does not hold for "
                  "cosine similarity; consider calibrate='permutation'.")
        bias, null_sd = _analytic_null(n_obs, top_n, selection, weighting, n_sim, rng)
    else:
        acc, acc2 = np.zeros_like(z_raw), np.zeros_like(z_raw)
        for _ in range(n_perm):
            nk, ni = _top_n_stream(blocks, method, top_n, selection, chunk_bytes, rng=rng)
            zn = _summarize(nk, ni, n_obs, weighting)
            acc += zn
            acc2 += zn ** 2
        mean = acc / n_perm
        off = ~np.eye(len(mean), dtype=bool)
        null_sd = float(np.sqrt(np.maximum(acc2[off] / n_perm - mean[off] ** 2, 0)).mean())
        if method == "cosine":
            bias = mean
        else:
            bias = float(mean[off].mean())

    z = z_raw - bias
    if selection == "absolute":
        z = np.maximum(z, 0.0)
    np.fill_diagonal(z, np.arctanh(1 - 1e-7))

    return ContextSupport(
        genes=pd.Index(genes) if genes is not None else pd.RangeIndex(top_key.shape[0]),
        contexts=[b.name for b in blocks], n_obs=n_obs, top_n=top_n, selection=selection,
        weighting=weighting, method=method, top_idx=top_idx, top_t=top_key,
        z_raw=z_raw, z=z, calibrate=calibrate, bias=bias, null_sd=null_sd, t_all=t_all,
    )
