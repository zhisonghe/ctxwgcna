"""Steps 2-4: per-context similarity -> Fisher z -> top-n contexts per gene pair -> summary z.

For every gene pair (i, j) and context c with n_c observations:
    z_c = arctanh(s_c(i, j))            Fisher z
    t_c = z_c * sqrt(n_c - 3)           standardized: comparable across contexts of different size
Contexts are ranked per pair by t_c ('positive' selection, for signed networks) or |t_c|
('absolute', for unsigned networks); the top-n are kept together with their context ids.
The summary is the (n_c - 3)-weighted mean of the selected z_c (the pooled Fisher z estimate),
optionally corrected for the selection bias of picking the best n of K contexts.

Array shapes used throughout: G genes, K contexts, n = top_n.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .similarity import standardized_z, z_scale

SELECTIONS = ("positive", "absolute")
CALIBRATIONS = ("none", "analytic", "permutation")


def directional(t, selection):
    """The ranking statistic: t itself ('positive') or |t| ('absolute')."""
    return np.abs(t) if selection == "absolute" else t


def soft_context_weights(t, selection):
    """Soft context profile per edge from per-context statistics t of shape (K, ...).

    w^k = max(t_k, 0) / sum_k max(t_k, 0)   (|t_k| instead of t_k for 'absolute' selection)

    Flat over contexts for globally co-expressed pairs, peaked for context-local ones. Edges with
    no positive support in any context get all-zero weights.
    """
    w = np.maximum(directional(t, selection), 0.0)
    total = w.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, w / total, 0.0)


@dataclass
class ContextSupport:
    """Result of compute_support: which contexts support each gene pair, and the summary z."""
    genes: pd.Index
    contexts: list  # context names; the values in top_idx index into this list
    n_obs: np.ndarray  # (K,) observations per context
    top_n: int
    selection: str
    weighting: str
    method: str
    top_idx: np.ndarray  # (G, G, n) int16 context ids, sorted by decreasing ranking statistic
    top_t: np.ndarray  # (G, G, n) float32 ranking statistic of those contexts (t_c, or |t_c|)
    z_raw: np.ndarray  # (G, G) summary Fisher z before calibration
    z: np.ndarray  # (G, G) summary Fisher z after calibration
    calibrate: str
    bias: object  # float or (G, G) array subtracted from z_raw
    null_sd: float  # spread of the null summary (for reference)
    t_all: np.ndarray = None  # (K, G, G) float32 signed t_c of every context, if stored
    top_t_signed: np.ndarray = None  # (G, G, n) signed t_c of the selected contexts
    # (only stored for selection='absolute'; for 'positive' it equals top_t)

    @property
    def r(self):
        """Summary similarity on the correlation scale."""
        return np.tanh(self.z)

    # --- full per-context statistics (need t_all) ---

    def context_t(self, idx=None):
        """(K, m, m) signed per-context t, optionally restricted to the gene indices idx."""
        if self.t_all is None:
            raise ValueError("per-context statistics were not stored (store_context_t=False or too large)")
        return self.t_all if idx is None else self.t_all[:, idx][:, :, idx]

    def context_weights(self, idx=None):
        """(K, m, m) soft context weights per edge (see soft_context_weights)."""
        return soft_context_weights(self.context_t(idx), self.selection)

    def pair_profile(self, gene_a, gene_b):
        """Per-context support of one gene pair over all contexts (not just the top-n)."""
        i, j = self.genes.get_loc(gene_a), self.genes.get_loc(gene_b)
        t = self.context_t()[:, i, j].astype(np.float64)
        weight = self.context_weights([i, j])[:, 0, 1]
        in_top = set(self.top_idx[i, j])
        return pd.DataFrame({
            "n_obs": self.n_obs,
            "t": t,
            "similarity": np.tanh(t / z_scale(self.n_obs)),
            "weight": weight,
            "in_top_n": [k in in_top for k in range(len(self.contexts))],
        }, index=pd.Index(self.contexts, name="context")).sort_values("t", ascending=False)

    # --- top-n supporting contexts (always available) ---

    def indicator(self, k, min_t=None):
        """(G, G) bool: context k is among the top-n supporting contexts of the pair
        (and, if min_t is given, its ranking statistic is >= min_t)."""
        hit = self.top_idx == k
        if min_t is not None:
            hit &= self.top_t >= min_t
        return hit.any(axis=-1)

    def pair(self, gene_a, gene_b):
        """The top-n supporting contexts of one gene pair, best first."""
        i, j = self.genes.get_loc(gene_a), self.genes.get_loc(gene_b)
        idx = self.top_idx[i, j]
        t = (self.top_t if self.top_t_signed is None else self.top_t_signed)[i, j]
        z = t / z_scale(self.n_obs[idx])
        return pd.DataFrame({
            "context": [self.contexts[k] for k in idx],
            "n_obs": self.n_obs[idx],
            "t": t,  # signed
            "rank_stat": self.top_t[i, j],  # what the contexts were ranked by (|t| if absolute)
            "z": z,
            "similarity": np.tanh(z),
        })

    def context_matrix(self, rank=0):
        """(G, G) DataFrame with the name of the rank-th best supporting context of each pair."""
        names = np.asarray(self.contexts, dtype=object)
        return pd.DataFrame(names[self.top_idx[..., rank]], index=self.genes, columns=self.genes)


def _top_n_stream(blocks, method, top_n, selection, chunk_bytes, rng=None, store=None):
    """Find the top-n contexts of every gene pair, streaming over contexts in memory-bounded chunks.

    Holding all K (G, G) matrices at once would need K * G^2 * 4 bytes; instead contexts are
    processed in chunks, and after each chunk only the running best n per pair are kept.

    :param rng: if given, each gene's observations are permuted independently within each context
        (the null: no gene-gene association, gene marginals preserved). Used for calibration.
    :param store: optional (K, G, G) array; the signed t of every context is written into it.
    :return: (top_key, top_idx, top_signed), each (G, G, n) and sorted best first. top_key is the
        ranking statistic, top_idx the context ids, top_signed the signed t of the selected contexts
        for selection='absolute' (None otherwise, where it equals top_key).
    """
    K = len(blocks)
    G = blocks[0].X.shape[1]
    absolute = selection == "absolute"

    # running best n per pair; -inf / -1 are placeholders that any real context beats
    top_key = np.full((G, G, top_n), -np.inf, dtype=np.float32)
    top_idx = np.full((G, G, top_n), -1, dtype=np.int16)
    top_sgn = np.zeros((G, G, top_n), dtype=np.float32) if absolute else None

    per_chunk = max(1, int(chunk_bytes // (G * G * 4)))  # contexts whose float32 matrices fit a chunk
    for start in range(0, K, per_chunk):
        ks = list(range(start, min(K, start + per_chunk)))
        keys = np.empty((G, G, len(ks)), dtype=np.float32)
        sgn = np.empty((G, G, len(ks)), dtype=np.float32) if absolute else None
        for j, k in enumerate(ks):
            X = blocks[k].X
            if rng is not None:
                X = rng.permuted(X, axis=0)
            t = standardized_z(X, method)
            if store is not None:
                store[k] = t
            keys[..., j] = directional(t, selection)
            if absolute:
                sgn[..., j] = t

        # merge: candidates = previous best n + this chunk; keep the n largest (unordered)
        cand_key = np.concatenate([top_key, keys], axis=-1)
        cand_idx = np.concatenate(
            [top_idx, np.broadcast_to(np.asarray(ks, dtype=np.int16), (G, G, len(ks)))], axis=-1)
        keep = np.argpartition(-cand_key, top_n - 1, axis=-1)[..., :top_n]
        top_key = np.take_along_axis(cand_key, keep, axis=-1)
        top_idx = np.take_along_axis(cand_idx, keep, axis=-1)
        if absolute:
            top_sgn = np.take_along_axis(np.concatenate([top_sgn, sgn], axis=-1), keep, axis=-1)

    # sort the kept n best first
    order = np.argsort(-top_key, axis=-1, kind="stable")

    def sort_best_first(a):
        return None if a is None else np.take_along_axis(a, order, axis=-1)

    return sort_best_first(top_key), sort_best_first(top_idx), sort_best_first(top_sgn)


def _summarize(top_key, top_idx, n_obs, weighting):
    """Summary Fisher z per pair from the selected t: weighted mean of z_c = t_c / sqrt(n_c - 3).

    weighting='n' uses weights n_c - 3 (inverse variance of z_c, i.e. the pooled Fisher z estimate);
    'equal' gives each selected context the same weight.
    """
    scale = z_scale(n_obs)[top_idx]  # sqrt(n_c - 3) of each selected context
    z_selected = top_key / scale
    w = scale ** 2 if weighting == "n" else np.ones_like(scale)
    return (w * z_selected).sum(axis=-1) / w.sum(axis=-1)


def _analytic_null(n_obs, top_n, selection, weighting, n_sim, rng, chunk=20_000):
    """Mean and SD of the summary under the null, by Monte-Carlo.

    Assumes independent contexts and t_c ~ N(0, 1) (valid for centred methods). Picking the best n
    of K such draws gives a positive mean, which is the selection bias subtracted from z_raw.
    Simulated in chunks of `chunk` draws so memory stays ~chunk x K x 8 bytes (16 MB per 100 contexts).
    """
    K = len(n_obs)
    total, total_sq, done = 0.0, 0.0, 0
    while done < n_sim:
        m = min(chunk, n_sim - done)
        t = rng.standard_normal((m, K))
        if selection == "absolute":
            np.abs(t, out=t)
        idx = np.argpartition(-t, top_n - 1, axis=1)[:, :top_n]  # best n per draw
        zsum = _summarize(np.take_along_axis(t, idx, axis=1), idx, n_obs, weighting)
        total += zsum.sum()
        total_sq += (zsum ** 2).sum()
        done += m
    mean = total / n_sim
    return float(mean), float(np.sqrt(max(total_sq / n_sim - mean ** 2, 0.0)))


def _permutation_null(blocks, method, top_n, selection, weighting, n_perm, chunk_bytes, rng):
    """Null summary by permuting each gene within each context, n_perm times.

    Returns (mean, null_sd): mean is the (G, G) mean null summary per pair, null_sd the average
    per-pair SD (off-diagonal).
    """
    n_obs = np.array([b.n_obs for b in blocks])
    acc = acc_sq = 0.0
    for _ in range(n_perm):
        key, idx, _ = _top_n_stream(blocks, method, top_n, selection, chunk_bytes, rng=rng)
        zsum = _summarize(key, idx, n_obs, weighting)
        acc = acc + zsum
        acc_sq = acc_sq + zsum ** 2
    mean = acc / n_perm
    off = ~np.eye(len(mean), dtype=bool)
    null_sd = float(np.sqrt(np.maximum(acc_sq[off] / n_perm - mean[off] ** 2, 0)).mean())
    return mean, null_sd


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
    :param n_sim: Monte-Carlo draws for calibrate='analytic'.
    :param chunk_mb: memory budget for the per-context matrices processed at once.
    :param store_context_t: keep the full (K, G, G) per-context statistics (float32) for soft context
        weights and module splitting. 'auto' stores them if they need <= store_max_mb.
    :return: ContextSupport.
    """
    # --- validate ---
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
    if calibrate == "permutation" and int(n_perm) < 1:
        raise ValueError("n_perm must be >= 1 for calibrate='permutation'")
    n_obs = np.array([b.n_obs for b in blocks])
    if n_obs.min() < 4:
        raise ValueError("every context needs >= 4 observations (Fisher z standard error 1/sqrt(n - 3))")

    chunk_bytes = chunk_mb * 2 ** 20
    rng = np.random.default_rng(seed)

    # --- optional storage of every context's statistics ---
    G = blocks[0].X.shape[1]
    store_mb = K * G * G * 4 / 2 ** 20
    if store_context_t == "auto":
        store_context_t = store_mb <= store_max_mb
        if not store_context_t:
            print(f"Not storing per-context statistics ({store_mb:.0f} MB > store_max_mb={store_max_mb}); "
                  "they will be recomputed for module genes when needed.")
    t_all = np.empty((K, G, G), dtype=np.float32) if store_context_t else None

    # --- top-n selection and summary ---
    top_key, top_idx, top_sgn = _top_n_stream(blocks, method, top_n, selection, chunk_bytes, store=t_all)
    z_raw = _summarize(top_key, top_idx, n_obs, weighting)

    # --- selection-bias calibration ---
    if calibrate == "none":
        bias, null_sd = 0.0, float("nan")
    elif calibrate == "analytic":
        if method == "cosine":
            print("Warning: analytic calibration assumes a zero null mean, which does not hold for "
                  "cosine similarity; consider calibrate='permutation'.")
        bias, null_sd = _analytic_null(n_obs, top_n, selection, weighting, n_sim, rng)
    else:
        null_mean, null_sd = _permutation_null(blocks, method, top_n, selection, weighting, n_perm,
                                               chunk_bytes, rng)
        # cosine's null mean differs per pair (it depends on the genes' mean FC): keep it per pair;
        # for centred methods the null is the same for all pairs, so pool it (less noisy)
        off = ~np.eye(G, dtype=bool)
        bias = null_mean if method == "cosine" else float(null_mean[off].mean())

    z = z_raw - bias
    if selection == "absolute":
        z = np.maximum(z, 0.0)  # |t| based: a summary below its null mean means no support
    np.fill_diagonal(z, np.arctanh(1 - 1e-7))  # self-similarity 1 (Fisher z of the clip value)

    return ContextSupport(
        genes=pd.Index(genes) if genes is not None else pd.RangeIndex(G),
        contexts=[b.name for b in blocks], n_obs=n_obs, top_n=top_n, selection=selection,
        weighting=weighting, method=method, top_idx=top_idx, top_t=top_key,
        z_raw=z_raw, z=z, calibrate=calibrate, bias=bias, null_sd=null_sd, t_all=t_all,
        top_t_signed=top_sgn,
    )
