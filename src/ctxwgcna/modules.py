"""Step 5b: module detection and context-aware module summaries."""
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .similarity import similarity_matrix

UNASSIGNED = "M0"  # label of genes in no module (WGCNA's "grey")


def _cutree_hybrid():
    """dynamicTreeCut.cutreeHybrid, made compatible with current scipy and Python.

    dynamicTreeCut 0.1.1 (unmaintained) uses scipy.stats.rankdata(..., method='ordinal') results as
    array indices. scipy >= 1.18 returns them as floats, which numpy rejects as indices (IndexError);
    the ordinal ranks are whole numbers, so casting them back to int restores the old behaviour
    exactly. The patch is applied once, to dynamicTreeCut's own module namespace only.
    """
    import dynamicTreeCut.dynamicTreeCut as dtc
    from scipy.stats import rankdata

    if not getattr(dtc, "_ctxwgcna_patched", False):
        def rankdata_int(a, method="average", **kwargs):
            ranks = rankdata(a, method=method, **kwargs)
            return ranks.astype(np.int64) if method == "ordinal" else ranks

        dtc.rankdata = rankdata_int
        dtc._ctxwgcna_patched = True
    return dtc.cutreeHybrid


def detect_modules(tom, genes, min_module_size=30, deep_split=2, **cutree_kwargs):
    """Average-linkage clustering of 1 - TOM + dynamic hybrid tree cut.

    Uses the PyPI `dynamicTreeCut` port with pamStage=False, which reproduced R's
    dynamicTreeCut::cutreeDynamic(method='hybrid', pamStage=FALSE) exactly in our check.
    PyWGCNA's own cutreeHybrid is NOT used: on the same distance matrix it disagreed with R
    (ARI 0.54) and absorbed noise genes into modules. The PyPI port's PAM stage crashes
    (NameError), so PAM is off.

    Returns (labels, linkage): labels is a Series gene -> 'M1', 'M2', ... (M1 largest);
    'M0' = unassigned.
    """
    cutreeHybrid = _cutree_hybrid()

    diss = 1.0 - np.asarray(tom, dtype=np.float64)  # no rounding: TOM can be ~1e-6 with weak signal
    np.fill_diagonal(diss, 0.0)
    Z = linkage(squareform(diss, checks=False), method="average")
    cutree_kwargs.setdefault("pamStage", False)
    cutree_kwargs.setdefault("verbose", 0)
    with warnings.catch_warnings():
        # dynamicTreeCut applies '~' to Python bools (deprecated in Python 3.12). The affected branch
        # only runs with external splits (minExternalSplit), which ctxwgcna does not use.
        warnings.filterwarnings("ignore", message="Bitwise inversion", category=DeprecationWarning)
        res = cutreeHybrid(Z, diss, minClusterSize=min_module_size, deepSplit=deep_split, **cutree_kwargs)
    numbers = np.asarray(res["labels"], dtype=int)  # 0 = unassigned, 1 = largest module, ...
    return pd.Series([f"M{v}" for v in numbers], index=pd.Index(genes), name="module"), Z


def module_indices(modules):
    """{module label: gene positions} for all assigned modules, in order M1, M2, ..."""
    names = [m for m in pd.unique(modules) if m != UNASSIGNED]
    names.sort(key=lambda m: int(m[1:]))
    return {m: np.flatnonzero((modules == m).to_numpy()) for m in names}


def _offdiag_adjacency(adj):
    A = np.array(adj, dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    return A


def module_context_profile(adj, support, modules, min_t=None):
    """Which contexts support each module's internal edges (from the top-n indicators).

    For module m and context k: adjacency-weighted share of the within-module support carried by k,
    where each edge's top-n supporting contexts count 1 each (only those with t >= min_t if min_t is
    given). Rows are normalised to sum to 1 (all-zero rows, i.e. no qualifying support, stay 0).
    Compare with the uniform expectation 1/K.
    """
    A = _offdiag_adjacency(adj)
    idx = module_indices(modules)
    out = pd.DataFrame(0.0, index=list(idx), columns=support.contexts)
    for k, name in enumerate(support.contexts):
        I = support.indicator(k, min_t)
        for m, g in idx.items():
            within = np.ix_(g, g)
            out.loc[m, name] = (A[within] * I[within]).sum()  # adjacency of edges supported by k
    total = out.sum(axis=1)
    return out.div(total.where(total > 0, 1.0), axis=0)


def module_context_profile_soft(adj, weights_fn, modules):
    """Adjacency-weighted soft context profile per module, from per-edge context weights.

    profile[m, k] = sum_{i<j in m} a_ij w_ij^k / sum_{i<j in m} a_ij   (rows sum to ~1)

    :param weights_fn: callable idx -> (K, m, m) soft context weights of the genes idx.
    """
    A = _offdiag_adjacency(adj)
    rows = {}
    for m, g in module_indices(modules).items():
        a = A[np.ix_(g, g)]
        w = weights_fn(g)
        rows[m] = (w * a[None]).sum(axis=(1, 2)) / a.sum()
    return pd.DataFrame(rows).T


def module_context_coherence(blocks, modules, method="pearson"):
    """Mean within-module similarity in each context (independent of the top-n selection)."""
    idx = module_indices(modules)
    out = pd.DataFrame(np.nan, index=list(idx), columns=[b.name for b in blocks])
    for b in blocks:
        S = similarity_matrix(b.X, method)
        for m, g in idx.items():
            n = len(g)
            # mean over off-diagonal pairs: subtract the n diagonal ones
            out.loc[m, b.name] = (S[np.ix_(g, g)].sum() - n) / (n * (n - 1))
    return out


def eigengene(X, center=True):
    """First PC of the (standardized) module genes, sign-aligned with their average profile.

    :param X: (n_obs, n_genes) values of the module genes.
    :param center: standardize each gene first (False for cosine, which is uncentred).
    Returns (eigengene values per observation scaled to norm sqrt(n_obs), fraction of variance
    explained).
    """
    X = np.asarray(X, dtype=np.float64)
    if center:
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - X.mean(axis=0)) / sd
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    me = U[:, 0] * s[0]
    if np.corrcoef(me, X.mean(axis=1))[0, 1] < 0:  # SVD sign is arbitrary
        me = -me
    return me / np.linalg.norm(me) * np.sqrt(len(me)), float(s[0] ** 2 / (s ** 2).sum())
