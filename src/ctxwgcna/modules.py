"""Step 5b: module detection and context-aware module summaries."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .similarity import similarity_matrix

UNASSIGNED = "M0"


def detect_modules(tom, genes, min_module_size=30, deep_split=2, **cutree_kwargs):
    """Average-linkage clustering of 1 - TOM + dynamic hybrid tree cut.

    Uses the PyPI `dynamicTreeCut` port with pamStage=False, which reproduced R's
    dynamicTreeCut::cutreeDynamic(method='hybrid', pamStage=FALSE) exactly in our check.
    PyWGCNA's own cutreeHybrid is NOT used: on the same distance matrix it disagreed with R
    (ARI 0.54) and absorbed noise genes into modules. The PyPI port's PAM stage crashes
    (NameError), so PAM is off.

    Returns (labels, linkage): labels is a Series gene -> 'M1', 'M2', ... (M1 largest);
    'M0' = unassigned (WGCNA's grey).
    """
    from dynamicTreeCut import cutreeHybrid

    diss = 1.0 - np.asarray(tom, dtype=np.float64)  # no rounding: TOM can be ~1e-6 with weak signal
    np.fill_diagonal(diss, 0.0)
    Z = linkage(squareform(diss, checks=False), method="average")
    cutree_kwargs.setdefault("pamStage", False)
    cutree_kwargs.setdefault("verbose", 0)
    res = cutreeHybrid(Z, diss, minClusterSize=min_module_size, deepSplit=deep_split, **cutree_kwargs)
    num = np.asarray(res["labels"], dtype=int)
    return pd.Series([f"M{v}" for v in num], index=pd.Index(genes), name="module"), Z


def _module_index(modules):
    mods = [m for m in pd.unique(modules) if m != UNASSIGNED]
    mods.sort(key=lambda m: int(m[1:]))
    return {m: np.flatnonzero((modules == m).to_numpy()) for m in mods}


def module_context_profile(adj, support, modules, min_t=None):
    """Which contexts support each module's internal edges.

    For module m and context k: adjacency-weighted share of the within-module support carried by k,
    where each edge's top-n supporting contexts count 1 each (only those with t >= min_t if min_t is
    given). Rows are normalised to sum to 1 (all-zero rows, i.e. no qualifying support, stay 0).
    Compare with the uniform expectation 1/K.
    """
    A = np.array(adj, dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    idx = _module_index(modules)
    out = pd.DataFrame(0.0, index=list(idx), columns=support.contexts)
    for k, name in enumerate(support.contexts):
        I = support.indicator(k, min_t)
        for m, g in idx.items():
            sub = A[np.ix_(g, g)]
            out.loc[m, name] = (sub * I[np.ix_(g, g)]).sum()
    tot = out.sum(axis=1)
    return out.div(tot.where(tot > 0, 1.0), axis=0)


def module_context_coherence(blocks, modules, method="pearson"):
    """Mean within-module similarity in each context (independent of the top-n selection)."""
    idx = _module_index(modules)
    out = pd.DataFrame(np.nan, index=list(idx), columns=[b.name for b in blocks])
    for b in blocks:
        S = similarity_matrix(b.X, method)
        for m, g in idx.items():
            sub = S[np.ix_(g, g)]
            n = len(g)
            out.loc[m, b.name] = (sub.sum() - n) / (n * (n - 1))
    return out


def eigengene(X, center=True):
    """First PC of the (standardized) module genes, sign-aligned with their average profile.

    Returns (eigengene values per observation, fraction of variance explained).
    """
    X = np.asarray(X, dtype=np.float64)
    if center:
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - X.mean(axis=0)) / sd
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    me = U[:, 0] * s[0]
    if np.corrcoef(me, X.mean(axis=1))[0, 1] < 0:
        me = -me
    return me / np.linalg.norm(me) * np.sqrt(len(me)), float(s[0] ** 2 / (s ** 2).sum())
