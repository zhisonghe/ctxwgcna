"""Gene-gene similarity within one context, and its Fisher-z / standardized forms.

All matrix functions take X with shape (n_obs, n_genes) and return a symmetric (n_genes, n_genes)
matrix. Similarities are always computed between genes (columns), across the observations of one
context (rows).
"""
import numpy as np
from scipy.stats import rankdata

METHODS = ("pearson", "spearman", "bicor", "cosine")

# Similarities are clipped to +-(1 - _EPS) before arctanh, so a perfect correlation gives a large
# but finite Fisher z (about 8.4) instead of inf.
_EPS = 1e-7


def _unit_columns(X):
    """Scale every column to unit length; all-zero columns stay zero (similarity 0 with everything)."""
    norm = np.linalg.norm(X, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Xn = X / norm
    Xn[:, norm == 0] = 0.0  # zero-variance genes (or an all-zero LFC vector for cosine)
    return Xn


def _bicor_transform(X):
    """Centre and down-weight outliers so that a dot product gives the biweight midcorrelation.

    Matches WGCNA::bicor with maxPOutliers=1: each value is weighted by (1 - u^2)^2 for |u| < 1,
    with u = (x - median) / (9 * MAD). Genes whose MAD is zero fall back to Pearson (plain centring),
    as WGCNA does.
    """
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    Xc = X - med
    with np.errstate(invalid="ignore", divide="ignore"):
        u = Xc / (9.0 * mad)
    weights = (1.0 - u ** 2) ** 2 * (np.abs(u) < 1)
    Xt = Xc * weights
    zero_mad = mad == 0
    if zero_mad.any():
        Xt[:, zero_mad] = X[:, zero_mad] - X[:, zero_mad].mean(axis=0)
    return Xt


def similarity_matrix(X, method="pearson"):
    """Similarity between the columns (genes) of X.

    pearson / spearman / bicor are centred correlations: co-variation around the context's mean.
    cosine is uncentred: it also rewards genes that move in the same direction relative to 0
    (e.g. both up-regulated throughout a system), which is only meaningful for log-FC data (0 = no change).
    """
    X = np.asarray(X, dtype=np.float64)
    # Each method is "transform the columns, then take the cosine of the transformed columns".
    if method == "pearson":
        Xt = X - X.mean(axis=0)
    elif method == "spearman":
        R = rankdata(X, axis=0)
        Xt = R - R.mean(axis=0)
    elif method == "bicor":
        Xt = _bicor_transform(X)
    elif method == "cosine":
        Xt = X
    else:
        raise ValueError(f"unknown similarity method {method!r}; choose from {METHODS}")
    Xn = _unit_columns(Xt)
    S = Xn.T @ Xn
    S = (S + S.T) / 2.0  # exact symmetry, so pairs (i, j) and (j, i) are ranked identically later
    np.clip(S, -1.0, 1.0, out=S)
    np.fill_diagonal(S, 1.0)
    return S


def fisher_z(S):
    """Fisher z-transform arctanh(S), with S clipped away from +-1."""
    return np.arctanh(np.clip(S, -1.0 + _EPS, 1.0 - _EPS))


def z_scale(n_obs):
    """sqrt(n - 3): 1 / standard error of a Fisher z from n observations (floored at 1).

    Works on a scalar or an array of context sizes. Contexts must have >= 4 observations (checked
    when the contexts are built); the floor only guards against misuse.
    """
    return np.sqrt(np.maximum(np.asarray(n_obs, dtype=np.float64) - 3.0, 1.0))


def standardized_z(X, method="pearson"):
    """Per-context statistic t = arctanh(s) * sqrt(n - 3) for all gene pairs of one context.

    Under no association t is approximately N(0, 1) whatever the context size n, so t is comparable
    across contexts with different numbers of observations. This is the quantity contexts are ranked
    by (the top-n selection) and the one thresholded in the submodule split.
    """
    return fisher_z(similarity_matrix(X, method)) * z_scale(X.shape[0])
