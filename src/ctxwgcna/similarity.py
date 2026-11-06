"""Gene-gene similarity within one context.

All functions take X with shape (n_obs, n_genes) and return a symmetric (n_genes, n_genes) matrix.
"""
import numpy as np
from scipy.stats import rankdata

METHODS = ("pearson", "spearman", "bicor", "cosine")
_EPS = 1e-7


def _unit_columns(X):
    norm = np.linalg.norm(X, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Xn = X / norm
    # Genes with zero variance (or an all-zero FC vector for cosine) get similarity 0 with everything.
    Xn[:, norm == 0] = 0.0
    return Xn


def _bicor_transform(X):
    """Biweight midcorrelation weighting (as in WGCNA::bicor with maxPOutliers=1)."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    Xc = X - med
    with np.errstate(invalid="ignore", divide="ignore"):
        u = Xc / (9.0 * mad)
    w = (1.0 - u ** 2) ** 2 * (np.abs(u) < 1)
    Xt = Xc * w
    # WGCNA falls back to Pearson for genes whose MAD is zero.
    zero = mad == 0
    if zero.any():
        Xt[:, zero] = X[:, zero] - X[:, zero].mean(axis=0)
    return Xt


def similarity_matrix(X, method="pearson"):
    """Similarity between columns of X.

    pearson / spearman / bicor are centred correlations (co-variation around the context mean).
    cosine is uncentred: it also rewards genes that move in the same direction relative to FC = 0,
    e.g. both up-regulated throughout a system.
    """
    X = np.asarray(X, dtype=np.float64)
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
    S = (S + S.T) / 2.0  # exact symmetry, so ranking of (i,j) and (j,i) is identical
    np.clip(S, -1.0, 1.0, out=S)
    np.fill_diagonal(S, 1.0)
    return S


def fisher_z(S):
    return np.arctanh(np.clip(S, -1.0 + _EPS, 1.0 - _EPS))
