"""Step 5a: adjacency, soft-threshold selection, and the context-restricted TOM."""
import numpy as np
import pandas as pd

NETWORK_TYPES = ("signed", "signed hybrid", "unsigned")
TOM_MODES = ("standard", "shared", "matched")


def adjacency_from_r(r, power, network_type="signed"):
    if network_type == "signed":
        a = ((1.0 + r) / 2.0) ** power
    elif network_type == "signed hybrid":
        a = np.where(r > 0, r, 0.0) ** power
    elif network_type == "unsigned":
        a = np.abs(r) ** power
    else:
        raise ValueError(f"network_type must be one of {NETWORK_TYPES}")
    a = np.asarray(a, dtype=np.float64)
    np.fill_diagonal(a, 1.0)
    return a


def _ols(y, X):
    """Least squares with intercept; returns (coefficients, R^2, adjusted R^2)."""
    X = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss_res = np.sum((y - X @ coef) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n, p = X.shape
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if n > p else np.nan
    return coef, r2, adj


def scale_free_fit_index(k, n_breaks=10):
    """Scale-free topology fit of connectivities k (port of R WGCNA::scaleFreeFitIndex).

    k is binned into n_breaks equal-width bins (right-closed, as R cut()); log10 p(k) is regressed on
    log10 of the mean k per bin. Empty bins use the bin midpoint and p = 0 (log10(1e-9)), as in R.
    Returns dict with Rsquared.SFT, slope.SFT, truncatedExponentialAdjRsquared.
    """
    k = np.asarray(k, dtype=np.float64)
    lo, hi = k.min(), k.max()
    if hi <= lo:
        return {"Rsquared.SFT": np.nan, "slope.SFT": np.nan, "truncatedExponentialAdjRsquared": np.nan}
    edges = np.linspace(lo, hi, n_breaks + 1)
    mids = 0.5 * (edges[1:] + edges[:-1])
    # right-closed bins (a, b], first bin includes the minimum
    b = np.clip(np.searchsorted(edges, k, side="left") - 1, 0, n_breaks - 1)
    counts = np.bincount(b, minlength=n_breaks)
    sums = np.bincount(b, weights=k, minlength=n_breaks)
    with np.errstate(invalid="ignore", divide="ignore"):
        dk = np.where(counts > 0, sums / counts, mids)
    dk = np.where(dk == 0, mids, dk)
    log_dk = np.log10(dk)
    log_p = np.log10(counts / len(k) + 1e-9)
    coef1, r2, _ = _ols(log_p, log_dk)
    _, _, adj2 = _ols(log_p, np.column_stack([log_dk, dk]))
    return {"Rsquared.SFT": r2, "slope.SFT": coef1[1], "truncatedExponentialAdjRsquared": adj2}


def scale_free_table(r, powers, network_type="signed", n_breaks=10):
    rows = []
    for p in powers:
        a = adjacency_from_r(r, p, network_type)
        k = a.sum(axis=0) - 1.0
        fit = scale_free_fit_index(k, n_breaks=n_breaks)
        r2, slope = float(fit["Rsquared.SFT"]), float(fit["slope.SFT"])
        rows.append({"power": p, "SFT.R.sq": r2, "slope": slope,
                     "signed R.sq": -np.sign(slope) * r2,
                     "truncated R.sq": float(fit["truncatedExponentialAdjRsquared"]),
                     "mean(k)": k.mean(), "median(k)": np.median(k), "max(k)": k.max()})
    return pd.DataFrame(rows)


def pick_power(table, r2_cut=0.85, mean_cut=100):
    """Lowest power with signed R^2 >= r2_cut and mean(k) <= mean_cut, else the best-fitting one."""
    ok = table[(table["signed R.sq"] >= r2_cut) & (table["mean(k)"] <= mean_cut)]
    if len(ok):
        return int(ok["power"].min()), True
    fit = table["signed R.sq"].dropna()
    if fit.empty:
        raise ValueError("the scale-free fit failed for every power (e.g. constant connectivity); "
                         "set the power explicitly")
    return int(table.loc[fit.idxmax(), "power"]), False


def context_tom(adj, support=None, mode="matched", min_t=None):
    """Topological overlap with optional context restriction.

        TOM_ij = (L_ij + a_ij) / (min(k_i, k_j) + 1 - a_ij)

    standard: L_ij = sum_u a_iu a_uj                       (WGCNA, TOMDenom='min')
    shared:   L_ij = sum_u a_iu a_uj |C_iu & C_uj| / n     (i-u and u-j share supporting contexts)
    matched:  L_ij = sum_u a_iu a_uj |C_ij & C_iu & C_uj| / n
              (shared neighbour must be linked to i and j through a context that also supports i-j)
    where C_xy is the set of top-n supporting contexts of pair (x, y). Computed as K matrix
    products: sum_k (A o I_k)(A o I_k) [o I_k for matched]. Since L shrinks relative to 'standard',
    TOM stays in [0, 1].

    :param min_t: optionally count a context as supporting only if its ranking statistic >= min_t.
    """
    if mode not in TOM_MODES:
        raise ValueError(f"mode must be one of {TOM_MODES}")
    A = np.array(adj, dtype=np.float64)
    np.fill_diagonal(A, 0.0)

    if mode == "standard":
        L = A @ A
    else:
        if support is None:
            raise ValueError(f"mode={mode!r} needs a ContextSupport")
        L = np.zeros_like(A)
        for k in np.unique(support.top_idx):
            I = support.indicator(k, min_t)
            np.fill_diagonal(I, False)
            if not I.any():
                continue
            M = A * I
            P = M @ M
            L += P if mode == "shared" else I * P
        L /= support.top_n

    k = A.sum(axis=1)
    tom = (L + A) / (np.minimum.outer(k, k) + 1.0 - A)
    np.fill_diagonal(tom, 1.0)
    return tom
