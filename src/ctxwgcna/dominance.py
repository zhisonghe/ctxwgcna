"""Detect dominant observations: LFC profiles (e.g. one condition in one system) with far larger
changes than the rest of their context, which would otherwise dominate the correlations.

Why remove them rather than use a robust correlation: in simulations, Spearman/bicor neutralised a
dominant condition but also erased a genuine programme active in only 5 of 300 conditions; Pearson
with the dominant rows removed recovered both (see NOTES.md). Flagged rows should be analysed
separately (e.g. their responder genes mapped onto the modules).

A row is flagged only if it is both much LARGER (sum of squared LFC far above its context's median,
and a robust outlier) and BROAD (a substantial fraction of all genes changes extremely in it). Breadth is
what makes a row harmful: one row in which most genes move creates a giant correlated block in Pearson
correlations. A large row confined to a few dozen genes is how a genuine few-condition programme looks
(in simulation such rows reached 7x the median magnitude but moved only 6% of genes) and must be kept.

Each LFC profile (row) is compared with the other rows of its own context (e.g. the other conditions
of the same system), so dominance can be system-specific. Several dominant rows are allowed, but they
are expected to be a small minority (max_frac, default 5% of a context's rows): the median/MAD
reference is only valid then, and widespread "dominance" is a continuum, not a few outliers.

Magnitude is measured as distance from 0, so this assumes log FC (log2 or ln; 0 = no change), not
linear FC. For an expression matrix, centre each gene within its context first.
"""
import math
import warnings

import numpy as np
import pandas as pd

from .data import as_frames, check_unique, is_anndata, prepare_values, report_preparation, warn_if_linear_fc

MAD_TO_SD = 1.4826  # MAD * 1.4826 estimates the SD of normally distributed data


def _robust_gene_z(X):
    """Per-gene robust z within a block (rows = observations): (x - median) / (1.4826 * MAD).

    Genes with MAD = 0 (more than half the rows identical, e.g. exact zeros in sparse LFC) fall back to
    the leave-one-out SD: each row is scaled by the SD of the OTHER rows. Including the row itself
    would cap |z| at (n-1)/sqrt(n) (4.3 for n = 20), so a gene that changes only in the scored row
    could never count as extreme in a small context. A gene flat in all other rows but changed in this
    one gets z = +-inf (extreme).

    NaN entries (nan='skip') are ignored in the medians and sums and get z = NaN (never extreme).
    """
    valid = ~np.isnan(X)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN gene columns: nanmedian warns
        med = np.nanmedian(X, axis=0)
        mad = MAD_TO_SD * np.nanmedian(np.abs(X - med), axis=0)
        Z = np.where(mad > 0, (X - med) / mad, 0.0)
        flat = mad == 0
        if flat.any():
            Y = np.where(valid[:, flat], X[:, flat], 0.0)
            n = valid[:, flat].sum(axis=0)  # observed rows per gene (= n_rows without NaN)
            # leave-one-out mean and (unbiased) variance of each column, for every left-out row
            S, Q = Y.sum(axis=0), (Y ** 2).sum(axis=0)
            m_loo = (S - Y) / (n - 1)
            var_loo = np.maximum((Q - Y ** 2) / (n - 1) - m_loo ** 2, 0.0) * (n - 1) / (n - 2)
            sd_loo = np.where(n > 2, np.sqrt(var_loo), 0.0)
            dev = Y - med[flat]
            tiny = 1e-12 * max(1.0, np.abs(Y).max())  # rounding noise of the sums
            z_flat = np.where(sd_loo > tiny, dev / np.maximum(sd_loo, tiny),
                              np.where(np.abs(dev) > tiny, np.sign(dev) * np.inf, 0.0))
            Z[:, flat] = np.where(n > 2, z_flat, 0.0)  # <= 2 observed rows: no spread to compare with
    Z[~valid] = np.nan
    return Z


def _magnitude_scores(X):
    """Per row: sum of squares, fold over the context median, and robust z of log2(fold).

    With NaN entries (nan='skip') the sum of squares is taken over the observed genes and scaled up
    to all genes (mean square x n_genes), so rows with a few missing genes stay comparable.
    """
    n_valid = (~np.isnan(X)).sum(axis=1)
    ss = np.nansum(X ** 2, axis=1) * (X.shape[1] / n_valid)
    ref = np.median(ss)
    if ref == 0:
        # more than half the rows have no change at all: any changed row is infinitely larger than
        # the typical one; breadth (min_gene_frac) and the cap decide which are dominant
        fold = np.where(ss > 0, np.inf, 0.0)
        return ss, fold, fold.copy()
    fold = ss / ref
    log_fold = np.log2(np.maximum(fold, 1e-12))
    centred = log_fold - np.median(log_fold)
    mad = MAD_TO_SD * np.median(np.abs(centred))
    mad = mad if mad > 0 else max(np.std(log_fold), 1e-12)
    return ss, fold, centred / mad


def _score_context(X, min_fold, min_z, min_gene_frac, extreme_z, max_frac):
    """Score the rows of one context. Returns (columns dict, n_candidates, cap)."""
    ss, fold, robust_z = _magnitude_scores(X)
    n_extreme = (np.abs(_robust_gene_z(X)) >= extreme_z).sum(axis=1)  # breadth
    frac_extreme = n_extreme / (~np.isnan(X)).sum(axis=1)  # of the observed genes
    candidate = (fold >= min_fold) & (robust_z >= min_z) & (frac_extreme >= min_gene_frac)

    # cap: flag at most max_frac of the rows (at least 1), the largest folds first
    cap = max(1, math.floor(max_frac * X.shape[0]))
    dominant = candidate.copy()
    if candidate.sum() > cap:
        strongest = np.argsort(-np.where(candidate, fold, -np.inf))[:cap]
        dominant[:] = False
        dominant[strongest] = True
    cols = {"ss": ss, "fold": fold, "robust_z": robust_z, "n_extreme": n_extreme,
            "frac_extreme": frac_extreme, "candidate": candidate, "dominant": dominant}
    return cols, int(candidate.sum()), cap


def _summarise_by_label(rows, min_gene_frac):
    """Per label (e.g. condition): in how many contexts it is dominant, how large and how broad it is.

    Breadth (frac_extreme) is summarised more fully than size: it tracks the number of responding
    genes, and n_broad counts the contexts where it passes the breadth criterion on its own.
    """
    g = rows.groupby("label", sort=False)
    fe = g["frac_extreme"]
    summary = pd.DataFrame({
        "n_contexts": g["fold"].count(), "n_dominant": g["dominant"].sum().astype(int),
        "n_broad": (rows["frac_extreme"] >= min_gene_frac).groupby(rows["label"], sort=False).sum().astype(int),
        "max_fold": g["fold"].max(), "median_fold": g["fold"].median(),
        "max_frac_extreme": fe.max(), "median_frac_extreme": fe.median(),
        "mean_frac_extreme": fe.mean(), "min_frac_extreme": fe.min()})
    summary.insert(2, "frac_dominant", summary["n_dominant"] / summary["n_contexts"])
    return summary.sort_values(["n_dominant", "max_fold"], ascending=False)


def detect_dominant(lfc, obs=None, context_col="system", label_col="condition", min_fold=5.0, min_z=4.0,
                    min_gene_frac=0.10, extreme_z=5.0, max_frac=0.05, nan="raise", clip=None, max_abs="auto",
                    verbose=True,
                    layer=None, genes=None):
    """Flag dominant LFC profiles.

    :param lfc: log fold changes (log2 or ln; 0 = no change): DataFrame, rows = observations
        (LFC profiles), columns = genes; or AnnData.
    :param obs: DataFrame with the same index, holding `context_col` (and optionally `label_col`).
        Defaults to adata.obs for AnnData input.
    :param layer, genes: AnnData layer and gene subset, as in ContextWGCNA.
    :param context_col: rows are compared within groups of this column (e.g. each system).
    :param label_col: column used to summarise across contexts (e.g. condition); may be None.
    :param min_fold: magnitude: the row's sum of squared LFC must be >= min_fold x the context median.
    :param min_z: ...and a robust outlier on the log scale: (log2 fold - median) / (1.4826 MAD) >= min_z,
        so that in contexts where large spreads are normal, min_fold alone does not flag rows.
    :param min_gene_frac: breadth: at least this fraction of genes must be extreme in the row, i.e.
        |robust z| >= extreme_z relative to that gene's LFC distribution within the context.
    :param max_frac: at most this fraction of a context's rows (at least 1) is flagged. If more rows
        pass, the strongest are flagged, the rest are reported as candidates, and a warning is printed:
        dominance that widespread is not a few outliers, and removal may not be appropriate.
    :param nan: how to treat missing values. 'raise' (default): error. 'skip': rows that are entirely
        NaN (e.g. untested combinations) are not scored (metrics NaN, never dominant); other NaN
        entries are ignored (sum of squares scaled to all genes, frac_extreme over observed genes).
        A number, e.g. 0: fill NaN with it (0 = "no change").
    :param max_abs: treat |LFC| > max_abs as missing measurements, e.g. the |LFC| ~ 20-35 a GLM
        returns when one group has zero counts. 'auto' (default): cutoff from the gap below such a
        separate mode of large |LFC|; nothing is masked if there is none (see
        ctxwgcna.data.auto_max_abs). A number: fixed cutoff (e.g. 15). None: no masking. Masked values are ignored per entry like NaN under
        'skip' (or filled with nan if it is a number) and never trigger nan='raise'. Applied before
        clipping, so a 'pNN' clip is computed on the unmasked values.
    :param clip: limit extreme FCs before scoring, so that a few unstable values (e.g. capped at
        +-10) do not dominate the sum of squares. None (default): no clipping. c: clip to [-c, c].
        (lo, hi): clip to [lo, hi]. 'p95' (any percentile 'pNN'): clip to +- that percentile of |LFC|
        over all observed values. Applies to the scoring only; lfc itself is not changed.

    Returns (rows, summary):
      rows     one row per LFC profile: context, label, ss, fold (ss / context median), robust_z,
               n_extreme, frac_extreme, candidate (passes all thresholds), dominant (flagged after cap)
      rows.attrs['preprocessing'] records the NaN rule and clipping used (bounds, fraction clipped).
      summary  per label (if label_col given): n_contexts, n_dominant, frac_dominant, n_broad
               (contexts with frac_extreme >= min_gene_frac), max_fold, median_fold, and the max /
               median / mean / min of frac_extreme; sorted by n_dominant then max_fold
    """
    lfc, obs = as_frames(lfc, obs, layer=layer, genes=genes)
    if obs is None:
        raise ValueError("obs (the observation table) is required")
    check_unique(lfc, obs)
    for col in (context_col, label_col):
        if col is not None and col not in obs.columns:
            raise ValueError(f"{col!r} is not a column of obs (pass label_col=None to skip the summary)")
    if not lfc.index.equals(obs.index):
        obs = obs.loc[lfc.index]
    # NaN rule (a NaN left in would make its context's median magnitude NaN and flag nothing there)
    # and optional clipping of extreme values
    masked_as = nan if not isinstance(nan, str) else "nan"  # masked entries: ignored, or filled like NaN
    X_all, scored, info = prepare_values(lfc.to_numpy(dtype=np.float64), lfc.index, nan, clip,
                                         max_abs=max_abs, masked_as=masked_as)
    if verbose:
        warn_if_linear_fc(X_all, "detect_dominant")
    ctx = obs[context_col].to_numpy()

    parts, capped = [], []  # capped: (context, n_candidates, cap) where the cap was hit
    for c in pd.unique(ctx):
        in_context = (ctx == c) & scored
        unscored = (ctx == c) & ~scored  # all-NaN rows with nan='skip'
        if in_context.any():
            cols, n_cand, cap = _score_context(X_all[in_context], min_fold, min_z, min_gene_frac,
                                               extreme_z, max_frac)
            if n_cand > cap:
                capped.append((c, n_cand, cap))
            part = pd.DataFrame({"context": c, **cols}, index=lfc.index[in_context])
        else:
            part = pd.DataFrame(index=lfc.index[:0])
        if unscored.any():
            blank = pd.DataFrame({"context": c, "ss": np.nan, "fold": np.nan, "robust_z": np.nan,
                                  "n_extreme": 0, "frac_extreme": np.nan, "candidate": False,
                                  "dominant": False}, index=lfc.index[unscored])
            part = blank if part.empty else pd.concat([part, blank])
        if label_col is not None:
            part.insert(1, "label", obs.loc[part.index, label_col].to_numpy())
        parts.append(part)
    rows = pd.concat(parts).loc[lfc.index]  # back to the input row order
    rows.attrs["preprocessing"] = info
    summary = _summarise_by_label(rows, min_gene_frac) if label_col is not None else None

    if verbose:
        report_preparation(info, "" if nan == "raise" else "not scored / ignored per entry" if nan == "skip"
                           else f"filled with {nan}")
        n_dom = int(rows["dominant"].sum())
        print(f"Dominance (min_fold={min_fold}, min_z={min_z}, min_gene_frac={min_gene_frac}, "
              f"max_frac={max_frac}): {n_dom} of {len(rows)} LFC profiles flagged in "
              f"{rows.loc[rows.dominant, 'context'].nunique()} of {rows['context'].nunique()} contexts.")
        if summary is not None and n_dom:
            top = summary[summary.n_dominant > 0]
            print("  flagged labels (n contexts): "
                  + ", ".join(f"{k} ({v})" for k, v in top["n_dominant"].items()))
        if capped:
            print(f"WARNING: in {len(capped)} context(s) more than max_frac={max_frac} of the rows passed "
                  f"the thresholds (e.g. {capped[0][0]}: {capped[0][1]} candidates, cap {capped[0][2]}); "
                  "only the strongest were flagged. Widespread dominance is not a few outliers: inspect "
                  "'rows', raise min_fold, or reconsider removal.")
    return rows, summary


def drop_dominant(data, obs=None, rows=None):
    """Remove the observations flagged as dominant in `rows` (from detect_dominant).

    Works on any data with the same rows (LFC or expression).
    DataFrame input: drop_dominant(data, obs, rows) returns (data, obs) without them, in data's row order.
    AnnData input:   drop_dominant(adata, rows=rows) returns a filtered copy of the AnnData.
    """
    if rows is None:
        raise ValueError("pass rows=, the first table returned by detect_dominant")
    if is_anndata(data):
        dominant = rows["dominant"].reindex(data.obs_names, fill_value=False).to_numpy(dtype=bool)
        return data[~dominant].copy()
    kept = data.index[~rows.loc[data.index, "dominant"].to_numpy()]
    return data.loc[kept], obs.loc[kept]
