"""Detect dominant observations: LFC profiles (e.g. one condition in one system) with far larger
changes than the rest of their context, which would otherwise dominate the correlations.

Why remove them rather than use a robust correlation: in simulations, Spearman/bicor neutralised a
dominant condition but also erased a genuine programme active in only 5 of 300 conditions; Pearson
with the dominant rows removed recovered both (see NOTES.md). Flagged rows should be analysed
separately (e.g. their responder genes mapped onto the modules).

Terms used in the parameter and column names:
  magnitude        a row's sum of squared LFC over all genes (how much changes in total)
  magnitude_ratio  magnitude / median magnitude of the rows in the same context
  magnitude_z      robust z of log2(magnitude_ratio) among the rows of the context
  gene extreme     a gene whose LFC in the row is >= gene_extreme_z robust SDs from that gene's
                   median LFC in the context; frac_genes_extreme = their share of the row's genes

A row is flagged only if it is both much LARGER (magnitude far above its context's median, and a
robust outlier: magnitude_ratio and magnitude_z) and BROAD (frac_genes_extreme) (a substantial fraction of all genes changes extremely in it). Breadth is
what makes a row harmful: one row in which most genes move creates a giant correlated block in Pearson
correlations. A large row confined to a few dozen genes is how a genuine few-condition programme looks
(in simulation such rows reached 7x the median magnitude but moved only 6% of genes) and must be kept.

Each LFC profile (row) is compared with the other rows of its own context (e.g. the other conditions
of the same system), so dominance can be system-specific. Several dominant rows are allowed, but they
are expected to be a small minority (max_frac_flagged, default 5% of a context's rows): the median/MAD
reference is only valid then, and widespread "dominance" is a continuum, not a few outliers.

Magnitude is measured as distance from 0, so this assumes log FC (log2 or ln; 0 = no change), not
linear FC. For an expression matrix, centre each gene within its context first.
"""
import math
import warnings

import numpy as np
import pandas as pd

from .data import (MAD_TO_SD, as_frames, check_unique, flag_artifact_rows, is_anndata, prepare_values,
                   report_preparation, warn_if_linear_fc)



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
    """Per row: magnitude (sum of squared LFC), magnitude_ratio (over the context median) and
    magnitude_z (robust z of log2 magnitude_ratio).

    With NaN entries (nan='skip') the sum of squares is taken over the observed genes and scaled up
    to all genes (mean square x n_genes), so rows with a few missing genes stay comparable.
    """
    n_valid = (~np.isnan(X)).sum(axis=1)
    magnitude = np.nansum(X ** 2, axis=1) * (X.shape[1] / n_valid)
    ref = np.median(magnitude)
    if ref == 0:
        # more than half the rows have no change at all: any changed row is infinitely larger than
        # the typical one; breadth (min_frac_genes_extreme) and the cap decide which are dominant
        ratio = np.where(magnitude > 0, np.inf, 0.0)
        return magnitude, ratio, ratio.copy()
    ratio = magnitude / ref
    log_ratio = np.log2(np.maximum(ratio, 1e-12))
    centred = log_ratio - np.median(log_ratio)
    mad = MAD_TO_SD * np.median(np.abs(centred))
    mad = mad if mad > 0 else max(np.std(log_ratio), 1e-12)
    return magnitude, ratio, centred / mad


def _score_context(X, t):
    """Score the rows of one context (t: the thresholds). Returns (columns dict, n_candidates, cap)."""
    magnitude, ratio, mag_z = _magnitude_scores(X)
    n_extreme = (np.abs(_robust_gene_z(X)) >= t["gene_extreme_z"]).sum(axis=1)  # breadth
    frac_extreme = n_extreme / (~np.isnan(X)).sum(axis=1)  # of the observed genes
    candidate = ((ratio >= t["min_magnitude_ratio"]) & (mag_z >= t["min_magnitude_z"])
                 & (frac_extreme >= t["min_frac_genes_extreme"]))

    # cap: flag at most max_frac_flagged of the rows (at least 1), the largest magnitude_ratio first
    cap = max(1, math.floor(t["max_frac_flagged"] * X.shape[0]))
    dominant = candidate.copy()
    if candidate.sum() > cap:
        strongest = np.argsort(-np.where(candidate, ratio, -np.inf))[:cap]
        dominant[:] = False
        dominant[strongest] = True
    cols = {"magnitude": magnitude, "magnitude_ratio": ratio, "magnitude_z": mag_z,
            "n_genes_extreme": n_extreme, "frac_genes_extreme": frac_extreme,
            "candidate": candidate, "dominant": dominant}
    return cols, int(candidate.sum()), cap


def _summarise_by_condition(rows, min_frac_genes_extreme):
    """Per condition: in how many contexts it is dominant, how large and how broad it is.

    Breadth (frac_genes_extreme) is summarised more fully than size: it tracks the number of
    responding genes, and n_broad counts the contexts where it passes the breadth criterion on its own.
    """
    g = rows.groupby("condition", sort=False)
    fe = g["frac_genes_extreme"]
    summary = pd.DataFrame({
        "n_contexts": g["magnitude_ratio"].count(), "n_dominant": g["dominant"].sum().astype(int),
        "n_broad": (rows["frac_genes_extreme"] >= min_frac_genes_extreme)
        .groupby(rows["condition"], sort=False).sum().astype(int),
        "n_artifact_excluded": g["artifact_excluded"].sum().astype(int),
        "max_magnitude_ratio": g["magnitude_ratio"].max(), "median_magnitude_ratio": g["magnitude_ratio"].median(),
        "max_frac_genes_extreme": fe.max(), "median_frac_genes_extreme": fe.median(),
        "mean_frac_genes_extreme": fe.mean(), "min_frac_genes_extreme": fe.min(),
        "median_frac_masked": g["frac_masked"].median()})
    summary.insert(2, "frac_dominant", summary["n_dominant"] / summary["n_contexts"])
    return summary.sort_values(["n_dominant", "max_magnitude_ratio"], ascending=False)


# old parameter names (before 2026-11) -> current names; still accepted with a warning
RENAMED = {"label_col": "condition_col", "min_fold": "min_magnitude_ratio", "min_z": "min_magnitude_z",
           "min_gene_frac": "min_frac_genes_extreme", "extreme_z": "gene_extreme_z",
           "max_frac": "max_frac_flagged"}


def detect_dominant(lfc, obs=None, context_col="system", condition_col="condition", min_magnitude_ratio=5.0,
                    min_magnitude_z=4.0, min_frac_genes_extreme=0.10, gene_extreme_z=5.0, max_frac_flagged=0.05,
                    nan="raise", clip=None, max_abs="auto", max_artifact_z=None, verbose=True, layer=None,
                    genes=None, **old_names):
    """Flag dominant LFC profiles: rows that change far more, and in far more genes, than the other
    rows of their context.

    Input
    :param lfc: log fold changes (log2 or ln; 0 = no change): DataFrame, rows = observations
        (LFC profiles), columns = genes; or AnnData.
    :param obs: DataFrame with the same index, holding `context_col` (and optionally `condition_col`).
        Defaults to adata.obs for AnnData input.
    :param layer, genes: AnnData layer and gene subset, as in ContextWGCNA.
    :param context_col: rows are compared within groups of this column (e.g. each cell type).
    :param condition_col: column used to summarise across contexts (e.g. the perturbation); may be None.

    Thresholds (a row is a candidate if it passes all three)
    :param min_magnitude_ratio: the row's magnitude (sum of squared LFC) is >= this many times the
        median magnitude of its context.
    :param min_magnitude_z: ...and a robust outlier: (log2 magnitude_ratio - median) / (1.4826 MAD)
        >= this, so that in contexts where large spreads are normal the ratio alone flags nothing.
    :param min_frac_genes_extreme: ...and broad: at least this fraction of the row's genes is extreme.
    :param gene_extreme_z: a gene is extreme in a row if its LFC there is >= this many robust SDs
        (1.4826 MAD) from the gene's median LFC across the rows of the context.
    :param max_frac_flagged: at most this fraction of a context's rows (at least 1) is flagged. If
        more pass, the largest are flagged, the rest stay candidates, and a warning is printed:
        dominance that widespread is not a few outliers, and removal may not be appropriate.

    Data preparation (applied in this order: nan, max_abs, max_artifact_z, clip)
    :param nan: missing values. 'raise' (default): error. 'skip': rows that are entirely NaN (e.g.
        untested combinations) are not scored (metrics NaN, never dominant); other NaN entries are
        ignored (magnitude scaled to all genes, frac_genes_extreme over the observed genes). A number,
        e.g. 0: fill NaN with it (0 = "no change").
    :param max_abs: treat |LFC| > max_abs as missing measurements, e.g. the |LFC| ~ 20-35 a GLM
        returns when one group has zero counts. 'auto' (default): cutoff from the gap below such a
        separate mode of large |LFC|, if it is scattered over most rows; nothing is masked otherwise
        (see ctxwgcna.data.auto_max_abs). A number: fixed cutoff (e.g. 15). None: no masking. Masked
        values are ignored per entry like NaN under 'skip' (or filled with nan if it is a number) and
        never trigger nan='raise'.
    :param max_artifact_z: rows whose fraction of masked genes (frac_masked) is unusually high for
        their context (artifact_z > max_artifact_z) are not scored: their LFCs rest on very few cells,
        and their magnitude is inflated by noise. artifact_z is a robust z of log10(frac_masked)
        relative to the context median, with the spread pooled over all contexts. None (default):
        keep all rows. 3 is a reasonable value. Needs masking (max_abs) to have an effect.
    :param clip: limit extreme LFCs before scoring, so that a few unstable values (e.g. capped at
        +-10) do not dominate the magnitude. None (default): no clipping. c: clip to [-c, c].
        (lo, hi): clip to [lo, hi]. 'p95' (any 'pNN'): clip to +- that percentile of |LFC| over all
        observed, unmasked values. Applies to the scoring only; lfc itself is not changed.

    Returns (rows, summary):
      rows     one row per LFC profile: context, condition, magnitude, magnitude_ratio, magnitude_z,
               n_genes_extreme, frac_genes_extreme, candidate (passes all thresholds), dominant
               (flagged, after the cap), frac_masked (share of genes masked by max_abs), artifact_z,
               artifact_excluded (not scored because artifact_z > max_artifact_z). Rows that are not
               scored (all NaN, or artifact_excluded) have NaN metrics and are never dominant.
      rows.attrs['preprocessing'] records the data preparation (cutoffs, counts, fractions).
      summary  per condition (if condition_col is given): n_contexts (scored), n_dominant,
               frac_dominant, n_broad (contexts with frac_genes_extreme >= min_frac_genes_extreme),
               n_artifact_excluded, max / median magnitude_ratio, max / median / mean / min
               frac_genes_extreme, median_frac_masked; sorted by n_dominant, then max_magnitude_ratio.
    """
    t = {"min_magnitude_ratio": min_magnitude_ratio, "min_magnitude_z": min_magnitude_z,
         "min_frac_genes_extreme": min_frac_genes_extreme, "gene_extreme_z": gene_extreme_z,
         "max_frac_flagged": max_frac_flagged, "condition_col": condition_col}
    for old, value in old_names.items():
        if old not in RENAMED:
            raise TypeError(f"detect_dominant() got an unexpected keyword argument {old!r}")
        warnings.warn(f"detect_dominant: {old}= is now called {RENAMED[old]}=", FutureWarning, stacklevel=2)
        t[RENAMED[old]] = value
    condition_col = t.pop("condition_col")

    lfc, obs = as_frames(lfc, obs, layer=layer, genes=genes)
    if obs is None:
        raise ValueError("obs (the observation table) is required")
    check_unique(lfc, obs)
    for col in (context_col, condition_col):
        if col is not None and col not in obs.columns:
            raise ValueError(f"{col!r} is not a column of obs (pass condition_col=None to skip the summary)")
    if not lfc.index.equals(obs.index):
        obs = obs.loc[lfc.index]
    # NaN rule (a NaN left in would make its context's median magnitude NaN and flag nothing there),
    # masking of implausible values, and optional clipping
    masked_as = nan if not isinstance(nan, str) else "nan"  # masked entries: ignored, or filled like NaN
    X_all, scored, info = prepare_values(lfc.to_numpy(dtype=np.float64), lfc.index, nan, clip,
                                         max_abs=max_abs, masked_as=masked_as)
    if verbose:
        warn_if_linear_fc(X_all, "detect_dominant")
    ctx = obs[context_col].to_numpy()
    artifact_z, excluded = flag_artifact_rows(info, ctx, lfc.shape[1], max_artifact_z)
    scored = scored & ~excluded

    parts, capped = [], []  # capped: (context, n_candidates, cap) where the cap was hit
    blank = {"magnitude": np.nan, "magnitude_ratio": np.nan, "magnitude_z": np.nan, "n_genes_extreme": 0,
             "frac_genes_extreme": np.nan, "candidate": False, "dominant": False}
    for c in pd.unique(ctx):
        in_context = (ctx == c) & scored
        unscored = (ctx == c) & ~scored  # all-NaN rows (nan='skip') and artifact-excluded rows
        if in_context.any():
            cols, n_cand, cap = _score_context(X_all[in_context], t)
            if n_cand > cap:
                capped.append((c, n_cand, cap))
            part = pd.DataFrame({"context": c, **cols}, index=lfc.index[in_context])
        else:
            part = pd.DataFrame(index=lfc.index[:0])
        if unscored.any():
            empty = pd.DataFrame({"context": c, **blank}, index=lfc.index[unscored])
            part = empty if part.empty else pd.concat([part, empty])
        parts.append(part)
    rows = pd.concat(parts).loc[lfc.index]  # back to the input row order
    rows["frac_masked"] = info["row_frac_masked"]
    rows["artifact_z"] = artifact_z
    rows["artifact_excluded"] = excluded
    if condition_col is not None:
        rows.insert(1, "condition", obs[condition_col].to_numpy())
    rows.attrs["preprocessing"] = info
    summary = _summarise_by_condition(rows, t["min_frac_genes_extreme"]) if condition_col is not None else None

    if verbose:
        report_preparation(info, "" if nan == "raise" else "not scored / ignored per entry" if nan == "skip"
                           else f"filled with {nan}")
        n_dom = int(rows["dominant"].sum())
        settings = ", ".join(f"{k}={v}" for k, v in t.items())
        print(f"Dominance ({settings}): {n_dom} of {len(rows)} LFC profiles flagged in "
              f"{rows.loc[rows.dominant, 'context'].nunique()} of {rows['context'].nunique()} contexts.")
        if summary is not None and n_dom:
            top = summary[summary.n_dominant > 0]
            print("  flagged conditions (n contexts): "
                  + ", ".join(f"{k} ({v})" for k, v in top["n_dominant"].items()))
        if capped:
            print(f"WARNING: in {len(capped)} context(s) more than max_frac_flagged={t['max_frac_flagged']} of "
                  f"the rows passed the thresholds (e.g. {capped[0][0]}: {capped[0][1]} candidates, cap "
                  f"{capped[0][2]}); only the largest were flagged. Widespread dominance is not a few "
                  "outliers: inspect 'rows', raise min_magnitude_ratio, or reconsider removal.")
    return rows, summary


def drop_dominant(data, obs=None, rows=None):
    """Remove the observations flagged as dominant in `rows` (from detect_dominant).

    Works on any data with the same rows (LFC or expression). Rows excluded as artifacts are kept
    (they are not dominant); pass max_artifact_z to ContextWGCNA to leave them out of the network.
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
