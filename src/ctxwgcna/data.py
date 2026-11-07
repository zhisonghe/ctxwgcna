"""Input handling (DataFrame or AnnData), validation, and splitting into context blocks (step 1)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ContextBlock:
    """The observations of one context (e.g. all conditions of one system)."""
    name: str
    obs_names: pd.Index  # row labels of the data matrix that belong to this context
    X: np.ndarray  # (n_obs, n_genes) values of those rows

    @property
    def n_obs(self):
        return self.X.shape[0]


def is_anndata(data):
    """True for an AnnData object (checked by its attributes, so anndata need not be installed)."""
    return all(hasattr(data, a) for a in ("X", "obs", "var", "obs_names", "var_names", "layers"))


def as_frames(data, obs=None, layer=None, genes=None):
    """Return (data, obs) DataFrames from either a DataFrame + obs table or an AnnData object.

    :param data: DataFrame (rows = observations, columns = genes), or AnnData (adata.X or a layer
        holds observations x genes; adata.obs the annotations).
    :param obs: observation table. Required for a DataFrame; for AnnData it defaults to adata.obs
        (pass a table to override it).
    :param layer: AnnData only: use adata.layers[layer] instead of adata.X.
    :param genes: optional gene subset: a list of gene names, or (AnnData only) the name of a
        boolean column of adata.var, e.g. 'highly_variable'. For AnnData the subset is taken before
        the matrix is made dense.
    """
    if is_anndata(data):
        adata = data
        if genes is not None:
            if isinstance(genes, str):
                if genes not in adata.var.columns:
                    raise ValueError(f"{genes!r} is not a column of adata.var")
                keep = adata.var[genes].to_numpy(dtype=bool)
            else:
                missing = pd.Index(genes).difference(adata.var_names)
                if len(missing):
                    raise ValueError(f"genes not in adata.var_names, e.g. {list(missing[:5])}")
                keep = adata.var_names.get_indexer(pd.Index(genes))
            adata = adata[:, keep]
        if layer is None:
            X = adata.X
        elif layer in adata.layers:
            X = adata.layers[layer]
        else:
            raise ValueError(f"layer {layer!r} not found; available: {list(adata.layers.keys())}")
        if X is None:
            raise ValueError("adata.X is empty; pass layer=... to use one of adata.layers")
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)  # sparse -> dense
        data = pd.DataFrame(X, index=pd.Index(adata.obs_names), columns=pd.Index(adata.var_names))
        return data, (adata.obs if obs is None else obs)

    if layer is not None:
        raise ValueError("layer is only used with AnnData input")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame (observations x genes) or an AnnData object")
    if genes is not None:
        if isinstance(genes, str):
            raise ValueError("genes as a column name needs AnnData input; pass a list of gene names")
        missing = pd.Index(genes).difference(data.columns)
        if len(missing):
            raise ValueError(f"genes not in the columns of the data, e.g. {list(missing[:5])}")
        data = data[list(genes)]
    return data, obs


def check_unique(data, obs=None):
    """Raise if observation (row) or gene (column) labels are duplicated.

    Duplicated labels make row alignment between `data` and `obs` ambiguous, and gene lookups by name
    return several positions instead of one.
    """
    indexes = [("data row (observation)", data.index), ("data column (gene)", data.columns)]
    if obs is not None:
        indexes.append(("obs row", obs.index))
    for what, index in indexes:
        if not index.is_unique:
            examples = list(index[index.duplicated()].unique()[:5])
            raise ValueError(f"duplicated {what} labels, e.g. {examples}; labels must be unique")


def check_finite(values, index):
    """Raise if the (n_obs, n_genes) array `values` has NaN or infinite entries.

    Rows that are entirely NaN usually mark combinations that were never tested (e.g. a condition
    with too few cells in one system); the message counts them and shows how to drop them, since
    filling them with 0 would claim "no change" where nothing was measured.
    """
    nan = np.isnan(values)
    if nan.any():
        all_nan = nan.all(axis=1)
        msg = (f"data contains NaN in {int(nan.any(axis=1).sum())} of {len(values)} rows "
               f"({int(all_nan.sum())} of them entirely NaN, e.g. {list(index[all_nan][:3])}).")
        if all_nan.any():
            msg += " Drop untested rows first, e.g. keep = data.notna().any(axis=1); data, obs = data[keep], obs[keep]."
        if (nan.any(axis=1) & ~all_nan).any():
            msg += (" Partly-NaN rows: drop the affected genes, or fill with 0 only if 'not measured' "
                    "really means 'no change'.")
        raise ValueError(msg)
    if not np.isfinite(values).all():
        raise ValueError("data contains infinite values; cap or filter them first")


def _clip_bounds(X, clip):
    """(lo, hi, description) for `clip`: None, c (-> +-c), (lo, hi), or 'pNN' (+- NNth pct of |X|)."""
    if clip is None:
        return None
    if isinstance(clip, str):
        if not (clip.startswith("p") and _is_number(clip[1:]) and 0 < float(clip[1:]) <= 100):
            raise ValueError(f"clip={clip!r}: a string must be a percentile like 'p95' or 'p99.5'")
        finite = np.abs(X[np.isfinite(X)])
        c = float(np.percentile(finite, float(clip[1:]))) if finite.size else 0.0
        if c <= 0:
            raise ValueError(f"clip={clip!r} gives a cutoff of {c}; use a higher percentile or a value")
        return -c, c, f"+-{c:.3g} ({clip} of |value|)"
    if np.isscalar(clip):
        if not clip > 0:
            raise ValueError("clip as a single value must be > 0 (it is used as +-clip)")
        return -float(clip), float(clip), f"+-{float(clip):.3g}"
    lo, hi = (float(v) for v in clip)
    if not lo < hi:
        raise ValueError("clip=(lo, hi) needs lo < hi")
    return lo, hi, f"[{lo:.3g}, {hi:.3g}]"


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def auto_max_abs(X, bins=100, smooth=5, min_ratio=10.0, min_frac=1e-4):
    """Data-derived max_abs: the antimode between the bulk of |X| and a separate high-|X| mode.

    Looks at the histogram of log2|X| above the median |X| (`bins` bins, log counts smoothed over
    `smooth` bins). The bulk peak is the highest bin. A second mode counts as separate when its peak
    is at least `min_ratio` times higher than the lowest point between the two (on the smoothed
    counts) and at least `min_frac` of all values lie from that low point upwards. The cutoff is the
    middle (on the log scale) of the longest run of lowest-count bins between the two peaks, i.e. the
    middle of the gap when it is empty. Returns (cutoff, upper peak) or None when |X| has no second
    mode (a single, possibly heavy, tail), so nothing should be masked.

    Typical use: GLM LFCs of ~20-35 where one group has zero counts, which form their own mode far
    above any genuine LFC.
    """
    a = np.abs(np.asarray(X, dtype=np.float64))
    a = a[np.isfinite(a) & (a > 0)]
    if a.size < 100:
        return None
    v = np.log2(a)
    v = v[v >= np.log2(np.median(a))]
    h, edges = np.histogram(v, bins=bins)
    ls = np.convolve(np.log1p(h), np.ones(smooth) / smooth, mode="same")
    p1, best = int(np.argmax(ls)), None
    for p2 in range(p1 + 2, bins):
        low = p1 + int(np.argmin(ls[p1:p2]))
        if (ls[p2] - ls[low] >= np.log(min_ratio) and h[low:].sum() >= min_frac * a.size
                and (best is None or ls[p2] > ls[best])):
            best = p2
    if best is None:
        return None
    seg = h[p1:best + 1]
    is_min = np.append(seg == seg.min(), False)
    runs, start = [], None  # (length, first, end) of runs of lowest-count bins
    for i, flag in enumerate(is_min):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((i - start, start, i))
            start = None
    _, first, end = max(runs)
    cutoff = 2 ** ((edges[p1 + first] + edges[p1 + end]) / 2)
    return float(cutoff), float(2 ** ((edges[best] + edges[best + 1]) / 2))


def prepare_values(X, index, nan="raise", clip=None, max_abs=None, masked_as="nan"):
    """Apply a NaN rule, optional masking of implausible values, and optional clipping to an
    (n_obs, n_genes) array, in that order.

    :param nan: 'raise' (error on NaN), 'skip' (all-NaN rows are marked unscored; other NaN entries
        are left in place for the caller to ignore), or a number to fill NaN with (e.g. 0).
    :param max_abs: values with |x| > max_abs (including +-inf) are treated as missing measurements,
        e.g. the |LFC| ~ 30 a GLM returns when one group has zero counts. They never trigger
        nan='raise'; they become `masked_as`. 'auto' derives the cutoff from the data (auto_max_abs:
        the gap below a separate high-|x| mode) and masks nothing when there is no such mode.
    :param masked_as: 'nan' (left missing, for callers that ignore single entries) or a number.
    :param clip: None; c (clip to [-c, c]); (lo, hi); or 'pNN' (+- the NNth percentile of |X| over
        all observed, unmasked values). Infinite values are allowed only when masked or clipped.
    Returns (X, scored: bool mask of rows that hold data, info: dict describing what was done).
    """
    is_nan = np.isnan(X)
    all_nan = is_nan.all(axis=1)
    info = {"nan": nan, "n_nan_rows": int(is_nan.any(axis=1).sum()), "n_all_nan_rows": int(all_nan.sum()),
            "max_abs": max_abs, "max_abs_rule": None, "n_masked": 0, "frac_masked": 0.0,
            "clip": clip, "clip_bounds": None, "frac_clipped": 0.0}
    if isinstance(nan, str):
        if nan == "raise":
            try:
                check_finite(np.where(np.isinf(X), 0.0, X), index)  # inf is checked after masking/clipping
            except ValueError as err:
                raise ValueError(f"{err} Or pass nan='skip' (ignore them) or nan=0 (fill).") from None
        elif nan != "skip":
            raise ValueError("nan must be 'raise', 'skip' or a fill value (e.g. 0)")
        scored = ~all_nan  # 'skip': all-NaN rows are not scored; other NaNs are ignored per entry
    else:
        X = np.where(is_nan, float(nan), X)  # fill
        scored = np.ones(len(X), dtype=bool)
    if isinstance(max_abs, str):
        if max_abs != "auto":
            raise ValueError("max_abs must be a number or 'auto'")
        found = auto_max_abs(X)
        max_abs = None if found is None else found[0]
        info["max_abs"], info["max_abs_rule"] = max_abs, "auto"
        info["max_abs_mode"] = None if found is None else found[1]
    if max_abs is not None:
        if not max_abs > 0:
            raise ValueError("max_abs must be > 0")
        with np.errstate(invalid="ignore"):
            masked = np.abs(X) > max_abs  # NaN compares False
        info["n_masked"] = int(masked.sum())
        observed = ~np.isnan(X)
        info["frac_masked"] = float(masked[observed].mean()) if observed.any() else 0.0
        X = np.where(masked, np.nan if masked_as == "nan" else float(masked_as), X)
        if masked_as == "nan":
            scored &= ~np.isnan(X).all(axis=1)  # a row masked completely has nothing left to score
    bounds = _clip_bounds(X, clip)
    if bounds is not None:
        lo, hi, desc = bounds
        observed = ~np.isnan(X)
        info["frac_clipped"] = float(((X < lo) | (X > hi))[observed].mean()) if observed.any() else 0.0
        info["clip_bounds"] = (lo, hi)
        info["clip_desc"] = desc
        X = np.clip(X, lo, hi)
    if np.isinf(X).any():
        raise ValueError("data contains infinite values; pass max_abs=... or clip=..., or cap them first")
    return X, scored, info


def report_preparation(info, nan_message):
    """Print what prepare_values did (NaN rule, masking, clipping); nan_message describes the NaN rule."""
    if info["n_nan_rows"] and nan_message:
        print(f"NaN: {info['n_nan_rows']} rows contain NaN ({info['n_all_nan_rows']} entirely); {nan_message}.")
    auto = info.get("max_abs_rule") == "auto"
    if info["max_abs"] is not None:
        how = f" (max_abs='auto': gap below a mode at |value| ~ {info['max_abs_mode']:.3g})" if auto else ""
        print(f"Masked |value| > {info['max_abs']:.3g}{how}: {info['n_masked']} values "
              f"({100 * info['frac_masked']:.2f}%), treated as missing.")
    elif auto:
        print("max_abs='auto': no separate mode of large |value| found; nothing masked.")
    if info["clip_bounds"] is not None:
        print(f"Clipped to {info['clip_desc']}: {100 * info['frac_clipped']:.2f}% of values.")


def warn_if_linear_fc(X, what):
    """Print a warning if X has no negative values, i.e. looks like linear-scale FC (1 = no change).

    Parts of ctxwgcna that read 0 as "no change" (dominance magnitude, cosine similarity, filling NaN
    with 0, symmetric clipping) need log FC. Expression matrices are also non-negative, so this is
    only called where FC input is required.
    """
    finite = X[np.isfinite(X)]
    if finite.size and finite.min() >= 0:
        print(f"WARNING: no negative values: this looks like linear-scale FC (1 = no change). {what} "
              "assumes log FC (0 = no change); pass log2(FC) instead.")


def prepare_frames(data, obs, nan="raise", clip=None, max_abs=None, verbose=True):
    """NaN rule, masking and clipping for the network: returns (data, obs, info), all-NaN rows dropped.

    Unlike detect_dominant, nan='skip' cannot ignore single NaN entries here: correlations would need
    pairwise-complete observations (a different n per gene pair). Rows that are only partly NaN
    therefore raise an error under 'skip'; fill them (nan=0) or drop the affected genes. Values masked
    by max_abs are filled with the nan value if it is a number, else with 0 ("no usable change").
    """
    masked_as = nan if not isinstance(nan, str) else 0.0
    X, scored, info = prepare_values(data.to_numpy(dtype=np.float64), data.index, nan, clip,
                                     max_abs=max_abs, masked_as=masked_as)
    if nan == "skip" and np.isnan(X[scored]).any():
        n_part = info["n_nan_rows"] - info["n_all_nan_rows"]
        raise ValueError(f"{n_part} rows are partly NaN; nan='skip' only drops rows that are entirely NaN. "
                         "Fill them (nan=0, if 'not measured' means 'no change') or drop the affected genes.")
    if info["n_nan_rows"] == 0 and info["n_masked"] == 0 and info["clip_bounds"] is None:
        if verbose and info["max_abs_rule"] == "auto":
            report_preparation(info, "")
        return data, obs, info  # nothing changed: keep the caller's frame as is
    kept = data.index[scored]
    out = pd.DataFrame(X[scored], index=kept, columns=data.columns)
    if obs is not None:
        obs = obs[obs.index.isin(kept)]
    if verbose:
        msg = ("rows that are entirely NaN dropped" if nan == "skip"
               else f"filled with {nan}" if not isinstance(nan, str) else "")
        report_preparation(info, msg)
        if info["n_masked"]:
            print(f"  (masked values set to {masked_as:g} for the correlations)")
    return out, obs, info


def make_context_blocks(data, obs, context_col, contexts=None, min_obs=10):
    """Group the rows of `data` by `obs[context_col]`.

    :param data: DataFrame, rows = observations (e.g. one system x condition LFC profile, or one sample
        of an expression matrix), columns = genes.
    :param obs: DataFrame with the same row labels as `data`, holding e.g. 'system' and 'condition'.
    :param context_col: column of `obs` defining the contexts (e.g. 'system' or 'condition').
        Within a context, similarities are computed across its observations.
    :param contexts: optional subset (list) of context names to use, in this order.
    :param min_obs: contexts with fewer observations are dropped (with a message); must be >= 4,
        since the Fisher z standard error 1/sqrt(n - 3) is undefined below that.
    :return: list of ContextBlock.
    """
    # --- validate ---
    if min_obs < 4:
        raise ValueError("min_obs must be >= 4 (Fisher z standard error is 1/sqrt(n - 3))")
    if obs is None:
        raise ValueError("obs (the observation table) is required")
    check_unique(data, obs)
    if not data.index.equals(obs.index):
        if set(data.index) != set(obs.index):
            raise ValueError("data and obs must have the same row index (observations)")
        obs = obs.loc[data.index]  # same labels, different order: align obs to data
    if context_col not in obs.columns:
        raise ValueError(f"{context_col!r} is not a column of obs")
    values = data.to_numpy(dtype=np.float64)
    check_finite(values, data.index)

    labels = obs[context_col]
    names = list(pd.unique(labels)) if contexts is None else list(contexts)
    present = set(labels)
    missing = [c for c in names if c not in present]
    if missing:
        raise ValueError(f"contexts not found in obs[{context_col!r}]: {missing}")

    # --- split ---
    blocks, dropped = [], []
    for name in names:
        in_context = (labels == name).to_numpy()
        n = int(in_context.sum())
        if n < min_obs:
            dropped.append((name, n))
            continue
        blocks.append(ContextBlock(name=str(name), obs_names=data.index[in_context], X=values[in_context]))
    if dropped:
        print(f"Dropped {len(dropped)} context(s) with < {min_obs} observations: {dropped}")
    if not blocks:
        raise ValueError("no contexts left")
    return blocks
