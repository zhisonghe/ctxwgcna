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
    """Return (fc, obs) DataFrames from either a DataFrame + obs table or an AnnData object.

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
        fc = pd.DataFrame(X, index=pd.Index(adata.obs_names), columns=pd.Index(adata.var_names))
        return fc, (adata.obs if obs is None else obs)

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


def check_unique(fc, obs=None):
    """Raise if observation (row) or gene (column) labels are duplicated.

    Duplicated labels make row alignment between `fc` and `obs` ambiguous, and gene lookups by name
    return several positions instead of one.
    """
    indexes = [("fc row (observation)", fc.index), ("fc column (gene)", fc.columns)]
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
        msg = (f"fc contains NaN in {int(nan.any(axis=1).sum())} of {len(values)} rows "
               f"({int(all_nan.sum())} of them entirely NaN, e.g. {list(index[all_nan][:3])}).")
        if all_nan.any():
            msg += " Drop untested rows first, e.g. keep = fc.notna().any(axis=1); fc, obs = fc[keep], obs[keep]."
        if (nan.any(axis=1) & ~all_nan).any():
            msg += (" Partly-NaN rows: drop the affected genes, or fill with 0 only if 'not measured' "
                    "really means 'no change'.")
        raise ValueError(msg)
    if not np.isfinite(values).all():
        raise ValueError("fc contains infinite values; cap or filter them first")


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
        return -c, c, f"+-{c:.3g} ({clip} of |FC|)"
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


def prepare_values(X, index, nan="raise", clip=None):
    """Apply a NaN rule and optional clipping to an (n_obs, n_genes) array.

    :param nan: 'raise' (error on NaN), 'skip' (all-NaN rows are marked unscored; other NaN entries
        are left in place for the caller to ignore), or a number to fill NaN with (e.g. 0).
    :param clip: None; c (clip to [-c, c]); (lo, hi); or 'pNN' (+- the NNth percentile of |X| over
        all observed values). Infinite values are allowed only when they get clipped.
    Returns (X, scored: bool mask of rows that hold data, info: dict describing what was done).
    """
    is_nan = np.isnan(X)
    all_nan = is_nan.all(axis=1)
    info = {"nan": nan, "n_nan_rows": int(is_nan.any(axis=1).sum()), "n_all_nan_rows": int(all_nan.sum()),
            "clip": clip, "clip_bounds": None, "frac_clipped": 0.0}
    if isinstance(nan, str):
        if nan == "raise":
            try:
                check_finite(np.where(np.isinf(X), 0.0, X), index)  # inf is checked after clipping
            except ValueError as err:
                raise ValueError(f"{err} Or pass nan='skip' (ignore them) or nan=0 (fill).") from None
        elif nan != "skip":
            raise ValueError("nan must be 'raise', 'skip' or a fill value (e.g. 0)")
        scored = ~all_nan  # 'skip': all-NaN rows are not scored; other NaNs are ignored per entry
    else:
        X = np.where(is_nan, float(nan), X)  # fill
        scored = np.ones(len(X), dtype=bool)
    bounds = _clip_bounds(X, clip)
    if bounds is not None:
        lo, hi, desc = bounds
        observed = ~np.isnan(X)
        info["frac_clipped"] = float(((X < lo) | (X > hi))[observed].mean()) if observed.any() else 0.0
        info["clip_bounds"] = (lo, hi)
        info["clip_desc"] = desc
        X = np.clip(X, lo, hi)
    if np.isinf(X).any():
        raise ValueError("fc contains infinite values; pass clip=... or cap them first")
    return X, scored, info


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


def prepare_frames(fc, obs, nan="raise", clip=None, verbose=True):
    """NaN rule + clipping for the network: returns (fc, obs, info) with all-NaN rows dropped.

    Unlike detect_dominant, nan='skip' cannot ignore single NaN entries here: correlations would need
    pairwise-complete observations (a different n per gene pair). Rows that are only partly NaN
    therefore raise an error under 'skip'; fill them (nan=0) or drop the affected genes.
    """
    X, scored, info = prepare_values(fc.to_numpy(dtype=np.float64), fc.index, nan, clip)
    if nan == "skip" and np.isnan(X[scored]).any():
        n_part = info["n_nan_rows"] - info["n_all_nan_rows"]
        raise ValueError(f"{n_part} rows are partly NaN; nan='skip' only drops rows that are entirely NaN. "
                         "Fill them (nan=0, if 'not measured' means 'no change') or drop the affected genes.")
    if info["n_nan_rows"] == 0 and info["clip_bounds"] is None:
        return fc, obs, info  # nothing changed: keep the caller's frame as is
    kept = fc.index[scored]
    out = pd.DataFrame(X[scored], index=kept, columns=fc.columns)
    if obs is not None:
        obs = obs[obs.index.isin(kept)]
    if verbose:
        if info["n_all_nan_rows"] and nan == "skip":
            print(f"NaN: dropped {info['n_all_nan_rows']} rows that are entirely NaN.")
        elif info["n_nan_rows"] and nan not in ("raise", "skip"):
            print(f"NaN: filled in {info['n_nan_rows']} rows with {nan}.")
        if info["clip_bounds"] is not None:
            print(f"Clipped to {info['clip_desc']}: {100 * info['frac_clipped']:.2f}% of values.")
    return out, obs, info


def make_context_blocks(fc, obs, context_col, contexts=None, min_obs=10):
    """Group the rows of `fc` by `obs[context_col]`.

    :param fc: DataFrame, rows = observations (e.g. one system x condition FC profile, or one sample
        of an expression matrix), columns = genes.
    :param obs: DataFrame with the same row labels as `fc`, holding e.g. 'system' and 'condition'.
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
    check_unique(fc, obs)
    if not fc.index.equals(obs.index):
        if set(fc.index) != set(obs.index):
            raise ValueError("fc and obs must have the same row index (observations)")
        obs = obs.loc[fc.index]  # same labels, different order: align obs to fc
    if context_col not in obs.columns:
        raise ValueError(f"{context_col!r} is not a column of obs")
    values = fc.to_numpy(dtype=np.float64)
    check_finite(values, fc.index)

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
        blocks.append(ContextBlock(name=str(name), obs_names=fc.index[in_context], X=values[in_context]))
    if dropped:
        print(f"Dropped {len(dropped)} context(s) with < {min_obs} observations: {dropped}")
    if not blocks:
        raise ValueError("no contexts left")
    return blocks
