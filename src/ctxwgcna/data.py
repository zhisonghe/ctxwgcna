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
