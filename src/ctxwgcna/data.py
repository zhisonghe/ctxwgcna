"""Input validation and splitting of the data matrix into context blocks (step 1)."""
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
    check_unique(fc, obs)
    if not fc.index.equals(obs.index):
        if set(fc.index) != set(obs.index):
            raise ValueError("fc and obs must have the same row index (observations)")
        obs = obs.loc[fc.index]  # same labels, different order: align obs to fc
    if context_col not in obs.columns:
        raise ValueError(f"{context_col!r} is not a column of obs")
    values = fc.to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError("fc contains NaN; impute or filter before building contexts")
    if not np.isfinite(values).all():
        raise ValueError("fc contains infinite values; cap or filter them before building contexts")

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
