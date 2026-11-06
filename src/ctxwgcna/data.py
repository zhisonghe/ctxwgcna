"""Split a fold-change matrix into context blocks."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ContextBlock:
    name: str
    obs_names: pd.Index  # observations (rows of the FC matrix) belonging to this context
    X: np.ndarray  # (n_obs, n_genes)

    @property
    def n_obs(self):
        return self.X.shape[0]


def check_unique(fc, obs=None):
    """Raise if observation (row) or gene (column) labels are duplicated."""
    for what, idx in (("fc row (observation)", fc.index), ("fc column (gene)", fc.columns)) + (
            (("obs row", obs.index),) if obs is not None else ()):
        if not idx.is_unique:
            dup = list(idx[idx.duplicated()].unique()[:5])
            raise ValueError(f"duplicated {what} labels, e.g. {dup}; labels must be unique")


def make_context_blocks(fc, obs, context_col, contexts=None, min_obs=10):
    """Group the rows of `fc` by `obs[context_col]`.

    :param fc: DataFrame, rows = observations (e.g. one system x condition FC profile), columns = genes.
    :param obs: DataFrame with the same index as `fc`, holding e.g. 'system' and 'condition' columns.
    :param context_col: column of `obs` defining the contexts (e.g. 'system' or 'condition').
        Within a context, similarities are computed across its observations.
    :param contexts: optional subset (list) of context names to use.
    :param min_obs: contexts with fewer observations are dropped (with a message); must be >= 4.
    """
    if min_obs < 4:
        raise ValueError("min_obs must be >= 4 (Fisher z standard error is 1/sqrt(n - 3))")
    check_unique(fc, obs)
    if not fc.index.equals(obs.index):
        if set(fc.index) != set(obs.index):
            raise ValueError("fc and obs must have the same row index (observations)")
        obs = obs.loc[fc.index]
    if context_col not in obs.columns:
        raise ValueError(f"{context_col!r} is not a column of obs")
    if fc.isna().to_numpy().any():
        raise ValueError("fc contains NaN; impute or filter before building contexts")
    if not np.isfinite(fc.to_numpy(dtype=np.float64)).all():
        raise ValueError("fc contains infinite values; cap or filter them before building contexts")

    labels = obs[context_col]
    names = list(pd.unique(labels)) if contexts is None else list(contexts)
    missing = [c for c in names if c not in set(labels)]
    if missing:
        raise ValueError(f"contexts not found in obs[{context_col!r}]: {missing}")

    values = fc.to_numpy(dtype=np.float64)
    blocks, dropped = [], []
    for name in names:
        mask = (labels == name).to_numpy()
        if mask.sum() < min_obs:
            dropped.append((name, int(mask.sum())))
            continue
        blocks.append(ContextBlock(name=str(name), obs_names=fc.index[mask], X=values[mask]))
    if dropped:
        print(f"Dropped {len(dropped)} context(s) with < {min_obs} observations: {dropped}")
    if not blocks:
        raise ValueError("no contexts left")
    return blocks
