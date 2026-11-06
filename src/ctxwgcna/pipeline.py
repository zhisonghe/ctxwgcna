"""ContextWGCNA: WGCNA on a 'union over contexts' similarity with a context-restricted TOM."""
import numpy as np
import pandas as pd

from . import modules as _mod
from . import network as _net
from . import submodules as _sub
from .data import make_context_blocks
from .similarity import fisher_z, similarity_matrix
from .support import compute_support


class ContextWGCNA:
    """
    Steps:
      1. contexts       obs[context_col] groups the FC rows (e.g. by 'system' or 'condition')
      2. similarity     per context: pearson / spearman / bicor / cosine
      3. top-n          Fisher z, standardized by sqrt(n_c - 3), best n contexts per pair (ids kept)
      4. summary        weighted mean of selected z, minus selection bias (calibrate=...)
      5. WGCNA          soft-threshold adjacency -> context-restricted TOM (tom_mode=...) ->
                        dynamic tree cut -> per-module context profiles

    :param fc: DataFrame, rows = observations (FC profiles), columns = genes.
    :param obs: DataFrame indexed like fc with context annotations.
    :param network_type: 'signed' (default), 'signed hybrid' or 'unsigned'. Determines the
        top-n selection rule unless `selection` is given ('positive' for signed types,
        'absolute' for unsigned).
    :param tom_mode: 'matched' (default), 'shared', or 'standard' (plain WGCNA TOM).
    """

    def __init__(self, fc, obs, context_col="system", contexts=None, similarity="pearson", top_n=3,
                 network_type="signed", selection=None, weighting="n", calibrate="analytic", n_perm=10,
                 tom_mode="matched", support_min_t=None, min_obs=10, seed=0, chunk_mb=128,
                 store_context_t="auto", store_max_mb=2048):
        if network_type not in _net.NETWORK_TYPES:
            raise ValueError(f"network_type must be one of {_net.NETWORK_TYPES}")
        self.fc, self.obs = fc, obs
        self.genes = pd.Index(fc.columns)
        self.context_col = context_col
        self.similarity = similarity
        self.top_n = top_n
        self.network_type = network_type
        self.selection = selection or ("absolute" if network_type == "unsigned" else "positive")
        self.weighting, self.calibrate, self.n_perm = weighting, calibrate, n_perm
        self.tom_mode, self.support_min_t = tom_mode, support_min_t
        self.seed, self.chunk_mb = seed, chunk_mb
        self.store_context_t, self.store_max_mb = store_context_t, store_max_mb

        self.blocks = make_context_blocks(fc, obs, context_col, contexts=contexts, min_obs=min_obs)
        self.support = None
        self.power = None
        self.sft = None
        self.adjacency = None
        self.tom = None
        self.modules = None
        self.linkage = None
        self._min_module_size = None
        self.submodules = None
        self.memberships = None

    # -- steps -----------------------------------------------------------------------------------
    def compute_support(self):
        self.support = compute_support(
            self.blocks, method=self.similarity, top_n=self.top_n, selection=self.selection,
            weighting=self.weighting, calibrate=self.calibrate, n_perm=self.n_perm,
            chunk_mb=self.chunk_mb, seed=self.seed, genes=self.genes,
            store_context_t=self.store_context_t, store_max_mb=self.store_max_mb)
        return self.support

    def pick_power(self, powers=None, r2_cut=0.85, mean_cut=100):
        if self.support is None:
            self.compute_support()
        powers = list(range(1, 11)) + list(range(12, 31, 2)) if powers is None else powers
        self.sft = _net.scale_free_table(self.support.r, powers, self.network_type)
        self.power, found = _net.pick_power(self.sft, r2_cut, mean_cut)
        if not found:
            print(f"No power reached signed R^2 >= {r2_cut}; using best-fitting power {self.power}. "
                  "Inspect .sft before trusting it.")
        return self.sft

    def build_network(self, power=None):
        if power is not None:
            self.power = power
        if self.support is None:
            self.compute_support()
        if self.power is None:
            self.pick_power()
        self.adjacency = _net.adjacency_from_r(self.support.r, self.power, self.network_type)
        self.tom = _net.context_tom(self.adjacency, self.support, mode=self.tom_mode,
                                    min_t=self.support_min_t)
        return self.tom

    def find_modules(self, min_module_size=30, deep_split=2, **cutree_kwargs):
        if self.tom is None:
            self.build_network()
        self._min_module_size = min_module_size
        self.modules, self.linkage = _mod.detect_modules(
            self.tom, self.genes, min_module_size=min_module_size, deep_split=deep_split, **cutree_kwargs)
        return self.modules

    def run(self, power=None, min_module_size=30, deep_split=2, **cutree_kwargs):
        self.compute_support()
        if power is None:
            self.pick_power()
        self.build_network(power)
        return self.find_modules(min_module_size=min_module_size, deep_split=deep_split, **cutree_kwargs)

    # -- second level: overlapping submodules -----------------------------------------------------
    def _context_t(self, idx):
        """(K, m, m) per-context standardized z for gene indices idx (stored, or recomputed)."""
        if self.support.t_all is not None:
            return self.support.context_t(idx)
        out = np.empty((len(self.blocks), len(idx), len(idx)), dtype=np.float32)
        for k, b in enumerate(self.blocks):
            out[k] = fisher_z(similarity_matrix(b.X[:, idx], self.similarity)) * np.sqrt(max(b.n_obs - 3, 1))
        return out

    def split_modules(self, t_thr=3.0, min_size=None, min_partners=None, jaccard_cut=0.5,
                      min_context_frac=0.5, dedup_jaccard=0.8):
        """Split each hard module into context-defined submodules that may share genes.

        See ctxwgcna.submodules for the procedure. min_size defaults to the min_module_size used in
        find_modules; min_partners to ceil(min_size / 2). Sets and returns
        (self.submodules, self.memberships).
        """
        if self.modules is None:
            raise ValueError("run find_modules() first")
        if self.similarity == "cosine":
            print("Warning: split significance (t >= t_thr) assumes a zero null mean, which does not hold "
                  "for cosine similarity; raise t_thr or interpret with care.")
        min_size = self._min_module_size if min_size is None else min_size
        self.submodules, self.memberships = _sub.split_modules(
            self.modules, self._context_t, self.support.contexts, t_thr=t_thr, min_partners=min_partners,
            min_size=min_size, jaccard_cut=jaccard_cut, min_context_frac=min_context_frac,
            dedup_jaccard=dedup_jaccard, absolute=self.selection == "absolute")
        return self.submodules, self.memberships

    def membership_matrix(self):
        """Boolean genes x submodules matrix (a gene may be True in several columns)."""
        m = self.memberships
        if m is None:
            raise ValueError("run split_modules() first")
        mat = pd.crosstab(m["gene"], m["submodule"]).astype(bool)
        return mat.reindex(self.genes, fill_value=False)

    # -- interpretation --------------------------------------------------------------------------
    def module_context_profile(self, soft=True):
        """Context profile of each module's internal edges (rows sum to 1).

        soft=True: adjacency-weighted mean of the stored per-edge soft context weights (all contexts).
        soft=False: share of edges whose top-n supporting contexts include each context.
        """
        if soft:
            prof = _sub.module_context_profile_soft(
                self.adjacency, lambda idx: self.support.context_weights(idx) if self.support.t_all is not None
                else _soft_weights(self._context_t(idx), self.selection), self.modules)
            prof.columns = self.support.contexts
            return prof
        return _mod.module_context_profile(self.adjacency, self.support, self.modules, self.support_min_t)

    def module_context_coherence(self):
        """Mean within-module similarity per context (direct check, independent of top-n)."""
        return _mod.module_context_coherence(self.blocks, self.modules, self.similarity)

    def context_eigengenes(self, module, contexts=None):
        """Module eigengene computed separately within each (supporting) context.

        Returns {context: (Series of eigengene values over that context's observations, var explained)}.
        """
        g = np.flatnonzero((self.modules == module).to_numpy())
        out = {}
        for b in self.blocks:
            if contexts is not None and b.name not in contexts:
                continue
            me, ve = _mod.eigengene(b.X[:, g], center=self.similarity != "cosine")
            out[b.name] = (pd.Series(me, index=b.obs_names, name=f"ME{module}"), ve)
        return out


def _soft_weights(t, selection):
    w = np.maximum(np.abs(t) if selection == "absolute" else t, 0.0)
    tot = w.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, w / tot, 0.0)


def global_wgcna(fc, similarity="pearson", network_type="signed", power=None, powers=None,
                 min_module_size=30, deep_split=2, r2_cut=0.85, mean_cut=100):
    """Baseline: one similarity over all observations pooled + standard TOM."""
    r = similarity_matrix(fc.to_numpy(), similarity)
    if power is None:
        powers = list(range(1, 11)) + list(range(12, 31, 2)) if powers is None else powers
        sft = _net.scale_free_table(r, powers, network_type)
        power, _ = _net.pick_power(sft, r2_cut, mean_cut)
    adj = _net.adjacency_from_r(r, power, network_type)
    tom = _net.context_tom(adj, mode="standard")
    labels, _ = _mod.detect_modules(tom, fc.columns, min_module_size=min_module_size, deep_split=deep_split)
    return labels, power
