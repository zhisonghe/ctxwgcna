"""ContextWGCNA: WGCNA on a 'union over contexts' similarity with a context-restricted TOM."""
import numpy as np
import pandas as pd

from . import modules as _mod
from . import network as _net
from . import submodules as _sub
from .data import as_frames, make_context_blocks, prepare_frames, warn_if_linear_fc
from .similarity import similarity_matrix, standardized_z
from .support import compute_support, soft_context_weights


class ContextWGCNA:
    """
    Steps:
      1. contexts       obs[context_col] groups the rows (e.g. by 'system' or 'condition')
      2. similarity     per context: pearson / spearman / bicor / cosine
      3. top-n          Fisher z, standardized by sqrt(n_c - 3), best n contexts per pair (ids kept)
      4. summary        weighted mean of selected z, minus selection bias (calibrate=...)
      5. WGCNA          soft-threshold adjacency -> context-restricted TOM (tom_mode=...) ->
                        dynamic tree cut -> per-module context profiles
      then optionally   split_modules(): overlapping, context-defined submodules per module

    Typical use:  cw = ContextWGCNA(data, obs); cw.run(); cw.split_modules()
             or:  cw = ContextWGCNA(adata, layer="log2fc", genes="highly_variable")
    Each step can also be called on its own; missing earlier steps are run automatically.

    :param data: the data: a DataFrame (rows = observations, i.e. LFC profiles or samples; columns =
        genes), or an AnnData object (observations x genes).
    :param obs: DataFrame indexed like data with context annotations. Required for a DataFrame;
        for AnnData it defaults to adata.obs.
    :param layer: AnnData only: use adata.layers[layer] instead of adata.X.
    :param genes: optional gene subset: a list of gene names, or (AnnData only) the name of a boolean
        adata.var column such as 'highly_variable'. Cost grows with genes^2, so subset large data.
    :param network_type: 'signed' (default), 'signed hybrid' or 'unsigned'. Determines the
        top-n selection rule unless `selection` is given ('positive' for signed types,
        'absolute' for unsigned).
    :param tom_mode: 'matched' (default), 'shared', or 'standard' (plain WGCNA TOM).
    :param support_min_t: count a context as supporting a pair (in the TOM and the hard module
        profile) only if its t >= this value.
    :param nan: missing values. 'raise' (default): error. 'skip': drop rows that are entirely NaN
        (e.g. untested combinations); partly-NaN rows still raise, since correlations here need
        complete rows. A number, e.g. 0: fill NaN with it.
    :param clip: limit extreme values before the correlations: None (default), c (clip to [-c, c]),
        (lo, hi), or 'pNN' (+- the NNth percentile of |value| over all values, e.g. 'p95').
    :param max_abs: treat |value| > max_abs as missing measurements (the |LFC| ~ 20-35 a GLM returns
        when one group has zero counts): 'auto' (default; cutoff from the data, nothing masked if
        there is no separate mode of large values, see detect_dominant), a number (e.g. 15), or None
        (no masking); they are set to 0 (or to nan if it is a number),
        before clipping.
    :param max_artifact_z: leave out rows with unusually many masked genes for their context
        (artifact_z > max_artifact_z, as in detect_dominant; e.g. 3). None (default): keep all.
    .data / .obs hold the data after these steps; .preprocessing records them.
    See README.md section 4 for all parameters.
    """

    def __init__(self, data, obs=None, context_col="system", contexts=None, similarity="pearson", top_n=3,
                 network_type="signed", selection=None, weighting="n", calibrate="analytic", n_perm=10,
                 tom_mode="matched", support_min_t=None, min_obs=10, seed=0, chunk_mb=128,
                 store_context_t="auto", store_max_mb=2048, layer=None, genes=None, nan="raise",
                 clip=None, max_abs="auto", max_artifact_z=None):
        if network_type not in _net.NETWORK_TYPES:
            raise ValueError(f"network_type must be one of {_net.NETWORK_TYPES}")
        # --- input: DataFrame + obs, or AnnData (optionally a layer and a gene subset) ---
        data, obs = as_frames(data, obs, layer=layer, genes=genes)
        data, obs, self.preprocessing = prepare_frames(data, obs, nan=nan, clip=clip, max_abs=max_abs,
                                                       context_col=context_col, max_artifact_z=max_artifact_z)
        if similarity == "cosine":  # uncentred: needs 0 = "no change"
            warn_if_linear_fc(data.to_numpy(dtype=np.float64), "similarity='cosine'")
        self.data, self.obs, self.layer = data, obs, layer
        self.genes = pd.Index(data.columns)
        self.context_col = context_col
        self.similarity = similarity
        self.top_n = top_n
        self.network_type = network_type
        self.selection = selection or ("absolute" if network_type == "unsigned" else "positive")
        self.weighting, self.calibrate, self.n_perm = weighting, calibrate, n_perm
        self.tom_mode, self.support_min_t = tom_mode, support_min_t
        self.seed, self.chunk_mb = seed, chunk_mb
        self.store_context_t, self.store_max_mb = store_context_t, store_max_mb

        # --- step 1 now; results of later steps are filled in as they run ---
        self.blocks = make_context_blocks(data, obs, context_col, contexts=contexts, min_obs=min_obs)
        self.support = None  # ContextSupport (steps 2-4)
        self.power = None  # soft-threshold power
        self.sft = None  # scale-free fit table
        self.adjacency = None
        self.tom = None
        self.modules = None  # Series gene -> module label ('M0' = unassigned)
        self.linkage = None  # gene dendrogram (scipy linkage matrix)
        self._min_module_size = None
        self.submodules = None
        self.memberships = None

    # -- steps -----------------------------------------------------------------------------------
    def compute_support(self):
        """Steps 2-4: per-context similarity, top-n contexts per pair, calibrated summary z."""
        self.support = compute_support(
            self.blocks, method=self.similarity, top_n=self.top_n, selection=self.selection,
            weighting=self.weighting, calibrate=self.calibrate, n_perm=self.n_perm,
            chunk_mb=self.chunk_mb, seed=self.seed, genes=self.genes,
            store_context_t=self.store_context_t, store_max_mb=self.store_max_mb)
        return self.support

    def pick_power(self, powers=None, r2_cut=0.85, mean_cut=100):
        """Choose the soft-threshold power by scale-free fit; sets .power and returns the fit table."""
        if self.support is None:
            self.compute_support()
        powers = _net.DEFAULT_POWERS if powers is None else powers
        self.sft = _net.scale_free_table(self.support.r, powers, self.network_type)
        self.power, found = _net.pick_power(self.sft, r2_cut, mean_cut)
        if not found:
            print(f"No power reached signed R^2 >= {r2_cut}; using best-fitting power {self.power}. "
                  "Inspect .sft before trusting it.")
        return self.sft

    def build_network(self, power=None):
        """Adjacency and context-restricted TOM (power: given, previously chosen, or picked now)."""
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
        """Hard (non-overlapping) modules by dynamic hybrid tree cut of 1 - TOM."""
        if self.tom is None:
            self.build_network()
        self._min_module_size = min_module_size
        self.modules, self.linkage = _mod.detect_modules(
            self.tom, self.genes, min_module_size=min_module_size, deep_split=deep_split, **cutree_kwargs)
        return self.modules

    def run(self, power=None, min_module_size=30, deep_split=2, **cutree_kwargs):
        """All steps from scratch; the power is picked by scale-free fit unless given."""
        self.compute_support()
        self.power = power
        self.build_network()
        return self.find_modules(min_module_size=min_module_size, deep_split=deep_split, **cutree_kwargs)

    # -- second level: overlapping submodules -----------------------------------------------------
    def _context_t(self, idx):
        """(K, m, m) per-context t for gene indices idx: stored ones, or recomputed for just idx."""
        if self.support.t_all is not None:
            return self.support.context_t(idx)
        out = np.empty((len(self.blocks), len(idx), len(idx)), dtype=np.float32)
        for k, b in enumerate(self.blocks):
            out[k] = standardized_z(b.X[:, idx], self.similarity)
        return out

    def _context_weights(self, idx):
        """(K, m, m) soft context weights for gene indices idx."""
        return soft_context_weights(self._context_t(idx), self.selection)

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

        soft=True: adjacency-weighted mean of the per-edge soft context weights (all contexts).
        soft=False: adjacency-weighted share of the edges' top-n supporting contexts.
        """
        if not soft:
            return _mod.module_context_profile(self.adjacency, self.support, self.modules, self.support_min_t)
        prof = _mod.module_context_profile_soft(self.adjacency, self._context_weights, self.modules)
        prof.columns = self.support.contexts
        return prof

    def module_context_coherence(self):
        """Mean within-module similarity per context (direct check, independent of top-n)."""
        return _mod.module_context_coherence(self.blocks, self.modules, self.similarity)

    def context_eigengenes(self, module, contexts=None):
        """Module eigengene computed separately within each context (optionally only `contexts`).

        Returns {context: (Series of eigengene values over that context's observations, var explained)}.
        """
        g = np.flatnonzero((self.modules == module).to_numpy())
        out = {}
        for b in self.blocks:
            if contexts is not None and b.name not in contexts:
                continue
            me, var_explained = _mod.eigengene(b.X[:, g], center=self.similarity != "cosine")
            out[b.name] = (pd.Series(me, index=b.obs_names, name=f"ME{module}"), var_explained)
        return out


def global_wgcna(data, similarity="pearson", network_type="signed", power=None, powers=None,
                 min_module_size=30, deep_split=2, r2_cut=0.85, mean_cut=100, layer=None, genes=None,
                 nan="raise", clip=None, max_abs="auto"):
    """Baseline: standard WGCNA with one similarity over all observations pooled + standard TOM.

    data is a DataFrame (observations x genes) or AnnData; layer, genes, nan, clip and max_abs work as
    in ContextWGCNA. Returns (module labels, power used).
    """
    data, _ = as_frames(data, None, layer=layer, genes=genes)
    data, _, _ = prepare_frames(data, None, nan=nan, clip=clip, max_abs=max_abs, verbose=False)
    r = similarity_matrix(data.to_numpy(), similarity)
    if power is None:
        sft = _net.scale_free_table(r, _net.DEFAULT_POWERS if powers is None else powers, network_type)
        power, _ = _net.pick_power(sft, r2_cut, mean_cut)
    adj = _net.adjacency_from_r(r, power, network_type)
    tom = _net.context_tom(adj, mode="standard")
    labels, _ = _mod.detect_modules(tom, data.columns, min_module_size=min_module_size, deep_split=deep_split)
    return labels, power
