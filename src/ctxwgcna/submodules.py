"""Second level: split each hard module into context-defined submodules that may share genes.

For a hard module M (genes S) and every context k, using the single-context standardized z
t_k(i, j) = arctanh(s_k(i, j)) * sqrt(n_k - 3) (one context, so no selection bias):

1. edge (i, j) is *significant in k* if t_k(i, j) >= t_thr (|t| for unsigned networks)
2. gene i is *active in k* if it has >= min_partners significant partners inside S
3. context k is *used* by M if >= min_size genes are active in it
4. used contexts are grouped by the Jaccard overlap of their active gene sets (average linkage,
   groups have mean Jaccard >= jaccard_cut): contexts that activate the same genes stay together
5. each context group g defines a submodule: genes active in >= ceil(min_context_frac * |g|) of the
   contexts of g; submodules with gene-set Jaccard >= dedup_jaccard are merged
A gene can belong to several submodules of its module (e.g. Q in P+Q [system 6] and Q+R [system 7]).
"""
import math

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _jaccard(B):
    """Pairwise Jaccard between rows of a boolean matrix."""
    B = B.astype(np.float64)
    inter = B @ B.T
    size = B.sum(axis=1)
    union = size[:, None] + size[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def split_module(t_ctx, t_thr=3.0, min_partners=10, min_size=20, jaccard_cut=0.5, min_context_frac=0.5,
                 dedup_jaccard=0.8, absolute=False):
    """Split one module given its per-context statistics t_ctx with shape (K, m, m).

    Returns a list of dicts with 'contexts' (context indices), 'members' (bool, m) and per-gene
    'frac_contexts' (share of the group's contexts in which the gene is active).
    """
    K, m, _ = t_ctx.shape
    t = np.abs(t_ctx) if absolute else t_ctx
    sig = t >= t_thr
    sig[:, np.arange(m), np.arange(m)] = False
    active = sig.sum(axis=2) >= min_partners  # (K, m)
    used = np.flatnonzero(active.sum(axis=1) >= min_size)
    if used.size == 0:
        return []

    if used.size == 1:
        groups = [used]
    else:
        D = 1.0 - _jaccard(active[used])
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="average")
        lab = fcluster(Z, t=1.0 - jaccard_cut, criterion="distance")
        groups = [used[lab == g] for g in np.unique(lab)]

    subs = []
    for g in groups:
        need = max(1, math.ceil(min_context_frac * len(g)))
        count = active[g].sum(axis=0)
        members = count >= need
        if members.sum() >= min_size:
            subs.append({"contexts": list(g), "members": members, "count": count})

    # merge submodules with near-identical gene sets (pool their contexts)
    merged = True
    while merged and len(subs) > 1:
        merged = False
        J = _jaccard(np.array([s["members"] for s in subs]))
        np.fill_diagonal(J, -1)
        a, b = np.unravel_index(np.argmax(J), J.shape)
        if J[a, b] >= dedup_jaccard:
            ctx = sorted(subs[a]["contexts"] + subs[b]["contexts"])
            need = max(1, math.ceil(min_context_frac * len(ctx)))
            count = active[ctx].sum(axis=0)
            new = {"contexts": ctx, "members": count >= need, "count": count}
            subs = [s for i, s in enumerate(subs) if i not in (a, b)] + [new]
            merged = True

    for s in subs:
        s["frac_contexts"] = s["count"] / len(s["contexts"])
    subs.sort(key=lambda s: -s["members"].sum())
    return subs


def split_modules(modules, get_t, contexts, t_thr=3.0, min_partners=None, min_size=20, jaccard_cut=0.5,
                  min_context_frac=0.5, dedup_jaccard=0.8, absolute=False):
    """Split every hard module. get_t(idx) must return (K, m, m) per-context t for gene indices idx.

    Returns (submodules, memberships):
      submodules  one row per submodule: parent, contexts, n_contexts, n_genes, n_overlap
                  (genes also in another submodule of the same parent)
      memberships long table gene -> submodule with frac_contexts and mean_t (mean within-submodule
                  t over the submodule's contexts)
    Each module is split into >= 1 submodule if any context activates >= min_size of its genes;
    a module that does not need splitting yields a single submodule '<module>.1'.
    """
    min_partners = math.ceil(min_size / 2) if min_partners is None else min_partners
    genes = modules.index
    sub_rows, mem_rows = [], []
    for mod in sorted([m for m in pd.unique(modules) if m != "M0"], key=lambda m: int(m[1:])):
        idx = np.flatnonzero((modules == mod).to_numpy())
        t_ctx = get_t(idx)
        subs = split_module(t_ctx, t_thr=t_thr, min_partners=min_partners, min_size=min_size,
                            jaccard_cut=jaccard_cut, min_context_frac=min_context_frac,
                            dedup_jaccard=dedup_jaccard, absolute=absolute)
        n_member = np.zeros(len(idx), dtype=int)
        for s in subs:
            n_member += s["members"]
        for i, s in enumerate(subs, start=1):
            name = f"{mod}.{i}"
            mem = np.flatnonzero(s["members"])
            tc = np.abs(t_ctx) if absolute else t_ctx
            sub_t = tc[np.ix_(s["contexts"], mem, mem)]
            mm = len(mem)
            mean_t = (sub_t.sum(axis=(0, 2)) - sub_t[:, np.arange(mm), np.arange(mm)].sum(axis=0)) / (
                len(s["contexts"]) * max(mm - 1, 1))
            sub_rows.append({"submodule": name, "parent": mod,
                             "contexts": ",".join(contexts[k] for k in s["contexts"]),
                             "n_contexts": len(s["contexts"]), "n_genes": mm,
                             "n_overlap": int((n_member[mem] > 1).sum())})
            for j, g in enumerate(mem):
                mem_rows.append({"gene": genes[idx[g]], "submodule": name, "parent": mod,
                                 "frac_contexts": float(s["frac_contexts"][g]), "mean_t": float(mean_t[j])})
        if not subs:
            sub_rows.append({"submodule": f"{mod}.0", "parent": mod, "contexts": "", "n_contexts": 0,
                             "n_genes": 0, "n_overlap": 0})
    submodules = pd.DataFrame(sub_rows).set_index("submodule")
    memberships = pd.DataFrame(mem_rows, columns=["gene", "submodule", "parent", "frac_contexts", "mean_t"])
    parent_size = modules.value_counts()
    placed = memberships.groupby("parent")["gene"].nunique()
    submodules["parent_size"] = submodules["parent"].map(parent_size)
    submodules["parent_unplaced"] = submodules["parent_size"] - submodules["parent"].map(placed).fillna(0).astype(int)
    return submodules, memberships


def module_context_profile_soft(adj, weights_fn, modules):
    """Adjacency-weighted soft context profile per module, from stored per-edge context weights.

    profile[m, k] = sum_{i<j in m} a_ij w_ij^k / sum_{i<j in m} a_ij   (rows sum to ~1)
    """
    A = np.array(adj, dtype=np.float64)
    np.fill_diagonal(A, 0.0)
    rows = {}
    for mod in sorted([m for m in pd.unique(modules) if m != "M0"], key=lambda m: int(m[1:])):
        idx = np.flatnonzero((modules == mod).to_numpy())
        a = A[np.ix_(idx, idx)]
        w = weights_fn(idx)
        rows[mod] = (w * a[None]).sum(axis=(1, 2)) / a.sum()
    return pd.DataFrame(rows).T
