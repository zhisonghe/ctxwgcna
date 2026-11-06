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

from .modules import module_indices

SUBMODULE_COLUMNS = ["gene", "submodule", "parent", "frac_contexts", "mean_t"]


def _jaccard(B):
    """Pairwise Jaccard index between the rows of a boolean matrix."""
    B = B.astype(np.float64)
    inter = B @ B.T
    size = B.sum(axis=1)
    union = size[:, None] + size[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _submodule(active, contexts, min_context_frac):
    """Submodule defined by a context group: genes active in enough of its contexts (step 5)."""
    need = max(1, math.ceil(min_context_frac * len(contexts)))
    count = active[contexts].sum(axis=0)  # in how many of the group's contexts each gene is active
    return {"contexts": list(contexts), "members": count >= need, "count": count}


def _group_contexts(active, used, jaccard_cut):
    """Group the used contexts by the Jaccard overlap of their active gene sets (step 4)."""
    if used.size == 1:
        return [used]
    D = 1.0 - _jaccard(active[used])
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - jaccard_cut, criterion="distance")
    return [used[labels == g] for g in np.unique(labels)]


def split_module(t_ctx, t_thr=3.0, min_partners=10, min_size=20, jaccard_cut=0.5, min_context_frac=0.5,
                 dedup_jaccard=0.8, absolute=False):
    """Split one module given its per-context statistics t_ctx with shape (K, m, m).

    Returns a list of dicts (largest first) with 'contexts' (context indices), 'members' (bool, m)
    and per-gene 'frac_contexts' (share of the group's contexts in which the gene is active).
    """
    K, m, _ = t_ctx.shape
    t = np.abs(t_ctx) if absolute else t_ctx

    # steps 1-3: significant edges -> active genes -> used contexts
    significant = t >= t_thr
    significant[:, np.arange(m), np.arange(m)] = False  # no self-edges
    active = significant.sum(axis=2) >= min_partners  # (K, m)
    used = np.flatnonzero(active.sum(axis=1) >= min_size)
    if used.size == 0:
        return []

    # steps 4-5: group contexts, one candidate submodule per group
    subs = [_submodule(active, g, min_context_frac) for g in _group_contexts(active, used, jaccard_cut)]
    subs = [s for s in subs if s["members"].sum() >= min_size]

    # merge submodules with near-identical gene sets (pool their contexts), most similar pair first
    while len(subs) > 1:
        J = _jaccard(np.array([s["members"] for s in subs]))
        np.fill_diagonal(J, -1)
        a, b = np.unravel_index(np.argmax(J), J.shape)
        if J[a, b] < dedup_jaccard:
            break
        merged = _submodule(active, sorted(subs[a]["contexts"] + subs[b]["contexts"]), min_context_frac)
        subs = [s for i, s in enumerate(subs) if i not in (a, b)] + [merged]

    for s in subs:
        s["frac_contexts"] = s["count"] / len(s["contexts"])
    subs.sort(key=lambda s: -s["members"].sum())
    return subs


def _mean_within_t(t_ctx, contexts, members):
    """Per member gene: mean t with the other members, over the given contexts."""
    sub_t = t_ctx[np.ix_(contexts, members, members)]  # (n_contexts, mm, mm)
    mm = len(members)
    diag = sub_t[:, np.arange(mm), np.arange(mm)].sum(axis=0)  # exclude self-pairs
    return (sub_t.sum(axis=(0, 2)) - diag) / (len(contexts) * max(mm - 1, 1))


def split_modules(modules, get_t, contexts, t_thr=3.0, min_partners=None, min_size=20, jaccard_cut=0.5,
                  min_context_frac=0.5, dedup_jaccard=0.8, absolute=False):
    """Split every hard module. get_t(idx) must return (K, m, m) per-context t for gene indices idx.

    Returns (submodules, memberships):
      submodules  one row per submodule: parent, contexts, n_contexts, n_genes, n_overlap (genes
                  also in another submodule of the same parent), parent_size, parent_unplaced
                  (parent genes in no submodule)
      memberships long table gene -> submodule with frac_contexts and mean_t (mean within-submodule
                  t over the submodule's contexts)
    A module that does not need splitting yields a single submodule '<module>.1'; one in which no
    context activates >= min_size genes yields an empty placeholder '<module>.0'.
    """
    min_partners = math.ceil(min_size / 2) if min_partners is None else min_partners
    genes = modules.index
    sub_rows, mem_rows = [], []
    for mod, idx in module_indices(modules).items():
        t_ctx = get_t(idx)
        subs = split_module(t_ctx, t_thr=t_thr, min_partners=min_partners, min_size=min_size,
                            jaccard_cut=jaccard_cut, min_context_frac=min_context_frac,
                            dedup_jaccard=dedup_jaccard, absolute=absolute)
        if not subs:
            sub_rows.append({"submodule": f"{mod}.0", "parent": mod, "contexts": "", "n_contexts": 0,
                             "n_genes": 0, "n_overlap": 0})
            continue

        n_memberships = np.sum([s["members"] for s in subs], axis=0)  # per gene of the module
        t_dir = np.abs(t_ctx) if absolute else t_ctx
        for i, s in enumerate(subs, start=1):
            name = f"{mod}.{i}"
            members = np.flatnonzero(s["members"])
            mean_t = _mean_within_t(t_dir, s["contexts"], members)
            sub_rows.append({"submodule": name, "parent": mod,
                             "contexts": ",".join(contexts[k] for k in s["contexts"]),
                             "n_contexts": len(s["contexts"]), "n_genes": len(members),
                             "n_overlap": int((n_memberships[members] > 1).sum())})
            for j, g in enumerate(members):
                mem_rows.append({"gene": genes[idx[g]], "submodule": name, "parent": mod,
                                 "frac_contexts": float(s["frac_contexts"][g]), "mean_t": float(mean_t[j])})

    submodules = pd.DataFrame(sub_rows).set_index("submodule")
    memberships = pd.DataFrame(mem_rows, columns=SUBMODULE_COLUMNS)
    placed = memberships.groupby("parent")["gene"].nunique()
    submodules["parent_size"] = submodules["parent"].map(modules.value_counts())
    submodules["parent_unplaced"] = (submodules["parent_size"]
                                     - submodules["parent"].map(placed).fillna(0).astype(int))
    return submodules, memberships
