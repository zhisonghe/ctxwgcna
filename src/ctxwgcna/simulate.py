"""Synthetic log-FC data with global and context-local co-regulation, for testing and benchmarking."""
import numpy as np
import pandas as pd

# Planted co-regulated gene sets: name -> (gene index range, systems in which the set is active).
# G1 is global, L1/L2 are local to two systems each. PQ (system 6) and QR (system 7) overlap in the
# genes 220-259 ("Q"): a context-blind TOM tends to chain P-Q-R together; a context-matched TOM should
# keep P and R apart, and the submodule split should place Q in both.
# DEFAULT_TRUTH gives each gene one label for scoring: (label, first gene, last gene + 1).
DEFAULT_DESIGN = {
    "G1": (range(0, 60), "all"),
    "L1": (range(60, 120), [0, 1]),
    "L2": (range(120, 180), [2, 3]),
    "PQ": (range(180, 260), [6]),
    "QR": (range(220, 300), [7]),
}
DEFAULT_TRUTH = [("G1", 0, 60), ("L1", 60, 120), ("L2", 120, 180),
                 ("P", 180, 220), ("Q", 220, 260), ("R", 260, 300)]


def simulate_lfc(n_genes=500, n_systems=10, n_conditions=100, design=None, truth=None,
                loading=(0.6, 1.0), noise_sd=1.0, seed=0):
    """Return (lfc, obs, truth_labels).

    lfc rows are (system, condition) observations, columns genes. In each system a co-regulated
    gene set shares a latent condition profile f(c) scaled by gene-specific loadings; outside its
    active systems the set is pure noise.
    """
    design = DEFAULT_DESIGN if design is None else design
    truth = DEFAULT_TRUTH if truth is None else truth
    rng = np.random.default_rng(seed)
    systems = [f"sys{s}" for s in range(n_systems)]
    conditions = [f"c{c:03d}" for c in range(n_conditions)]

    lfc = noise_sd * rng.standard_normal((n_systems, n_conditions, n_genes))
    for genes, active in design.values():
        genes = np.asarray(list(genes))
        active_systems = range(n_systems) if active == "all" else active
        loadings = rng.uniform(*loading, size=len(genes))  # fixed per gene, shared across systems
        for s in active_systems:
            # one latent condition profile per system, scaled by each gene's loading
            lfc[s][:, genes] += rng.standard_normal((n_conditions, 1)) * loadings

    gene_names = [f"g{i:04d}" for i in range(n_genes)]
    obs = pd.DataFrame([(s, c) for s in systems for c in conditions], columns=["system", "condition"])
    obs.index = [f"{s}|{c}" for s, c in zip(obs.system, obs.condition)]
    lfc = pd.DataFrame(lfc.reshape(n_systems * n_conditions, n_genes), index=obs.index, columns=gene_names)

    labels = np.array(["noise"] * n_genes, dtype=object)
    for name, a, b in truth:
        labels[a:b] = name
    return lfc, obs, pd.Series(labels, index=gene_names, name="truth")
