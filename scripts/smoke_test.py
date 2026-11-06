"""Smoke test: run the stock PyWGCNA pipeline on synthetic data with planted modules.

Reference only: needs PyWGCNA, which is not a ctxwgcna dependency (pip install PyWGCNA).
Run:  .venv/bin/python scripts/smoke_test.py
"""
import os
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import PyWGCNA

rng = np.random.default_rng(0)
n_samples, genes_per_module, n_modules, n_noise = 40, 150, 3, 150

# Each planted module = one latent factor + gene-specific noise.
blocks = []
for m in range(n_modules):
    latent = rng.normal(size=(n_samples, 1))
    loadings = rng.uniform(0.6, 1.0, size=(1, genes_per_module))
    blocks.append(latent @ loadings + 0.3 * rng.normal(size=(n_samples, genes_per_module)))
blocks.append(rng.normal(size=(n_samples, n_noise)))
X = np.hstack(blocks) + 8.0  # shift so values pass the TPM > 1 filter

truth = np.repeat([f"M{m}" for m in range(n_modules)] + ["noise"],
                  [genes_per_module] * n_modules + [n_noise])
genes = [f"g{i}" for i in range(X.shape[1])]
samples = [f"s{i}" for i in range(n_samples)]
expr = pd.DataFrame(X, index=samples, columns=genes)  # rows = samples, cols = genes

outdir = tempfile.mkdtemp(prefix="pywgcna_smoke_")
w = PyWGCNA.WGCNA(name="smoke", geneExp=expr, powers=[1, 2, 4, 6, 8, 10, 12],
                  minModuleSize=30, outputPath=outdir + "/", save=False)
w.preprocess(show=False)
w.findModules()

print(f"\nChosen soft power: {w.power}")
ct = pd.crosstab(pd.Series(truth, index=genes, name="planted"),
                 w.datExpr.var["moduleColors"].rename("detected"))
print(ct)

# Each planted module should map mostly to a single detected (non-grey) module.
ok = True
for m in range(n_modules):
    row = ct.loc[f"M{m}"].drop("grey", errors="ignore")
    if row.max() / genes_per_module < 0.8:
        ok = False
print("\nSMOKE TEST", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
