#!/usr/bin/env python
"""Run context-aware WGCNA end to end and write all results to an output directory.

Examples
--------
# real data (FC matrix: rows = observations, columns = genes; obs table with the same row index)
python scripts/run_ctxwgcna.py --fc fc.csv --obs obs.csv --context-col system --outdir results/

# demo on simulated data
python scripts/run_ctxwgcna.py --simulate --outdir results_demo/

See README.md for the meaning of every option.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from ctxwgcna import ContextWGCNA, detect_dominant, drop_dominant, global_wgcna


def read_table(path, sep):
    sep = sep or ("\t" if path.endswith((".tsv", ".txt", ".tsv.gz", ".txt.gz")) else ",")
    return pd.read_csv(path, sep=sep, index_col=0)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = p.add_argument_group("input / output")
    io.add_argument("--fc", help="FC matrix (csv/tsv), rows = observations, columns = genes")
    io.add_argument("--obs", help="observation table (csv/tsv), same row index as --fc")
    io.add_argument("--transpose", action="store_true", help="--fc has genes as rows; transpose it")
    io.add_argument("--sep", default=None, help="column separator (default: from file extension)")
    io.add_argument("--simulate", action="store_true", help="use simulated demo data instead of --fc/--obs")
    io.add_argument("--outdir", required=True)

    d = p.add_argument_group("dominant FC profiles (see README: 'Dominant conditions')")
    d.add_argument("--detect-dominant", action="store_true",
                   help="flag FC profiles with far larger and broader changes than the rest of their context")
    d.add_argument("--exclude-dominant", action="store_true",
                   help="detect AND remove flagged profiles before the network analysis")
    d.add_argument("--dominant-label-col", default="condition",
                   help="obs column to summarise flagged profiles by (default: condition)")
    d.add_argument("--dominant-min-fold", type=float, default=5.0)
    d.add_argument("--dominant-min-z", type=float, default=4.0)
    d.add_argument("--dominant-min-gene-frac", type=float, default=0.10)
    d.add_argument("--dominant-max-frac", type=float, default=0.05)

    c = p.add_argument_group("contexts and similarity")
    c.add_argument("--context-col", default="system")
    c.add_argument("--contexts", default=None, help="comma-separated subset of contexts to use")
    c.add_argument("--min-obs", type=int, default=10)
    c.add_argument("--similarity", default="pearson", choices=["pearson", "spearman", "bicor", "cosine"])

    s = p.add_argument_group("context support")
    s.add_argument("--top-n", type=int, default=3)
    s.add_argument("--weighting", default="n", choices=["n", "equal"])
    s.add_argument("--calibrate", default="analytic", choices=["none", "analytic", "permutation"])
    s.add_argument("--n-perm", type=int, default=10)

    n = p.add_argument_group("network")
    n.add_argument("--network-type", default="signed", choices=["signed", "signed hybrid", "unsigned"])
    n.add_argument("--power", type=int, default=None, help="soft-threshold power (default: scale-free pick)")
    n.add_argument("--r2-cut", type=float, default=0.85)
    n.add_argument("--tom-mode", default="matched", choices=["matched", "shared", "standard"])

    m = p.add_argument_group("modules")
    m.add_argument("--min-module-size", type=int, default=30)
    m.add_argument("--deep-split", type=int, default=2, choices=[0, 1, 2, 3, 4])

    sp = p.add_argument_group("submodules (overlapping split)")
    sp.add_argument("--no-split", action="store_true")
    sp.add_argument("--t-thr", type=float, default=3.0)
    sp.add_argument("--min-submodule-size", type=int, default=None)
    sp.add_argument("--min-partners", type=int, default=None)
    sp.add_argument("--jaccard-cut", type=float, default=0.5)
    sp.add_argument("--min-context-frac", type=float, default=0.5)

    o = p.add_argument_group("other")
    o.add_argument("--global-baseline", action="store_true", help="also run standard (pooled) WGCNA")
    o.add_argument("--save-matrices", action="store_true", help="save summary r, adjacency, TOM (npz)")
    o.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.simulate and not (args.fc and args.obs):
        p.error("give --fc and --obs, or --simulate")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    out = lambda name: os.path.join(args.outdir, name)
    t0 = time.time()

    truth = None
    if args.simulate:
        from ctxwgcna.simulate import simulate_fc
        fc, obs, truth = simulate_fc(n_genes=1000, n_systems=20, n_conditions=300, seed=args.seed)
    else:
        fc = read_table(args.fc, args.sep)
        fc = fc.T if args.transpose else fc
        obs = read_table(args.obs, args.sep)
    print(f"FC matrix: {fc.shape[0]} observations x {fc.shape[1]} genes")

    n_dropped = 0
    if args.detect_dominant or args.exclude_dominant:
        label_col = args.dominant_label_col if args.dominant_label_col in obs.columns else None
        dom_rows, dom_summary = detect_dominant(
            fc, obs, context_col=args.context_col, label_col=label_col, min_fold=args.dominant_min_fold,
            min_z=args.dominant_min_z, min_gene_frac=args.dominant_min_gene_frac,
            max_frac=args.dominant_max_frac)
        dom_rows.to_csv(out("dominance_rows.csv"))
        if dom_summary is not None:
            dom_summary.to_csv(out("dominance_summary.csv"))
        if args.exclude_dominant:
            n0 = len(fc)
            fc, obs = drop_dominant(fc, obs, dom_rows)
            n_dropped = n0 - len(fc)
            print(f"removed {n_dropped} dominant FC profiles; {len(fc)} remain")

    contexts = args.contexts.split(",") if args.contexts else None
    cw = ContextWGCNA(fc, obs, context_col=args.context_col, contexts=contexts, similarity=args.similarity,
                      top_n=args.top_n, network_type=args.network_type, weighting=args.weighting,
                      calibrate=args.calibrate, n_perm=args.n_perm, tom_mode=args.tom_mode,
                      min_obs=args.min_obs, seed=args.seed)
    print(f"{len(cw.blocks)} contexts: " + ", ".join(f"{b.name}(n={b.n_obs})" for b in cw.blocks))

    sup = cw.compute_support()
    print(f"support: top_n={sup.top_n}, selection={sup.selection}, calibrate={sup.calibrate}, "
          f"mean bias={np.mean(sup.bias):.4f} (Fisher z)")

    cw.pick_power(r2_cut=args.r2_cut)
    cw.sft.to_csv(out("soft_threshold.csv"), index=False)
    if args.power is not None:
        cw.power = args.power
    print(f"power: {cw.power} (scale-free table in soft_threshold.csv)")

    cw.build_network()
    mods = cw.find_modules(min_module_size=args.min_module_size, deep_split=args.deep_split)
    mods.to_csv(out("modules.csv"))
    print("modules:", mods.value_counts().to_dict())

    cw.module_context_profile().to_csv(out("module_context_profile.csv"))
    cw.module_context_coherence().to_csv(out("module_context_coherence.csv"))

    if not args.no_split:
        subs, mem = cw.split_modules(t_thr=args.t_thr, min_size=args.min_submodule_size,
                                     min_partners=args.min_partners, jaccard_cut=args.jaccard_cut,
                                     min_context_frac=args.min_context_frac)
        subs.to_csv(out("submodules.csv"))
        mem.to_csv(out("memberships.csv"), index=False)
        cw.membership_matrix().astype(int).to_csv(out("membership_matrix.csv"))
        print(f"submodules: {len(subs)} ({(subs.groupby('parent').size() > 1).sum()} modules split, "
              f"{int(subs['n_overlap'].sum())} gene-submodule overlaps)")

    if args.global_baseline:
        glob, gp = global_wgcna(fc, similarity=args.similarity, network_type=args.network_type,
                                power=args.power, min_module_size=args.min_module_size,
                                deep_split=args.deep_split)
        glob.rename("global_module").to_csv(out("global_modules.csv"))
        print(f"global baseline (power {gp}):", glob.value_counts().to_dict())

    if args.save_matrices:
        np.savez_compressed(out("matrices.npz"), genes=np.asarray(cw.genes), summary_r=cw.support.r,
                            adjacency=cw.adjacency, tom=cw.tom)

    if truth is not None:
        from sklearn.metrics import adjusted_rand_score
        truth.to_csv(out("simulation_truth.csv"))
        print(f"simulation: ARI(modules, truth) = {adjusted_rand_score(truth, mods):.3f}")

    config = {k: v for k, v in vars(args).items()}
    config.update(chosen_power=cw.power, n_contexts=len(cw.blocks), n_dominant_removed=n_dropped,
                  runtime_s=round(time.time() - t0, 1))
    with open(out("run_config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"done in {time.time() - t0:.1f}s -> {args.outdir}")


if __name__ == "__main__":
    sys.exit(main())
