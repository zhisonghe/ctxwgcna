# adapt_WGCNA

Context-aware WGCNA: a *union* of context-specific co-expression (systems or conditions as contexts),
with a TOM that only lets shared neighbours count when the links come from the same context.

## Layout
- `src/ctxwgcna/` – the adapted method (installed editable into `.venv`)
  - `data.py`        step 1: `make_context_blocks(fc, obs, context_col, contexts=None, min_obs=10)`
  - `similarity.py`  step 2: pearson | spearman | bicor | cosine (uncentred, FC=0 as reference)
  - `support.py`     steps 3–4: Fisher z, standardized by sqrt(n_c-3); top-n contexts per pair (ids kept);
                     (n_c-3)-weighted mean of selected z; selection-bias calibration
                     (`none` | `analytic` | `permutation`)
  - `network.py`     adjacency (signed default; signed hybrid; unsigned), scale-free table,
                     `context_tom(mode=standard|shared|matched)`
  - `modules.py`     tree cut, per-module context profile / coherence, eigengenes
  - `pipeline.py`    `ContextWGCNA` class; `global_wgcna` baseline (pooled similarity, standard TOM)
  - `simulate.py`    synthetic FC with global, system-local and P–Q–R "bridge" modules
- `tests/test_core.py` – `.venv/bin/python -m pytest -q`
- `external/PyWGCNA/` – clone (v2.2.0, f1d470e), reference only. PyWGCNA is **no longer a dependency** (dropped
  2026-11-05): `scale_free_fit_index` reimplemented in network.py (matches PyWGCNA to ≤3e-14 on 205 inputs; frozen
  reference values in tests). Still installed in the project `.venv` for `scripts/smoke_test.py` only.
- `scripts/smoke_test.py` – stock PyWGCNA pipeline on synthetic data (needs PyWGCNA installed).
- `scripts/run_ctxwgcna.py` – CLI runner; `README.md` – user documentation (install, run, parameters).

## Usage
```python
from ctxwgcna import ContextWGCNA
cw = ContextWGCNA(fc, obs, context_col="system", similarity="pearson", top_n=3,
                  network_type="signed", calibrate="analytic", tom_mode="matched")
modules = cw.run(power=12, min_module_size=20)   # or power=None -> scale-free criterion
cw.module_context_profile()     # which systems support each module's edges
cw.module_context_coherence()   # mean within-module similarity per system
cw.support.pair("geneA", "geneB")   # top-n supporting contexts of one pair
cw.support.pair_profile("geneA", "geneB")   # all contexts: t, similarity, soft weight

# second level: overlapping, context-defined submodules within each hard module
subs, mem = cw.split_modules(t_thr=3.0)   # min_size defaults to min_module_size
cw.membership_matrix()                     # genes x submodules (bool; a gene can be in several)
```
`fc`: rows = observations (system × condition FC vs per-system control), columns = genes; `obs` has the
same index with `system` / `condition` columns.

## Design decisions & definitions
- Ranking statistic t_c = arctanh(s_c)·sqrt(n_c−3); 'positive' selection for signed networks, '|t|' for unsigned.
- TOM_ij = (L_ij + a_ij)/(min(k_i,k_j)+1−a_ij);
  shared: L = Σ_u a_iu a_uj |C_iu∩C_uj|/n; matched: L = Σ_u a_iu a_uj |C_ij∩C_iu∩C_uj|/n.
  Computed as Σ_k (A∘I_k)² (∘I_k for matched). Stays in [0,1], ≤ standard TOM.
- Stored support: `support.t_all` (K, G, G) float32 signed standardized z per context (auto if ≤ 2 GB;
  else recomputed for module genes on demand). Soft edge weights w_ij^k = t⁺/Σ_k t⁺ (flat for global
  pairs, peaked for local ones; top-n ids are arbitrary for global pairs). `module_context_profile()`
  uses these by default (`soft=False` = old top-n indicator version).
- Submodules (`submodules.py`), per hard module M, single-context t (no selection bias):
  gene active in k ⇔ ≥ min_partners (default ceil(min_size/2)) partners in M with t_k ≥ t_thr (3.0);
  context used ⇔ ≥ min_size active genes; contexts grouped by Jaccard of active sets (avg linkage,
  ≥ 0.5); submodule = genes active in ≥ 50% of the group's contexts; near-identical submodules
  (Jaccard ≥ 0.8) merged. Unsplit module → single `M<k>.1`. Overlap only within a parent module.
- Calibration: analytic = Monte-Carlo mean of the same top-n summary under iid N(0,1) t's (assumes
  independent contexts and a zero null mean, i.e. centred methods). Permutation = permute each gene
  within each context; per-pair null for cosine (its null mean is not 0), pooled otherwise.

## Verified findings (2026-11-05)
- **PyWGCNA's `cutreeHybrid` is wrong**: on an identical distance matrix it disagreed with R
  `dynamicTreeCut::cutreeDynamic` (ARI 0.54 with PAM off) and absorbed noise genes into modules.
  PyPI `dynamicTreeCut==0.1.1` (pamStage=False) reproduced R exactly (ARI 1.0) → used instead.
  Its PAM stage crashes (`NameError: df_apply`), so PAM is off.
- PyWGCNA's `round(dissTOM, 8)` destroys structure when TOM is ~1e-6 (weak signal / high power); not used.
- PyWGCNA relies on pandas chained assignment (breaks on pandas 3) – moot since it was dropped; ctxwgcna tested on
  pandas 2.3.3 and 3.1.1 (FutureWarnings as errors).
- In `findModules`, PyWGCNA calls steps as `WGCNA.x(...)`, so subclass overrides are ignored (why we don't subclass).
- Simulation (500 genes, 10 systems × 100 conditions, signed, p=12): global WGCNA ARI 0.10 (local modules lost);
  context support + standard/shared/matched TOM ARI 0.77/0.86/0.85, local modules map to the right systems.
  Realistic size (1000 genes, 20 systems × 300 conditions, top_n=3): ARI 0.97, ~1 s (analytic), ~7 s (10 perms).
  Analytic and permutation bias agree for pearson (0.0852 both).
- Context TOM removes cross-context overlap (TOM P–R: 1.2× background under standard TOM → 1.0× matched),
  but P–Q–R still cluster together: Q is genuinely in both modules, and hard partitioning chains them.
  Needs overlapping modules / gene×context nodes if that matters.
- Scale-free auto power chose 9 (ARI 0.65) where 12 gave 0.85: treat the auto pick as a starting point.
- Submodule split (realistic size): P–Q–R module → {sys6: P+Q} and {sys7: Q+R}, all 40 Q genes in both;
  G1 (all 20 systems), L1 (sys0,1), L2 (sys2,3) stay single submodules; no noise gene placed.
  Pure-noise t gives no submodules.
- Dominant condition (2026-11-06 sim: 1000 genes, 20 sys × 300 cond; c000 gives 60% of genes |FC| 5–20× noise SD
  in all systems). Mean F1 over planted modules (0.75 = max for hard modules, P/Q/R share one):
  pearson 0.53 / 0.46 (responders form 170–270-gene blocks, swallow G1, fragment L1/L2); spearman, bicor,
  pearson-with-c000-removed all 0.75. BUT a real 5-condition programme F is lost by spearman/bicor (F1 0.0)
  and by per-row RMS scaling (0.0); only pearson + removal recovers everything (1.00).
  Detection: per-(system,condition) FC sum-of-squares share / within-system median: c000 37–143×,
  F conditions 1.2–1.9×, others ≤ 1.07×. → Recommendation: flag & remove dominant rows, keep pearson,
  analyse flagged conditions separately (responder sets mapped onto modules).
- `dominance.detect_dominant` (2026-11-06): magnitude-only flags real programmes (a 5-cond programme's strongest
  rows hit 7× median SS, and dropping them destroyed it) → criteria = SS fold ≥ 5 AND robust z (log, MAD) ≥ 4 AND
  ≥ 10% of genes extreme (|gene robust z| ≥ 5). Per system, so system-specific dominance works. max_frac 5% cap
  per system + warning. Sim (3 dominant conds incl. one in 5/20 systems + F): exactly 45 true rows flagged, F
  kept, recovery = oracle (G1 .98, L1 1, L2 1, F 1). No dominance → 0 flagged; 10% inflated → 30 candidates,
  15 flagged, warning.

## Open items
- Submodule split can't separate two programmes active in the *same* context within one hard module
  (gene activity is per context); would need clustering within a context's active graph.
- Overlap only within a parent module; genes outside it (other modules / M0) are not candidates.
- With cosine, split significance assumes zero null mean (warns) – per-context permutation null needed.
- Module merging (eigengene-based merging would wrongly use global MEs for local modules) – not implemented.
- Module–trait analysis restricted to supporting contexts.
- PAM stage for unassigned genes (port is broken).

## 2026-11-06 — external code review (Codex, Kimi) and fixes

Both reviewers found no errors in the core math (similarity, top-n, summary, calibration, TOM, tree
cut); Kimi brute-force checked shared/matched TOM and timed G=1000, K=300: support 8.6 s, matched TOM
3.9 s (so Codex's "TOM infeasible at 300 contexts" was rejected). Verified and fixed:
- dominance: genes with MAD = 0 used an SD that included the scored row, capping |z| at (n-1)/sqrt(n)
  (4.3 at n = 20 < extreme_z 5), so a broad dominant row in a 20-row context (condition contexts,
  sparse FC) was missed. Now leave-one-out SD. Also median ss = 0 gave fold 1 for all rows; now
  changed rows get fold inf and breadth + cap decide. Sparse null still flags nothing (test).
- pick_power crashed (KeyError nan) when all fits are NaN -> clear ValueError.
- detect_dominant: missing label/context column -> ValueError up front; duplicate labels rejected.
- analytic null simulated in chunks (was n_sim x K float64 at once, ~1.4 GB peak at K = 300);
  identical numbers (same RNG stream).
- module_context_profile(soft=False) rows renormalised (did not sum to 1 with support_min_t).
- selection='absolute': signed t of selected contexts kept (top_t_signed); pair() reports t (signed)
  and rank_stat.
- input checks: inf, duplicated row/gene labels, min_obs < 4, n_perm < 1.
- build_network computes support if needed; membership_matrix before split -> ValueError.
- CLI --power skips the scale-free scan.
Docs: analytic calibration is approximate for spearman/bicor; store_max_mb note for K = 300.
Demo unchanged after fixes (bias 0.0852, ARI 0.981). Tests: 29.

## 2026-11-06 — Python versions and the scipy 1.18 break in dynamicTreeCut

Fresh installs on Python 3.12–3.14 failed 6 tests: scipy 1.18 made `rankdata(method="ordinal")`
return float64, and dynamicTreeCut 0.1.1 uses those ranks as array indices (IndexError). Python 3.11
only worked because scipy 1.18 doesn't support 3.11, so pip installed 1.17. Fix:
`modules._cutree_hybrid()` swaps in a rankdata that casts ordinal ranks back to int, inside the
dynamicTreeCut module only. Verified: 33 tests pass on 3.10 (scipy 1.15), 3.11 (1.17), 3.12, 3.13
and 3.14 (1.18); modules, submodule memberships, summary z, pooled baseline and dominance flags are
identical between 3.11/scipy 1.17 and 3.13/scipy 1.18.
dynamicTreeCut also applies `~` to Python bools (DeprecationWarning since 3.12, an error in 3.16).
`~True` is -2, which is truthy, so `if ~DoMerge and nExternalSplits > 0` is wrong in that package,
but the branch only runs with minExternalSplit, which ctxwgcna doesn't use; the warning is filtered.
Before Python 3.16, vendor a fixed copy of the tree cut (check dynamicTreeCut's license first).
README now recommends 3.12.
