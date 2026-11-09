import warnings
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from ctxwgcna import ContextWGCNA, adjacency_from_r, global_wgcna, compute_support, context_tom, make_context_blocks
from ctxwgcna.network import scale_free_fit_index
from ctxwgcna.similarity import fisher_z, similarity_matrix
from ctxwgcna.simulate import simulate_lfc


@pytest.fixture
def X():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 12))
    X[:, 1] += 0.8 * X[:, 0]
    return X


def test_similarity_matches_reference(X):
    np.testing.assert_allclose(similarity_matrix(X, "pearson"), np.corrcoef(X, rowvar=False), atol=1e-12)
    np.testing.assert_allclose(similarity_matrix(X, "spearman"), spearmanr(X).statistic, atol=1e-12)
    Xn = X / np.linalg.norm(X, axis=0)
    np.testing.assert_allclose(similarity_matrix(X, "cosine"), Xn.T @ Xn, atol=1e-12)
    B = similarity_matrix(X, "bicor")
    assert np.allclose(B, B.T) and np.all(np.abs(B) <= 1) and B[0, 1] > 0.4


def _blocks(rng, n_ctx=7, n_obs=(15, 40), G=10):
    fc = []
    obs = []
    for k in range(n_ctx):
        n = rng.integers(*n_obs)
        fc.append(rng.standard_normal((n, G)))
        obs += [f"ctx{k}"] * n
    fc = pd.DataFrame(np.vstack(fc), columns=[f"g{i}" for i in range(G)])
    obs = pd.DataFrame({"system": obs}, index=fc.index)
    return make_context_blocks(fc, obs, "system", min_obs=5)


@pytest.mark.parametrize("selection", ["positive", "absolute"])
def test_streaming_top_n_matches_brute_force(selection):
    rng = np.random.default_rng(2)
    blocks = _blocks(rng)
    top_n = 3
    # brute force
    T = np.stack([fisher_z(similarity_matrix(b.X)) * np.sqrt(b.n_obs - 3) for b in blocks], axis=-1)
    key = np.abs(T) if selection == "absolute" else T
    order = np.argsort(-key, axis=-1)[..., :top_n]
    n = np.array([b.n_obs for b in blocks])
    s = np.sqrt(n - 3)[order]
    z_expected = ((s ** 2) * np.take_along_axis(key, order, -1) / s).sum(-1) / (s ** 2).sum(-1)
    off = ~np.eye(T.shape[0], dtype=bool)
    for chunk_mb in (1e-6, 128):  # one context per chunk vs all at once
        sup = compute_support(blocks, top_n=top_n, selection=selection, calibrate="none", chunk_mb=chunk_mb)
        np.testing.assert_array_equal(sup.top_idx[off], order[off])
        np.testing.assert_allclose(sup.z_raw[off], z_expected[off], rtol=1e-5)
        assert np.array_equal(sup.top_idx, sup.top_idx.transpose(1, 0, 2))


def _tom_reference(adj):
    """Plain-loop WGCNA TOM (TOMDenom='min'): (sum_{u!=i,j} a_iu a_uj + a_ij) / (min(k_i,k_j) + 1 - a_ij)."""
    G = len(adj)
    A = adj.copy()
    np.fill_diagonal(A, 0)
    k = A.sum(axis=1)
    T = np.eye(G)
    for i in range(G):
        for j in range(G):
            if i != j:
                shared = sum(A[i, u] * A[u, j] for u in range(G) if u not in (i, j))
                T[i, j] = (shared + A[i, j]) / (min(k[i], k[j]) + 1 - A[i, j])
    return T


def test_standard_tom_matches_reference():
    rng = np.random.default_rng(3)
    r = similarity_matrix(rng.standard_normal((30, 25)))
    adj = adjacency_from_r(r, 6, "unsigned")
    np.testing.assert_allclose(context_tom(adj, mode="standard"), _tom_reference(adj), atol=1e-12)


@pytest.mark.parametrize("seed,gen,expected", [
    # reference values from PyWGCNA 2.2.0 WGCNA.scaleFreeFitIndex (port of R WGCNA::scaleFreeFitIndex)
    (11, "exponential", (0.8699397265409788, -1.32539510373431, 0.9626398425011052)),
    (12, "pareto", (0.474366188060099, -4.638097206796633, 0.3591734086591222)),
])
def test_scale_free_fit_index_matches_pywgcna(seed, gen, expected):
    rng = np.random.default_rng(seed)
    k = rng.exponential(5, 500) if gen == "exponential" else rng.pareto(1.5, 500) + 1
    fit = scale_free_fit_index(k)
    got = (fit["Rsquared.SFT"], fit["slope.SFT"], fit["truncatedExponentialAdjRsquared"])
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_matched_tom_reduces_to_standard_with_one_context():
    rng = np.random.default_rng(4)
    G = 15
    fc = pd.DataFrame(rng.standard_normal((40, G)))
    obs = pd.DataFrame({"system": ["a"] * 40}, index=fc.index)
    sup = compute_support(make_context_blocks(fc, obs, "system"), top_n=1, calibrate="none")
    adj = adjacency_from_r(sup.r, 4, "signed")
    for mode in ("shared", "matched"):
        np.testing.assert_allclose(context_tom(adj, sup, mode=mode), context_tom(adj, mode="standard"))


def test_context_tom_bounded():
    rng = np.random.default_rng(5)
    blocks = _blocks(rng, G=20)
    sup = compute_support(blocks, top_n=2, calibrate="none")
    adj = adjacency_from_r(sup.r, 6, "signed")
    std = context_tom(adj, mode="standard")
    for mode in ("shared", "matched"):
        t = context_tom(adj, sup, mode=mode)
        assert t.min() >= 0 and t.max() <= 1 + 1e-12
        assert np.all(t <= std + 1e-12)


@pytest.mark.parametrize("calibrate", ["analytic", "permutation"])
def test_calibration_centres_null(calibrate):
    rng = np.random.default_rng(6)
    G, K, n = 60, 20, 30
    fc = pd.DataFrame(rng.standard_normal((K * n, G)))
    obs = pd.DataFrame({"system": np.repeat([f"s{k}" for k in range(K)], n)}, index=fc.index)
    sup = compute_support(make_context_blocks(fc, obs, "system"), top_n=3, calibrate=calibrate, n_perm=5)
    off = ~np.eye(G, dtype=bool)
    se = 1 / np.sqrt(n - 3)
    assert sup.z_raw[off].mean() > 1.0 * se  # top-3 of 20 is clearly inflated...
    assert abs(sup.z[off].mean()) < 0.1 * se  # ...and calibration removes it


def _majority(ct, t):
    row = ct.loc[t].drop("M0", errors="ignore")
    return row.idxmax(), row.max() / ct.loc[t].sum()


def pooled_wgcna(fc, power, min_module_size):
    """Reference standard WGCNA from the low-level functions (pooled correlation, standard TOM)."""
    from ctxwgcna import similarity_matrix
    from ctxwgcna.modules import detect_modules
    adj = adjacency_from_r(similarity_matrix(fc.to_numpy(), "pearson"), power, "signed")
    return detect_modules(context_tom(adj, mode="standard"), fc.columns, min_module_size=min_module_size)[0]


def test_simulation_recovers_local_modules_where_global_fails():
    fc, obs, truth = simulate_lfc(seed=0)
    glob = ContextWGCNA(fc).run(power=12, min_module_size=20)  # standard WGCNA
    cw = ContextWGCNA(fc, obs, context_col="system", top_n=2, tom_mode="matched")
    mods = cw.run(power=12, min_module_size=20)
    assert adjusted_rand_score(truth, mods) > 0.7
    assert adjusted_rand_score(truth, glob) < 0.3  # pooled correlation loses the local modules

    ct = pd.crosstab(truth, mods)
    majors = {}
    for t in ["G1", "L1", "L2"]:
        majors[t], frac = _majority(ct, t)
        assert frac >= 0.9, ct
    assert len(set(majors.values())) == 3, ct  # three distinct modules
    assert ct.loc["noise", "M0"] / ct.loc["noise"].sum() >= 0.9, ct  # noise stays unassigned

    prof = cw.module_context_profile()
    assert set(prof.loc[majors["L1"]].nlargest(2).index) == {"sys0", "sys1"}
    assert set(prof.loc[majors["L2"]].nlargest(2).index) == {"sys2", "sys3"}


def test_context_tom_removes_cross_context_overlap():
    """P-Q linked in sys6 and Q-R in sys7: shared neighbours Q must not raise TOM(P, R)."""
    fc, obs, truth = simulate_lfc(seed=0)
    T = truth.to_numpy()

    def block_mean(M, a, b):
        X = M[np.ix_(T == a, T == b)]
        return X[~np.eye(*X.shape, dtype=bool)].mean() if a == b else X.mean()

    ratio = {}
    for mode in ("standard", "matched"):
        cw = ContextWGCNA(fc, obs, top_n=2, tom_mode=mode, context_col="system")
        cw.compute_support()
        tom = cw.build_network(power=12)
        ratio[mode] = block_mean(tom, "P", "R") / block_mean(tom, "noise", "noise")
    assert ratio["standard"] > 1.1
    assert ratio["matched"] < 1.05


# -- second level: overlapping submodules ---------------------------------------------------------
from ctxwgcna.submodules import split_module


def test_split_module_overlapping_context_groups():
    # 3 contexts x 60 genes: ctx0 links genes 0-39, ctx1 links 20-59, ctx2 links 0-39 (same as ctx0)
    t = np.zeros((3, 60, 60), dtype=np.float32)
    for k, (a, b) in enumerate([(0, 40), (20, 60), (0, 40)]):
        t[k, a:b, a:b] = 5.0
    subs = split_module(t, t_thr=3, min_partners=10, min_size=20)
    got = sorted((tuple(s["contexts"]), tuple(np.flatnonzero(s["members"])[[0, -1]])) for s in subs)
    assert got == [((0, 2), (0, 39)), ((1,), (20, 59))]  # ctx0+ctx2 grouped; genes 20-39 in both


def test_split_module_pure_noise_gives_nothing():
    rng = np.random.default_rng(7)
    t = rng.standard_normal((10, 80, 80)).astype(np.float32)
    t = (t + t.transpose(0, 2, 1)) / np.sqrt(2)
    assert split_module(t, t_thr=3, min_partners=10, min_size=20) == []


def test_split_recovers_overlap_and_keeps_coherent_modules_whole():
    fc, obs, truth = simulate_lfc(seed=0)
    cw = ContextWGCNA(fc, obs, top_n=2, context_col="system")
    cw.run(power=12, min_module_size=20)
    subs, mem = cw.split_modules()
    M = cw.membership_matrix()
    by_truth = pd.concat([truth, M], axis=1).groupby("truth").sum()

    assert by_truth.loc["noise"].sum() == 0  # noise genes join no submodule
    q_subs = by_truth.columns[by_truth.loc["Q"] >= 36]  # Q (40 genes) in two submodules
    assert len(q_subs) == 2 and subs.loc[q_subs, "parent"].nunique() == 1
    ctx = {subs.loc[s, "contexts"]: s for s in q_subs}
    assert set(ctx) == {"sys6", "sys7"}
    assert by_truth.loc["P", ctx["sys6"]] >= 30 and by_truth.loc["P", ctx["sys7"]] == 0
    assert by_truth.loc["R", ctx["sys7"]] >= 36 and by_truth.loc["R", ctx["sys6"]] == 0
    for t, contexts in [("G1", "sys0,sys1,sys2,sys3,sys4,sys5,sys6,sys7,sys8,sys9"),
                        ("L1", "sys0,sys1"), ("L2", "sys2,sys3")]:
        s = by_truth.loc[t].idxmax()
        assert by_truth.loc[t, s] == 60 and subs.loc[s, "contexts"] == contexts
        assert (subs["parent"] == subs.loc[s, "parent"]).sum() == 1  # not split


def test_stored_and_recomputed_context_statistics_agree():
    fc, obs, _ = simulate_lfc(n_genes=320, n_systems=8, n_conditions=30, seed=3)
    a = ContextWGCNA(fc, obs, top_n=2, context_col="system")
    a.compute_support()
    b = ContextWGCNA(fc, obs, top_n=2, context_col="system")
    b.support = compute_support(b.blocks, top_n=2, store_context_t=False)
    idx = np.arange(0, 320, 7)
    np.testing.assert_allclose(a._context_t(idx), b._context_t(idx), rtol=1e-5, atol=1e-4)
    w = a.support.context_weights(idx)
    off = ~np.eye(len(idx), dtype=bool)
    np.testing.assert_allclose(w.sum(axis=0)[off][w.sum(axis=0)[off] > 0], 1.0, rtol=1e-5)


# -- dominant FC profiles ---------------------------------------------------------------------------
from ctxwgcna import detect_dominant, drop_dominant


def _add_response(fc, obs, cond, frac, effect, systems=None, seed=0):
    """Condition `cond`: a random `frac` of genes get a large FC (in `systems`, default all)."""
    rng = np.random.default_rng(seed)
    fc = fc.copy()
    hit = rng.random(fc.shape[1]) < frac
    eff = np.where(hit, rng.choice([-1, 1], fc.shape[1]) * rng.uniform(*effect, fc.shape[1]), 0.0)
    sel = obs.condition == cond
    if systems is not None:
        sel &= obs.system.isin(systems)
    fc.loc[sel] += eff
    return fc


@pytest.fixture(scope="module")
def dominance_data():
    fc, obs, _ = simulate_lfc(seed=4)  # 500 genes, 10 systems x 100 conditions
    fc = _add_response(fc, obs, "c000", 0.6, (10, 20), seed=1)                      # strong, everywhere
    fc = _add_response(fc, obs, "c001", 0.3, (4, 8), seed=2)                        # weaker, everywhere
    fc = _add_response(fc, obs, "c002", 0.5, (6, 12), systems=["sys0", "sys1"], seed=3)  # 2 systems only
    fc = _add_response(fc, obs, "c010", 0.06, (12, 18), seed=4)  # large but narrow: a real programme
    return fc, obs


def test_detects_multiple_and_system_specific_dominant_profiles(dominance_data):
    fc, obs = dominance_data
    rows, summary = detect_dominant(fc, obs, verbose=False)
    expected = obs.condition.isin(["c000", "c001"]) | ((obs.condition == "c002") & obs.system.isin(["sys0", "sys1"]))
    assert rows["dominant"].equals(pd.Series(expected.to_numpy(), index=fc.index, name="dominant"))
    assert summary.loc["c002", "n_dominant"] == 2
    # breadth summaries agree with the per-row values
    fe = rows.groupby("condition")["frac_genes_extreme"]
    for stat in ("max", "median", "mean", "min"):
        assert np.allclose(summary[f"{stat}_frac_genes_extreme"], fe.agg(stat).loc[summary.index])
    assert summary.loc["c002", "n_broad"] >= 2 and summary.loc["c010", "n_broad"] == 0
    # large but narrow (6% of genes) is kept: that is how a genuine few-condition programme looks
    assert not rows.loc[obs.condition == "c010", "dominant"].any()
    assert rows.loc[obs.condition == "c010", "magnitude_ratio"].min() > 5  # ...although its magnitude is large
    fc2, obs2 = drop_dominant(fc, obs, rows)
    assert len(fc2) == len(fc) - expected.sum() and fc2.index.equals(obs2.index)


def test_no_dominance_flags_nothing():
    fc, obs, _ = simulate_lfc(seed=5)
    rows, _ = detect_dominant(fc, obs, verbose=False)
    assert not rows["dominant"].any()


def test_cap_limits_flagged_fraction_and_warns(capsys):
    fc, obs, _ = simulate_lfc(seed=6)
    for i in range(10):  # 10% of the 100 conditions per system are inflated; cap is 5%
        fc = _add_response(fc, obs, f"c{50 + i:03d}", 0.5, (8, 16), seed=10 + i)
    rows, _ = detect_dominant(fc, obs, max_frac_flagged=0.05)
    per_sys = rows.groupby("context")[["candidate", "dominant"]].sum()
    assert (per_sys["candidate"] == 10).all() and (per_sys["dominant"] == 5).all()
    assert "WARNING" in capsys.readouterr().out


# --- regression tests for the external code review (Codex / Kimi, 2026-11-06) ---------------------

def _single_context(X, ctx="s"):
    idx = [f"r{i}" for i in range(X.shape[0])]
    fc = pd.DataFrame(X, index=idx, columns=[f"g{j}" for j in range(X.shape[1])])
    return fc, pd.DataFrame({"system": ctx, "condition": idx}, index=idx)


def test_dominance_small_context_with_flat_genes_is_detected():
    # 20 rows; most genes exactly 0 except in row 0 -> MAD = 0; previously |z| was capped at 4.3 < 5
    rng = np.random.default_rng(0)
    X = np.zeros((20, 500))
    X[:, 400:] = rng.normal(0, 1, (20, 100))
    X[0, :200] = 8
    rows, _ = detect_dominant(*_single_context(X), verbose=False)
    assert rows["dominant"].tolist() == [True] + [False] * 19
    assert rows.iloc[0]["frac_genes_extreme"] >= 0.4


def test_dominance_zero_median_magnitude():
    X = np.zeros((20, 500))
    X[0, :200] = 8  # broad
    X[1, :3] = 1  # changed but narrow
    rows, _ = detect_dominant(*_single_context(X), verbose=False)
    assert rows["dominant"].tolist() == [True] + [False] * 19


def test_dominance_sparse_null_flags_nothing():
    rng = np.random.default_rng(1)
    fc, obs = [], []
    for s in range(3):
        X = rng.normal(0, 1, (20, 400)) * (rng.random((20, 400)) < 0.1)
        f, o = _single_context(X, ctx=f"sys{s}")
        f.index = o.index = [f"sys{s}|{i}" for i in f.index]
        fc.append(f)
        obs.append(o)
    rows, _ = detect_dominant(pd.concat(fc), pd.concat(obs), verbose=False)
    assert not rows["dominant"].any()


def test_dominance_validates_columns_and_labels():
    fc, obs = _single_context(np.random.default_rng(0).normal(size=(20, 30)))
    with pytest.raises(ValueError, match="not a column"):
        detect_dominant(fc, obs.drop(columns="condition"), verbose=False)
    rows, summary = detect_dominant(fc, obs.drop(columns="condition"), condition_col=None, verbose=False)
    assert summary is None and len(rows) == 20
    with pytest.raises(ValueError, match="duplicated"):
        detect_dominant(fc.iloc[[0, 0, 1]], obs.iloc[[0, 0, 1]], verbose=False)


def test_dominance_rejects_nan_rows():
    # one untested (all-NaN) row used to make its whole context unscorable without any message
    fc, obs = _single_context(np.random.default_rng(0).normal(size=(20, 30)))
    fc.iloc[3] = np.nan
    with pytest.raises(ValueError, match="entirely NaN"):
        detect_dominant(fc, obs, verbose=False)
    keep = fc.notna().any(axis=1)
    rows, _ = detect_dominant(fc[keep], obs[keep], verbose=False)
    assert len(rows) == 19 and rows["magnitude_ratio"].notna().all()


def test_dominance_nan_options(dominance_data):
    fc, obs = dominance_data
    fc = fc.copy()
    untested = fc.index[obs.condition == "c003"][:2]
    fc.loc[untested] = np.nan
    keep = fc.notna().any(axis=1)
    ref_rows, ref_sum = detect_dominant(fc[keep], obs[keep], verbose=False)
    # skip: identical to dropping the all-NaN rows; those rows come back unscored, never dominant
    rows, summary = detect_dominant(fc, obs, nan="skip", verbose=False)
    pd.testing.assert_frame_equal(rows.loc[keep], ref_rows)
    assert rows.loc[untested, "magnitude_ratio"].isna().all() and not rows.loc[untested, "dominant"].any()
    assert summary.loc["c003", "n_contexts"] == ref_sum.loc["c003", "n_contexts"]
    assert rows.attrs["preprocessing"]["n_all_nan_rows"] == 2
    # fill: NaN -> 0, untested rows are scored as "no change"
    rows0, _ = detect_dominant(fc, obs, nan=0, verbose=False)
    assert (rows0.loc[untested, "magnitude_ratio"] == 0).all()
    # skip with a few NaN genes in a dominant row: still flagged
    fc2 = fc[keep].copy()
    dom_row = ref_rows.index[ref_rows.dominant][0]
    fc2.loc[dom_row, fc2.columns[:5]] = np.nan
    rows2, _ = detect_dominant(fc2, obs[keep], nan="skip", verbose=False)
    assert rows2.loc[dom_row, "dominant"]
    with pytest.raises(ValueError, match="nan='skip'"):
        detect_dominant(fc, obs, verbose=False)
    with pytest.raises(ValueError, match="nan must be"):
        detect_dominant(fc, obs, nan="drop", verbose=False)


def test_dominance_clip_options(dominance_data):
    fc, obs = dominance_data
    a, _ = detect_dominant(fc, obs, clip=2, verbose=False)
    b, _ = detect_dominant(fc.clip(-2, 2), obs, verbose=False)
    pd.testing.assert_frame_equal(a, b)
    c, _ = detect_dominant(fc, obs, clip=(-2, 2), verbose=False)
    pd.testing.assert_frame_equal(a, c)
    p, _ = detect_dominant(fc, obs, clip="p95", verbose=False)
    lo, hi = p.attrs["preprocessing"]["clip_bounds"]
    assert np.isclose(hi, np.percentile(np.abs(fc.to_numpy()), 95)) and lo == -hi
    assert abs(p.attrs["preprocessing"]["frac_clipped"] - 0.05) < 0.01
    # infinite values are accepted once clipped
    fc_inf = fc.copy()
    fc_inf.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        detect_dominant(fc_inf, obs, verbose=False)
    detect_dominant(fc_inf, obs, clip=5, verbose=False)
    for bad in ("q95", "p0", -1, (2, -2)):
        with pytest.raises(ValueError):
            detect_dominant(fc, obs, clip=bad, verbose=False)


@pytest.fixture(scope="module")
def small_data():
    fc, obs, _ = simulate_lfc(n_genes=320, n_systems=8, n_conditions=30, seed=6)
    return fc, obs


def test_contextwgcna_nan_and_clip(small_data):
    fc, obs = small_data
    base = ContextWGCNA(fc, obs, calibrate="none", context_col="system")
    assert base.data is fc and base.preprocessing["clip_bounds"] is None  # untouched by default
    # skip: all-NaN rows dropped, identical to dropping them by hand
    fc_nan = fc.copy()
    fc_nan.iloc[[0, 5]] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ContextWGCNA(fc_nan, obs, calibrate="none", context_col="system")
    skip = ContextWGCNA(fc_nan, obs, calibrate="none", nan="skip", context_col="system")
    keep = fc_nan.notna().any(axis=1)
    manual = ContextWGCNA(fc_nan[keep], obs[keep], calibrate="none", context_col="system")
    assert len(skip.data) == len(fc) - 2 and skip.data.index.equals(manual.data.index)
    np.testing.assert_array_equal(skip.compute_support().r, manual.compute_support().r)
    # partly-NaN rows cannot be skipped; filling works
    fc_part = fc.copy()
    fc_part.iloc[1, :3] = np.nan
    with pytest.raises(ValueError, match="partly NaN"):
        ContextWGCNA(fc_part, obs, calibrate="none", nan="skip", context_col="system")
    filled = ContextWGCNA(fc_part, obs, calibrate="none", nan=0, context_col="system")
    assert (filled.data.iloc[1, :3] == 0).all()
    # clip: identical to clipping by hand; percentile bounds recorded
    a = ContextWGCNA(fc, obs, calibrate="none", clip=1.5, context_col="system").compute_support().r
    b = ContextWGCNA(fc.clip(-1.5, 1.5), obs, calibrate="none", context_col="system").compute_support().r
    np.testing.assert_array_equal(a, b)
    p = ContextWGCNA(fc, obs, calibrate="none", clip="p95", context_col="system")
    assert np.isclose(p.preprocessing["clip_bounds"][1], np.percentile(np.abs(fc.to_numpy()), 95))
    assert np.abs(p.data.to_numpy()).max() <= p.preprocessing["clip_bounds"][1]


def test_linear_fc_warning(dominance_data, capsys):
    fc, obs = dominance_data
    detect_dominant(fc, obs)
    assert "linear-scale" not in capsys.readouterr().out
    detect_dominant(2.0 ** fc, obs)  # linear FC: all positive
    assert "linear-scale" in capsys.readouterr().out
    # log base does not matter: log2 -> ln flags the same rows
    a, _ = detect_dominant(fc, obs, verbose=False)
    b, _ = detect_dominant(fc * np.log(2), obs, verbose=False)
    assert a["dominant"].equals(b["dominant"])


def test_cli_fc_alias():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_ctxwgcna", "scripts/run_ctxwgcna.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for flag in ("--data", "--fc"):
        args = cli.parse_args([flag, "x.csv", "--obs", "o.csv", "--outdir", "out"])
        assert args.data == "x.csv"
    # old dominance flag names still work
    old = cli.parse_args(["--simulate", "--outdir", "o", "--dominant-min-fold", "3", "--dominant-label-col", "c",
                          "--dominant-max-frac", "0.1", "--dominant-min-z", "2", "--dominant-min-gene-frac", "0.2"])
    new = cli.parse_args(["--simulate", "--outdir", "o", "--dominant-min-magnitude-ratio", "3",
                          "--dominant-condition-col", "c", "--dominant-max-frac-flagged", "0.1",
                          "--dominant-min-magnitude-z", "2", "--dominant-min-frac-genes-extreme", "0.2"])
    assert vars(old) == vars(new)


def test_dominance_old_parameter_names(dominance_data):
    fc, obs = dominance_data
    new, s_new = detect_dominant(fc, obs, min_magnitude_ratio=4, min_magnitude_z=3, min_frac_genes_extreme=0.2,
                                 gene_extreme_z=4, max_frac_flagged=0.1, condition_col="condition", verbose=False)
    with pytest.warns(FutureWarning, match="is now called") as record:
        old, s_old = detect_dominant(fc, obs, min_fold=4, min_z=3, min_gene_frac=0.2, extreme_z=4,
                                     max_frac=0.1, label_col="condition", verbose=False)
    assert len(record) == 6 and "min_fold= is now called min_magnitude_ratio=" in str(record[0].message)
    pd.testing.assert_frame_equal(new, old)
    pd.testing.assert_frame_equal(s_new, s_old)
    with pytest.raises(TypeError, match="min_foldx"):
        detect_dominant(fc, obs, min_foldx=4, verbose=False)


def test_artifact_row_exclusion(small_data):
    lfc, obs = small_data
    rng = np.random.default_rng(3)
    # background artifacts in every row (~1% of genes), and three rows with 30% (too few cells)
    art = rng.random(lfc.shape) < 0.01
    bad = [np.flatnonzero((obs.system == s).to_numpy())[1] for s in obs.system.unique()[:3]]
    art[bad] = rng.random((3, lfc.shape[1])) < 0.3
    data = lfc.mask(art, 28.0 * np.sign(rng.standard_normal(lfc.shape)))
    rows, summary = detect_dominant(data, obs, max_abs=15, max_artifact_z=3, verbose=False)
    assert rows.index[rows.artifact_excluded].tolist() == lfc.index[bad].tolist()
    assert rows.loc[rows.artifact_excluded, "magnitude_ratio"].isna().all()
    assert not rows.loc[rows.artifact_excluded, "dominant"].any()
    assert np.allclose(rows["frac_masked"], art.sum(axis=1) / lfc.shape[1])
    assert summary["n_artifact_excluded"].sum() == 3 and rows.attrs["preprocessing"]["n_artifact_rows"] == 3
    # the remaining rows are scored as if the excluded rows were not there
    keep = ~rows.artifact_excluded.to_numpy()
    ref, _ = detect_dominant(data[keep], obs[keep], max_abs=15, verbose=False)
    cols = ["magnitude_ratio", "magnitude_z", "frac_genes_extreme", "dominant"]
    pd.testing.assert_frame_equal(rows.loc[keep, cols], ref[cols])
    # default (None): nothing excluded, z still reported
    rows_all, _ = detect_dominant(data, obs, max_abs=15, verbose=False)
    assert not rows_all.artifact_excluded.any() and (rows_all.loc[lfc.index[bad], "artifact_z"] > 3).all()
    # network: the same rows are left out
    cw = ContextWGCNA(data, obs, calibrate="none", max_abs=15, max_artifact_z=3, context_col="system")
    assert len(cw.data) == len(lfc) - 3 and not cw.data.index.isin(lfc.index[bad]).any()
    with pytest.raises(ValueError, match="context_col"):
        from ctxwgcna.data import prepare_frames
        prepare_frames(data, None, max_abs=15, max_artifact_z=3)


ARTIFACT_COLS = ["frac_masked", "artifact_z", "artifact_excluded"]

def test_max_abs_masking(dominance_data, small_data):
    fc, obs = dominance_data
    rng = np.random.default_rng(0)
    art = rng.random(fc.shape) < 0.02  # GLM-style artifacts: |LFC| ~ 30 from zero counts
    assert np.abs(fc.to_numpy()).max() < 25  # real effects in this fixture reach ~22
    art[0, 0] = True
    fc_art = fc.mask(art, 30.0 * np.sign(rng.standard_normal(fc.shape)))
    fc_art.iloc[0, 0] = np.inf
    expected, _ = detect_dominant(fc.mask(art), obs, nan="skip", verbose=False)
    # masked entries are ignored like NaN, also under nan='raise' (they never trigger the error)
    for nan in ("raise", "skip"):
        rows, _ = detect_dominant(fc_art, obs, nan=nan, max_abs=25, verbose=False)
        assert rows.attrs["preprocessing"]["n_masked"] == art.sum()
        pd.testing.assert_frame_equal(rows.drop(columns=ARTIFACT_COLS), expected.drop(columns=ARTIFACT_COLS))
        assert np.allclose(rows["frac_masked"], art.sum(axis=1) / fc.shape[1])
    # filled instead when nan is a number
    rows0, _ = detect_dominant(fc_art, obs, nan=0, max_abs=25, verbose=False)
    ref0, _ = detect_dominant(fc_art.mask(art, 0.0), obs, verbose=False)
    pd.testing.assert_frame_equal(rows0.drop(columns=ARTIFACT_COLS), ref0.drop(columns=ARTIFACT_COLS))
    with pytest.raises(ValueError, match="max_abs"):
        detect_dominant(fc, obs, max_abs=0, verbose=False)
    # network: masked values become 0, then the percentile clip is computed
    lfc, obs2 = small_data
    lfc_art = lfc.copy()
    lfc_art.iloc[3, :5] = 35.0
    cw = ContextWGCNA(lfc_art, obs2, calibrate="none", max_abs=15, clip="p95", context_col="system")
    ref = lfc_art.mask(lfc_art.abs() > 15, 0.0)
    assert cw.preprocessing["n_masked"] == 5 and (cw.data.iloc[3, :5] == 0).all()
    assert np.isclose(cw.preprocessing["clip_bounds"][1], np.percentile(np.abs(ref.to_numpy()), 95))
    np.testing.assert_array_equal(
        cw.compute_support().r,
        ContextWGCNA(ref.clip(*cw.preprocessing["clip_bounds"]), obs2, calibrate="none", context_col="system").compute_support().r)


def test_max_abs_auto(small_data):
    from ctxwgcna.data import auto_max_abs
    lfc, obs = small_data
    rng = np.random.default_rng(1)
    genuine = np.abs(lfc.to_numpy()).max()
    # no separate mode: nothing masked (also for heavy-tailed noise)
    assert auto_max_abs(lfc.to_numpy()) is None
    assert auto_max_abs(rng.standard_t(2, size=(1000, 300))) is None
    rows, _ = detect_dominant(lfc, obs, max_abs="auto", verbose=False)
    assert rows.attrs["preprocessing"]["n_masked"] == 0
    # GLM-style artifacts at |LFC| ~ 28: cutoff in the gap, same result as the manual mask
    art = rng.random(lfc.shape) < 0.01
    lfc_art = lfc.mask(art, rng.normal(28, 3, lfc.shape) * np.sign(rng.standard_normal(lfc.shape)))
    cutoff, mode = auto_max_abs(lfc_art.to_numpy())
    assert genuine < cutoff < np.abs(lfc_art.to_numpy()[art]).min() and 20 < mode < 35
    rows, _ = detect_dominant(lfc_art, obs, max_abs="auto", verbose=False)
    ref, _ = detect_dominant(lfc_art, obs, max_abs=cutoff, verbose=False)
    pd.testing.assert_frame_equal(rows, ref)
    assert rows.attrs["preprocessing"]["n_masked"] == art.sum()
    cw = ContextWGCNA(lfc_art, obs, calibrate="none", max_abs="auto", context_col="system")
    assert cw.preprocessing["max_abs_rule"] == "auto" and np.isclose(cw.preprocessing["max_abs"], cutoff)
    with pytest.raises(ValueError, match="max_abs"):
        detect_dominant(lfc, obs, max_abs="p99", verbose=False)
    # a genuine strong response (many large values in a few rows) is a mode too, but concentrated:
    # not masked, so the default 'auto' leaves dominant profiles alone
    strong = lfc.copy()
    picks = [np.flatnonzero((obs.system == sysname).to_numpy())[0] for sysname in obs.system.unique()[:3]]
    strong.iloc[picks, :150] += 15  # one strong row in each of three contexts
    info = auto_max_abs(strong.to_numpy(), details=True)
    assert info["cutoff"] is None and info["mode"] > 10 and info["row_frac"] < 0.25
    rows_auto, _ = detect_dominant(strong, obs, verbose=False)  # default max_abs='auto'
    rows_off, _ = detect_dominant(strong, obs, max_abs=None, verbose=False)
    pd.testing.assert_frame_equal(rows_auto, rows_off)
    assert rows_auto["dominant"].iloc[picks].all()


def test_pick_power_all_nan_raises():
    from ctxwgcna.network import pick_power
    t = pd.DataFrame({"power": [1, 2], "signed R.sq": [np.nan, np.nan], "mean(k)": [1.0, 1.0]})
    with pytest.raises(ValueError, match="set the power"):
        pick_power(t)
    t.loc[1, "signed R.sq"] = 0.3
    assert pick_power(t) == (2, False)


def test_analytic_null_chunking_is_exact():
    from ctxwgcna.support import _analytic_null
    n = np.array([20, 30, 25, 40, 22])
    a = _analytic_null(n, 2, "positive", "n", 50_000, np.random.default_rng(3), chunk=50_000)
    b = _analytic_null(n, 2, "positive", "n", 50_000, np.random.default_rng(3), chunk=7_000)
    np.testing.assert_allclose(a, b, rtol=1e-10)


def test_absolute_selection_keeps_sign():
    rng = np.random.default_rng(4)
    blocks = _blocks(rng)
    x = rng.standard_normal(blocks[2].n_obs)
    blocks[2].X[:, 0], blocks[2].X[:, 1] = x, -x + 0.05 * rng.standard_normal(len(x))  # strong negative
    sup = compute_support(blocks, top_n=3, selection="absolute", calibrate="none")
    T = np.stack([fisher_z(similarity_matrix(b.X)) * np.sqrt(b.n_obs - 3) for b in blocks])
    expect = np.take_along_axis(T.transpose(1, 2, 0), sup.top_idx.astype(int), -1)
    np.testing.assert_allclose(sup.top_t_signed, expect, rtol=1e-5, atol=1e-5)
    p = sup.pair(0, 1)
    assert p.iloc[0]["context"] == blocks[2].name and p.iloc[0]["t"] < 0 < p.iloc[0]["rank_stat"]
    pos = compute_support(blocks, top_n=3, selection="positive", calibrate="none")
    assert pos.top_t_signed is None and (pos.pair(0, 1)["t"] == pos.pair(0, 1)["rank_stat"]).all()


def test_input_validation():
    rng = np.random.default_rng(0)
    fc = pd.DataFrame(rng.normal(size=(20, 5)), columns=list("abcde"))
    obs = pd.DataFrame({"system": ["a"] * 20})
    bad = fc.copy()
    bad.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        make_context_blocks(bad, obs, "system", min_obs=5)
    with pytest.raises(ValueError, match="duplicated"):
        make_context_blocks(fc.set_axis(list("abcda"), axis=1), obs, "system", min_obs=5)
    with pytest.raises(ValueError, match="min_obs"):
        make_context_blocks(fc, obs, "system", min_obs=3)
    with pytest.raises(ValueError, match="n_perm"):
        compute_support(make_context_blocks(fc, obs, "system", min_obs=5), top_n=1,
                        calibrate="permutation", n_perm=0)


@pytest.fixture(scope="module")
def small_run():
    fc, obs, truth = simulate_lfc(n_genes=320, seed=5)
    cw = ContextWGCNA(fc, obs, support_min_t=2.0, context_col="system")
    return cw


def test_stepwise_api_and_profile_normalisation(small_run):
    cw = small_run
    with pytest.raises(ValueError, match="split_modules"):
        cw.membership_matrix()
    cw.build_network(power=12)  # computes the support itself
    cw.find_modules()
    prof = cw.module_context_profile(soft=False)
    s = prof.sum(axis=1)
    assert np.allclose(s[s > 0], 1.0)


def test_cli_given_power_skips_scan(tmp_path):
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "run_ctxwgcna", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_ctxwgcna.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--simulate", "--power", "12", "--no-split", "--outdir", str(tmp_path)])
    assert (tmp_path / "modules.csv").exists() and not (tmp_path / "soft_threshold.csv").exists()


# --- AnnData input -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def adata_example():
    anndata = pytest.importorskip("anndata")
    from scipy import sparse
    fc, obs, truth = simulate_lfc(n_genes=320, n_systems=8, n_conditions=40, seed=7)
    ad = anndata.AnnData(X=np.zeros(fc.shape), obs=obs.copy(),
                         var=pd.DataFrame({"keep": np.arange(fc.shape[1]) < 150}, index=fc.columns))
    ad.layers["log2fc"] = sparse.csr_matrix(fc.to_numpy())  # sparse layer, X deliberately different
    return ad, fc, obs


def test_anndata_input_matches_dataframe(adata_example):
    ad, fc, obs = adata_example
    a = ContextWGCNA(ad, layer="log2fc", context_col="system").compute_support()
    b = ContextWGCNA(fc, obs, context_col="system").compute_support()
    np.testing.assert_array_equal(a.z, b.z)
    assert list(a.genes) == list(fc.columns)
    # gene subset by boolean .var column, and by list of names
    sub = ContextWGCNA(ad, layer="log2fc", genes="keep", context_col="system")
    assert list(sub.genes) == list(fc.columns[:150])
    assert list(ContextWGCNA(ad, layer="log2fc", genes=["g0003", "g0001"]).genes) == ["g0003", "g0001"]
    labels = ContextWGCNA(ad, layer="log2fc").run(power=12, min_module_size=20)
    assert list(labels.index) == list(fc.columns)


def test_anndata_errors(adata_example):
    ad, fc, obs = adata_example
    with pytest.raises(ValueError, match="layer 'nope' not found"):
        ContextWGCNA(ad, layer="nope")
    with pytest.raises(ValueError, match="only used with AnnData"):
        ContextWGCNA(fc, obs, layer="log2fc")
    with pytest.raises(ValueError, match="obs .* is required"):
        ContextWGCNA(fc, context_col="system")
    with pytest.raises(ValueError, match="not a column of adata.var"):
        ContextWGCNA(ad, layer="log2fc", genes="nope")


def test_anndata_dominance(adata_example):
    ad, fc, obs = adata_example
    ad = ad.copy()
    X = ad.layers["log2fc"].toarray()
    first = ad.obs_names[ad.obs["system"] == "sys0"][0]
    X[ad.obs_names.get_loc(first), :120] += 15  # one broad, dominant profile
    ad.layers["log2fc"] = X
    rows, _ = detect_dominant(ad, layer="log2fc", verbose=False)
    assert rows.index[rows["dominant"]].tolist() == [first]
    kept = drop_dominant(ad, rows=rows)
    assert kept.n_obs == ad.n_obs - 1 and first not in kept.obs_names


def test_cli_h5ad(adata_example, tmp_path):
    import importlib.util, os
    ad, fc, obs = adata_example
    path = tmp_path / "data.h5ad"
    ad.write_h5ad(path)
    spec = importlib.util.spec_from_file_location(
        "run_ctxwgcna", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_ctxwgcna.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    mod.main(["--h5ad", str(path), "--layer", "log2fc", "--var-subset", "keep", "--power", "12",
              "--min-module-size", "20", "--no-split", "--outdir", str(out)])
    assert len(pd.read_csv(out / "modules.csv", index_col=0)) == 150


def test_steps_clear_only_downstream_results():
    fc, obs, _ = simulate_lfc(n_genes=320, n_systems=8, n_conditions=30, seed=3)
    cw = ContextWGCNA(fc, obs, top_n=2, calibrate="none", context_col="system")
    cw.run(power=12, min_module_size=10)
    cw.split_modules()
    support, tom, modules = cw.support, cw.tom, cw.modules

    cw.split_modules(t_thr=4)                       # submodules only
    assert cw.modules is modules and cw.tom is tom
    cw.find_modules(min_module_size=15)             # modules: TOM kept, submodules cleared
    assert cw.tom is tom and cw.submodules is None
    cw.split_modules()
    cw.build_network(power=8)                       # network: support kept, modules cleared
    assert cw.support is support and cw.modules is None and cw.submodules is None
    assert cw.power == 8

    cw.pick_power()                                 # picked power: network cleared, support kept
    assert cw.support is support and cw.tom is None and cw.sft is not None
    cw.build_network(power=12)
    cw.find_modules(min_module_size=10)
    cw.compute_support()                            # new support: everything after it cleared
    assert cw.tom is None and cw.modules is None and cw.sft is None
    assert cw.power == 12                           # a power given by the user is kept

    # changed settings are noticed: the stale support / TOM is rebuilt by the next step
    cw.find_modules(min_module_size=10)
    old_support = cw.support
    cw.top_n = 3
    with pytest.raises(ValueError, match="settings changed"):
        cw.split_modules()
    cw.find_modules(min_module_size=10)
    assert cw.support is not old_support and cw.support.top_n == 3
    tom = cw.tom
    cw.tom_mode = "standard"
    cw.find_modules(min_module_size=10)
    assert cw.tom is not tom and cw.support.top_n == 3
    cw.network_type = "unsigned"                    # selection follows network_type
    assert cw.selection == "absolute"


def test_single_context_equals_standard_wgcna():
    fc, obs, _ = simulate_lfc(n_genes=320, n_systems=8, n_conditions=30, seed=3)
    pooled = pooled_wgcna(fc, 12, 20)
    cw = ContextWGCNA(fc, obs.assign(all="all"), context_col="all", top_n=1, calibrate="none",
                      tom_mode="standard")
    assert (cw.run(power=12, min_module_size=20) == pooled).all()
    one = ContextWGCNA(fc, obs, contexts=["sys0"], top_n=1, calibrate="none", tom_mode="standard", context_col="system")
    sys0 = pooled_wgcna(fc[obs["system"] == "sys0"], 12, 20)
    assert (one.run(power=12, min_module_size=20) == sys0).all()


def test_context_col_none_is_standard_wgcna():
    fc, obs, _ = simulate_lfc(n_genes=320, n_systems=8, n_conditions=30, seed=3)
    pooled = pooled_wgcna(fc, 12, 20)
    cw = ContextWGCNA(fc)                                   # default: no obs needed
    assert (cw.top_n, cw.calibrate, cw.tom_mode) == (1, "none", "standard")
    assert [b.name for b in cw.blocks] == ["all"] and cw.context_col is None
    assert (cw.run(power=12, min_module_size=20) == pooled).all()
    with pytest.warns(UserWarning, match="runs standard WGCNA.*top_n=3.*tom_mode='matched'"):
        cw2 = ContextWGCNA(fc, obs, top_n=3, tom_mode="matched")
    assert (cw2.top_n, cw2.tom_mode) == (1, "standard") and list(cw2.obs.columns) == list(obs.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # standard values given: no warning
        ContextWGCNA(fc, obs, top_n=1, calibrate="none", tom_mode="standard")
    ctx = ContextWGCNA(fc, obs, context_col="system")       # context-aware defaults
    assert (ctx.top_n, ctx.calibrate, ctx.tom_mode) == (3, "analytic", "matched")


def test_global_wgcna_deprecated():
    fc, _, _ = simulate_lfc(n_genes=320, n_systems=8, n_conditions=30, seed=3)
    with pytest.warns(FutureWarning, match="global_wgcna is deprecated"):
        labels, power = global_wgcna(fc, power=12, min_module_size=20)
    assert power == 12 and (labels == pooled_wgcna(fc, 12, 20)).all()
