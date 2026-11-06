import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from ctxwgcna import ContextWGCNA, adjacency_from_r, global_wgcna, compute_support, context_tom, make_context_blocks
from ctxwgcna.network import scale_free_fit_index
from ctxwgcna.similarity import fisher_z, similarity_matrix
from ctxwgcna.simulate import simulate_fc


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


def test_simulation_recovers_local_modules_where_global_fails():
    fc, obs, truth = simulate_fc(seed=0)
    glob, _ = global_wgcna(fc, power=12, min_module_size=20)
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
    fc, obs, truth = simulate_fc(seed=0)
    T = truth.to_numpy()

    def block_mean(M, a, b):
        X = M[np.ix_(T == a, T == b)]
        return X[~np.eye(*X.shape, dtype=bool)].mean() if a == b else X.mean()

    ratio = {}
    for mode in ("standard", "matched"):
        cw = ContextWGCNA(fc, obs, top_n=2, tom_mode=mode)
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
    fc, obs, truth = simulate_fc(seed=0)
    cw = ContextWGCNA(fc, obs, top_n=2)
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
    fc, obs, _ = simulate_fc(n_genes=320, n_systems=8, n_conditions=30, seed=3)
    a = ContextWGCNA(fc, obs, top_n=2)
    a.compute_support()
    b = ContextWGCNA(fc, obs, top_n=2)
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
    fc, obs, _ = simulate_fc(seed=4)  # 500 genes, 10 systems x 100 conditions
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
    # large but narrow (6% of genes) is kept: that is how a genuine few-condition programme looks
    assert not rows.loc[obs.condition == "c010", "dominant"].any()
    assert rows.loc[obs.condition == "c010", "fold"].min() > 5  # ...although its magnitude is large
    fc2, obs2 = drop_dominant(fc, obs, rows)
    assert len(fc2) == len(fc) - expected.sum() and fc2.index.equals(obs2.index)


def test_no_dominance_flags_nothing():
    fc, obs, _ = simulate_fc(seed=5)
    rows, _ = detect_dominant(fc, obs, verbose=False)
    assert not rows["dominant"].any()


def test_cap_limits_flagged_fraction_and_warns(capsys):
    fc, obs, _ = simulate_fc(seed=6)
    for i in range(10):  # 10% of the 100 conditions per system are inflated; cap is 5%
        fc = _add_response(fc, obs, f"c{50 + i:03d}", 0.5, (8, 16), seed=10 + i)
    rows, _ = detect_dominant(fc, obs, max_frac=0.05)
    per_sys = rows.groupby("context")[["candidate", "dominant"]].sum()
    assert (per_sys["candidate"] == 10).all() and (per_sys["dominant"] == 5).all()
    assert "WARNING" in capsys.readouterr().out
