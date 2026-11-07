#!/usr/bin/env python
"""Draw the method schematic used in README.md (docs/schematic.png and docs/schematic-dark.png).

The data panels use a small simulation (360 genes, 20 systems x 100 conditions) with the planted
sets of ctxwgcna.simulate: G1 (all systems), L1 (sys0-1), L2 (sys2-3), P+Q (sys6) and Q+R (sys7).

    pip install -e ".[docs]"      # matplotlib
    python scripts/make_schematic.py
"""
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from ctxwgcna import ContextWGCNA, global_wgcna, similarity_matrix  # noqa: E402
from ctxwgcna.simulate import simulate_lfc  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs")
N_SYSTEMS, POWER = 20, 12

# Colours: two context identities (sys6, sys7) from a validated categorical pair, neutral greys for
# everything else, and a one-hue sequential ramp for the heatmaps. Dark mode uses the dark steps.
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  base="#c3c2b7", ctx_a="#2a78d6", ctx_b="#eb6834", other="#c3c2b7", strong="#52514e",
                  seq=["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 base="#383835", ctx_a="#3987e5", ctx_b="#d95926", other="#5a5955", strong="#c3c2b7",
                 seq=["#1a1a19", "#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]),
}
# planted gene sets of the simulation: (name, first gene, last gene + 1)
SETS = [("G1", 0, 60), ("L1", 60, 120), ("L2", 120, 180), ("P", 180, 220), ("Q", 220, 260),
        ("R", 260, 300), ("noise", 300, 360)]
PAIR = (225, 235)  # two Q genes: co-regulated in sys6 (with P) and in sys7 (with R)


# --- data -----------------------------------------------------------------------------------
def recovery(truth, modules):
    """Per planted set: fraction of its genes in its most common (assigned) module."""
    ct = pd.crosstab(truth, modules)
    ct = ct.drop(columns="M0", errors="ignore")
    size = truth.value_counts()
    return {name: (ct.loc[name].max() / size[name] if ct.shape[1] else 0.0) for name, _, _ in SETS[:-1]}


def run_example():
    """Simulate, run ContextWGCNA and the pooled baseline, and collect what the panels need."""
    lfc, obs, truth = simulate_lfc(n_genes=360, n_systems=N_SYSTEMS, n_conditions=100, seed=0)
    cw = ContextWGCNA(lfc, obs)
    cw.run(power=POWER, min_module_size=20)
    cw.split_modules()
    pooled_modules, _ = global_wgcna(lfc, power=POWER, min_module_size=20)
    pooled_r = similarity_matrix(lfc.to_numpy())
    i, j = PAIR
    module_of_q = cw.modules.iloc[i]
    return dict(
        pooled_r=pooled_r, union_r=cw.support.r, contexts=cw.support.contexts,
        t_pair=cw.support.t_all[:, i, j], top=list(cw.support.top_idx[i, j]),
        pooled_pair=pooled_r[i, j], union_pair=cw.support.r[i, j],
        module=module_of_q, module_genes=np.flatnonzero((cw.modules == module_of_q).to_numpy()),
        subs=cw.submodules[cw.submodules.parent == module_of_q],
        mem=cw.memberships[cw.memberships.parent == module_of_q], genes=cw.genes,
        rec_pooled=recovery(truth, pooled_modules), rec_ctx=recovery(truth, cw.modules))


# --- helpers --------------------------------------------------------------------------------
def style_axes(ax, th):
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("bottom", "left"))
        spine.set_color(th["base"])
    ax.tick_params(colors=th["muted"], labelcolor=th["ink2"], labelsize=8, length=3)
    ax.set_facecolor(th["surface"])
    ax.grid(axis="y", color=th["grid"], lw=0.6)
    ax.set_axisbelow(True)


def panel_title(ax, th, number, title, subtitle):
    ax.text(0, 1.20, f"{number}  {title}", transform=ax.transAxes, fontsize=11.5, fontweight="bold",
            color=th["ink"], va="bottom")
    ax.text(0, 1.18, subtitle, transform=ax.transAxes, fontsize=8.5, color=th["ink2"], va="top",
            linespacing=1.35)


def heatmap(ax, M, th, cmap, vmax, title, ylabels=True):
    lo, hi = SETS[0][1], SETS[-1][2]
    im = ax.imshow(M[lo:hi, lo:hi], cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    for spine in ax.spines.values():
        spine.set_color(th["base"])
    ax.set_title(title, fontsize=9, color=th["ink"], pad=4)
    pos, names = [(a + b) / 2 for _, a, b in SETS], [n for n, _, _ in SETS]
    ax.set_xticks(pos, names, fontsize=7, color=th["ink2"])
    if ylabels:
        ax.set_yticks(pos, names, fontsize=7, color=th["ink2"])
    else:
        ax.set_yticks([])
    ax.tick_params(length=0)
    return im


def colorbar(fig, im, cax, th, label):
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, colors=th["muted"], labelcolor=th["ink2"])
    cb.outline.set_edgecolor(th["base"])
    cb.set_label(label, fontsize=8, color=th["ink2"])


# --- panels ---------------------------------------------------------------------------------
def panel_contexts(ax, th):
    """1: the data matrix split into contexts, each giving its own gene-gene correlation."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(1.9, 9.6, "data: rows = conditions,\ncolumns = genes", ha="center", va="center", fontsize=7.5,
            color=th["muted"], linespacing=1.3)
    ax.text(5.6, 9.6, "gene × gene\nper context", ha="center", va="center", fontsize=7.5,
            color=th["muted"], linespacing=1.3)
    rows = [("sys0", th["other"]), ("sys1", th["other"]), None, ("sys6", th["ctx_a"]),
            ("sys7", th["ctx_b"]), None]
    y = 8.6
    for row in rows:
        if row is None:
            ax.text(1.9, y - 0.35, "⋮", ha="center", va="center", color=th["muted"], fontsize=12)
            y -= 0.8
            continue
        name, col = row
        ax.add_patch(Rectangle((0.8, y - 1.0), 2.2, 1.0, facecolor=col, alpha=0.35, edgecolor=th["base"],
                               lw=0.8))
        ax.text(0.65, y - 0.5, name, ha="right", va="center", fontsize=8, color=th["ink2"])
        ax.add_artist(FancyArrowPatch((3.2, y - 0.5), (5.0, y - 0.5), arrowstyle="-|>", mutation_scale=9,
                                      color=th["muted"], lw=0.9))
        ax.add_patch(Rectangle((5.15, y - 0.95), 0.9, 0.9, facecolor=th["surface"], edgecolor=col, lw=1.6))
        ax.plot([5.15, 6.05], [y - 0.05, y - 0.95], color=col, lw=1.2)
        y -= 1.35
    ax.text(6.6, 6.0, "t = arctanh(r)·√(n − 3)\nfor every gene pair and\ncontext: comparable\n"
            "across context sizes", fontsize=8, color=th["ink2"], va="center", linespacing=1.4)
    ax.text(0.0, 0.25, "Pre-step: LFC profiles much larger and broader than the\nrest of their context "
            "(dominant conditions) are removed first.", fontsize=7.5, color=th["muted"], linespacing=1.35)


def panel_pair(ax, th, d):
    """2: one gene pair across contexts; its top-n contexts form its 'union' support."""
    t = d["t_pair"]
    k = np.arange(len(t))
    colors = [th["other"]] * len(t)
    colors[d["contexts"].index("sys6")] = th["ctx_a"]
    colors[d["contexts"].index("sys7")] = th["ctx_b"]
    ax.bar(k, t, width=0.7, color=colors, edgecolor=th["surface"], linewidth=1)
    for c in d["top"]:  # mark the top-n contexts
        ax.text(c, max(t[c], 0) + 0.2, "▼", ha="center", fontsize=7, color=th["ink2"])
    ax.axhline(0, color=th["base"], lw=0.8)
    mid = (d["contexts"].index("sys6") + d["contexts"].index("sys7")) / 2  # one label for both bars
    ax.set_xticks([0, mid, len(t) - 1], [d["contexts"][0], "sys6 · sys7", d["contexts"][-1]], fontsize=7.5)
    ax.set_xlabel(f"{len(t)} systems (contexts)", fontsize=8, color=th["ink2"])
    ax.set_ylabel("t in each context", fontsize=8.5, color=th["ink2"])
    ax.set_ylim(min(t.min(), 0) - 0.5, t.max() + 3.2)
    style_axes(ax, th)
    ax.text(0.02, 0.97, "▼ top-3 contexts of this pair (two Q genes)\n"
                        f"pooled correlation over all contexts:  r = {d['pooled_pair']:.2f}\n"
                        f"union of its top-3, calibrated:  r = {d['union_pair']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=th["ink"], linespacing=1.5)


def panel_tom_rule(ax, th):
    """4: when does a shared neighbour u count for the pair (i, j)?"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def triangle(x0, c_iu, c_uj, c_ij, verdict, note):
        pts = {"i": (x0, 3.6), "j": (x0 + 3.0, 3.6), "u": (x0 + 1.5, 6.6)}
        for (a, b), col in [(("i", "u"), c_iu), (("u", "j"), c_uj), (("i", "j"), c_ij)]:
            (xa, ya), (xb, yb) = pts[a], pts[b]
            ax.plot([xa, xb], [ya, yb], color=col, lw=3 if col != th["other"] else 1.5,
                    solid_capstyle="round", zorder=1)
        for name, (x, y) in pts.items():
            ax.scatter([x], [y], s=330, color=th["surface"], edgecolor=th["ink2"], lw=1.2, zorder=2)
            ax.text(x, y, name, ha="center", va="center", fontsize=9, color=th["ink"], zorder=3)
        ax.text(x0 + 1.5, 2.3, verdict, ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=th["ink"], linespacing=1.35)
        ax.text(x0 + 1.5, 1.5, note, ha="center", va="top", fontsize=7.5, color=th["ink2"], linespacing=1.35)

    triangle(0.7, th["ctx_a"], th["ctx_b"], th["other"], "standard TOM: counts\nmatched TOM: ignored",
             "i–u and u–j hold in\ndifferent contexts")
    triangle(5.9, th["ctx_a"], th["ctx_a"], th["ctx_a"], "both: counts",
             "i–u, u–j and i–j share\na supporting context")
    for x, col, lab, lw in [(0.7, th["ctx_a"], "supported by sys6", 3), (4.4, th["ctx_b"], "by sys7", 3),
                            (6.9, th["other"], "weak", 1.5)]:
        ax.plot([x, x + 0.6], [9.0, 9.0], color=col, lw=lw, solid_capstyle="round")
        ax.text(x + 0.8, 9.0, lab, va="center", fontsize=7.5, color=th["ink2"])


def panel_modules(ax, th, d):
    """5: one hard module (P+Q+R), split into overlapping context-defined submodules."""
    genes = d["module_genes"]
    gpos = {g: k for k, g in enumerate(genes)}
    pos_of = {name: k for k, name in enumerate(d["genes"])}
    rows = [(f"hard module {d['module']}", genes, th["strong"], "")]
    sub_cols = {"sys6": th["ctx_a"], "sys7": th["ctx_b"]}
    for sub, row in d["subs"].iterrows():
        members = [pos_of[g] for g in d["mem"].loc[d["mem"].submodule == sub, "gene"]]
        rows.append((f"submodule {sub}", members, sub_cols.get(row["contexts"], th["muted"]),
                     f"context {row['contexts']}"))
    n = len(genes)
    for r, (label, members, col, ctx) in enumerate(rows):
        y = (len(rows) - 1 - r) * 1.5
        x = np.array(sorted(gpos[g] for g in members if g in gpos))
        ax.bar(x, 0.55, width=1.0, bottom=y, color=col, linewidth=0)
        ax.text(0, y + 0.7, label, ha="left", va="bottom", fontsize=8.5, color=th["ink"])
        ax.text(n, y + 0.7, ctx, ha="right", va="bottom", fontsize=8, color=th["ink2"])
    # planted P / Q / R ranges under the strips
    for name, a, b in SETS[3:6]:
        xs = [gpos[g] for g in genes if a <= g < b]
        if xs:
            ax.plot([min(xs) + 1, max(xs) - 1], [-0.45, -0.45], color=th["base"], lw=2, solid_capstyle="butt")
            ax.text((min(xs) + max(xs)) / 2, -0.75, f"planted {name}", ha="center", va="top", fontsize=7.5,
                    color=th["muted"])
    ax.text(n / 2, -1.6, "genes of the hard module (Q genes are in both submodules)", ha="center", va="top",
            fontsize=7.5, color=th["ink2"])
    ax.set_xlim(-1, n + 1)
    ax.set_ylim(-2.2, (len(rows) - 1) * 1.5 + 1.4)
    ax.axis("off")


def panel_recovery(ax, th, d):
    """6: share of each planted set recovered as one module, pooled WGCNA vs ctxwgcna."""
    names = [n for n, _, _ in SETS[:-1]]
    x = np.arange(len(names))
    w = 0.36
    for off, key, col, lab in [(-w / 2 - 0.02, "rec_pooled", th["other"], "pooled WGCNA"),
                               (w / 2 + 0.02, "rec_ctx", th["strong"], "ctxwgcna")]:
        vals = [d[key][n] for n in names]
        ax.bar(x + off, vals, width=w, color=col, label=lab)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_ylabel("genes recovered in one module", fontsize=8.5, color=th["ink2"])
    style_axes(ax, th)
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.13), ncol=2, frameon=False, fontsize=8,
                    handlelength=1.2, columnspacing=1.2)
    for text in leg.get_texts():
        text.set_color(th["ink2"])
    ax.text(4, 1.0, "P, Q, R: one hard module,\nseparated by the split (5)", ha="center", va="bottom",
            fontsize=7, color=th["muted"], linespacing=1.3)


# --- figure ---------------------------------------------------------------------------------
def draw(d, theme):
    th = THEMES[theme]
    cmap = LinearSegmentedColormap.from_list("seq", th["seq"])
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]})
    fig = plt.figure(figsize=(15, 10), facecolor=th["surface"])
    gs = fig.add_gridspec(2, 3, left=0.05, right=0.965, top=0.80, bottom=0.08, hspace=0.55, wspace=0.28)

    fig.text(0.04, 0.965, "ctxwgcna: WGCNA on the union of context-specific co-expression",
             fontsize=15, fontweight="bold", color=th["ink"], va="top")
    intro = ("Standard WGCNA pools all observations, so co-expression present in only a few systems or "
             "conditions is diluted. ctxwgcna keeps, for every gene pair, the contexts that support it, "
             "and uses them to build the network (1–3) and to restrict the topological overlap (4). "
             "Modules are then split by context into overlapping submodules (5).")
    fig.text(0.04, 0.925, textwrap.fill(intro, 170), fontsize=9.5, color=th["ink2"], va="top",
             linespacing=1.4)

    a1 = fig.add_subplot(gs[0, 0])
    panel_title(a1, th, "1", "Correlate within each context", "each system (or condition) separately")
    panel_contexts(a1, th)

    a2 = fig.add_subplot(gs[0, 1])
    panel_title(a2, th, "2", "Keep each pair's best contexts",
                "union, not consensus: the top-n contexts per pair are\n"
                "summarised and corrected for picking the best of K")
    panel_pair(a2, th, d)

    a3 = fig.add_subplot(gs[0, 2])
    a3.axis("off")
    panel_title(a3, th, "3", "Local modules become visible",
                "similarity of all gene pairs, ordered by planted set")
    h1, h2 = a3.inset_axes([0.0, 0.0, 0.43, 0.88]), a3.inset_axes([0.47, 0.0, 0.43, 0.88])
    heatmap(h1, d["pooled_r"], th, cmap, 0.5, "pooled (WGCNA)")
    im = heatmap(h2, d["union_r"], th, cmap, 0.5, "context union", ylabels=False)
    colorbar(fig, im, a3.inset_axes([0.93, 0.08, 0.03, 0.72]), th, "r")

    a4 = fig.add_subplot(gs[1, 0])
    panel_title(a4, th, "4", "Restrict TOM to matching contexts",
                "a shared neighbour u counts for i–j only through\na context that also supports i–j")
    panel_tom_rule(a4, th)

    a5 = fig.add_subplot(gs[1, 1])
    panel_title(a5, th, "5", "Hard modules, then overlapping submodules",
                "each module is split by the contexts that activate\nits genes; a gene may join several")
    panel_modules(a5, th, d)

    a6 = fig.add_subplot(gs[1, 2])
    panel_title(a6, th, "6", "Result on the simulation",
                "WGCNA on pooled data finds only the global set G1")
    panel_recovery(a6, th, d)

    fig.text(0.04, 0.02, f"Simulated example: 360 genes, {N_SYSTEMS} systems × 100 conditions; planted G1 "
             "(all systems), L1 (sys0–1), L2 (sys2–3), P+Q (sys6), Q+R (sys7); power "
             f"{POWER} for both methods. Generated by scripts/make_schematic.py.",
             fontsize=7.5, color=th["muted"])
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    d = run_example()
    for theme, name in (("light", "schematic.png"), ("dark", "schematic-dark.png")):
        fig = draw(d, theme)
        fig.savefig(os.path.join(OUT, name), dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print("wrote", os.path.normpath(os.path.join(OUT, name)))


if __name__ == "__main__":
    main()
