"""
Step 14 - Make the manuscript figures (PNG, 300 dpi, 178 mm wide) from the result files.

Reads results/, results_hybrid/, results_prox1/ and the benchmark scores; writes figures/Fig1..Fig6.png
Manuscript order: 1 design, 2 benchmark, 3 atlas, 4 network (negative), 5 PROX1 mechanism, 6 two signals
Run:  python 14_make_figures.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = "figures"
W = 178 / 25.4                                     # double-column width in inches
UP, DOWN, NEU = "#eb6834", "#2a78d6", "#8a8984"   # orange = UP in diabetic, blue = DOWN
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ACC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
HEP = ["HepG2", "liver", "hepatocyte"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5,
                     "axes.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 300})


def tag(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- Figure 1: study design
def fig1():
    fig, ax = plt.subplots(figsize=(W, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f4f6fa", ec="#2a78d6"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=7.2, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.2, body, ha="center", va="top", fontsize=5.3, color=MUTED, linespacing=1.4)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, shrinkA=0, shrinkB=0))

    box(0.5, 26, 22, 18, "1  T2D variants",
        "two SuSiE fine-mapped GWAS\n(FinnGen R12; GCST90475667)\nPIP ≥ 0.2, non-coding\n923 variants")
    box(26, 26, 22, 18, "2  AlphaGenome",
        "1 Mb window,\nrisk vs reference allele\nHepG2 · liver · hepatocyte\nRNA · DNase · ATAC · ChIP")
    box(51.5, 26, 22, 18, "3  Liver benchmark",
        "328 causal GTEx liver eQTLs\ndirection 82.5% correct\nwhen AlphaGenome confident\ngene: nearest 62% vs AG 42%",
        ec="#1baf7a", fc="#f1faf6")
    box(77, 26, 22.5, 18, "4  Hybrid atlas",
        "nearest protein-coding gene\n+ confident AG direction\n108 variants → 91 genes\n45 up · 46 down · 14 TFs")
    arrow(23.3, 35, 25.2, 35); arrow(48.8, 35, 50.7, 35); arrow(74.3, 35, 76.2, 35)
    box(51.5, 2, 22, 18, "5  Network tree (L2+)",
        "CollecTRI + OmniPath\nLevels 2–4, 3 yardsticks\n(patient liver, Open Targets,\nFinnGen) → not validated",
        ec=NEU, fc="#f6f6f4")
    box(77, 2, 22.5, 18, "6  PROX1 case study",
        "rs17712208 independent signal\nHNF1A-bound liver enhancer\n(ENCODE), in-silico mutagenesis\ncolocalises with alanine,\nglutamine",
        ec=UP, fc="#fdf3ee")
    arrow(82, 25.2, 70, 20.8); arrow(88.25, 25.2, 88.25, 20.8)
    ax.text(1, 11, "Which liver genes do T2D risk\nvariants change, in which direction,\nand can a sequence model add\n"
                   "mechanism that nearest-gene\nmapping cannot?", fontsize=6.6, color=INK, va="center", style="italic")
    save(fig, "Fig1_study_design")


# ---------------------------------------------------------------- Figure 2: benchmark
def fig2():
    d = pd.read_csv("results/benchmark_direction.csv")
    g = pd.read_csv("results/benchmark_gene_choice.csv")
    hy = pd.read_csv("results/benchmark_rules.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    # A direction
    ax = axs[0]
    lab = {"all benchmark eQTLs": "All eQTLs", "eQTL PIP >= 0.9": "eQTL PIP ≥ 0.9",
           "AlphaGenome confident (|quantile| >= 0.99)": "AG confident\n(|q| ≥ 0.99)",
           "largest 25% AlphaGenome effects": "Largest 25%\nAG effects",
           "eGene well expressed in liver": "eGene expressed",
           "stronger eQTLs (|beta| above median)": "Stronger eQTLs",
           "CONTROL: random other gene (should be ~50%)": "Control:\nrandom gene"}
    d = d[d["subset"].isin(lab)]
    y = np.arange(len(d))[::-1]
    lo = d["95%_CI"].str.split("-").str[0].astype(float); hi = d["95%_CI"].str.split("-").str[1].astype(float)
    col = [NEU if "CONTROL" in s else ACC[0] for s in d["subset"]]
    ax.barh(y, d["AlphaGenome_correct"], color=col, height=0.62)
    ax.errorbar(d["AlphaGenome_correct"], y, xerr=[d["AlphaGenome_correct"] - lo, hi - d["AlphaGenome_correct"]],
                fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.6)
    for yi, v, n in zip(y, d["AlphaGenome_correct"], d["variants"]):
        ax.text(0.02, yi, f"n={n}", va="center", fontsize=5.8, color="white")
    for yi, v, h in zip(y, d["AlphaGenome_correct"], hi):
        ax.text(h + 0.02, yi, f"{v:.0%}", va="center", fontsize=6.3, color=INK)
    ax.axvline(0.5, color=MUTED, lw=0.7, ls="--"); ax.text(0.5, len(d) - 0.35, "chance", fontsize=6, color=MUTED, ha="center")
    ax.set_yticks(y); ax.set_yticklabels([lab[s] for s in d["subset"]], fontsize=6.4)
    ax.set_xlim(0, 1.12); ax.set_xlabel("Direction correct (sign of eQTL β)")
    ax.set_title("Direction of effect", loc="left"); tag(ax, "A", -0.42)
    # B gene choice
    ax = axs[1]
    vals = [g.loc[0, "nearest_gene_correct"], g.loc[0, "AlphaGenome_correct"], g.loc[1, "AlphaGenome_correct"]]
    names = ["Nearest\ngene", "AG top\ngene", "AG top\nexpressed"]
    ax.bar(range(3), vals, color=[ACC[2], ACC[0], ACC[0]], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=6.4)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=6.4); ax.set_ylim(0, 0.8)
    ax.set_ylabel("True eGene named (n = 328)")
    ax.text(0.5, 0.74, f"McNemar p = {g.loc[0, 'p_McNemar']:.1e}", ha="center", fontsize=6, color=MUTED)
    ax.set_title("Target-gene choice", loc="left"); tag(ax, "B", -0.35)
    # C rules: gene AND direction correct, among variants each rule keeps
    ax = axs[2]
    ax.bar(range(len(hy)), hy["correct_among_kept"], color=[UP, ACC[0], ACC[0]], width=0.62)
    for i, r in hy.iterrows():
        ax.text(i, r["correct_among_kept"] + 0.02, f"{r['correct_among_kept']:.0%}\n(n={r['kept']})",
                ha="center", fontsize=6)
    ax.set_xticks(range(len(hy))); ax.set_xticklabels(["Hybrid", "AG top\ngene", "AG top\nexpressed"], fontsize=6.4); ax.set_ylim(0, 0.85)
    ax.set_ylabel("Gene and direction both correct")
    ax.set_title("Assignment rule", loc="left"); tag(ax, "C", -0.35)
    fig.tight_layout(w_pad=1.2)
    save(fig, "Fig2_benchmark")


# ---------------------------------------------------------------- Figure 3: atlas
def fig3():
    g = pd.read_csv("results_hybrid/level0_genes.csv")
    pr = pd.read_csv("results_hybrid/level1_prioritisation.csv")
    allg = pd.read_csv("results_hybrid/level1_all_nearest_genes.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.9), gridspec_kw={"width_ratios": [1.0, 1.25]})
    # A funnel
    ax = axs[0]
    steps = [("Fine-mapped non-coding\nT2D variants (PIP ≥ 0.2)", 923),
             ("Nearest protein-coding gene\nscored by AlphaGenome", allg["variant_id"].nunique()),
             ("Confident direction\n(|quantile| ≥ 0.99)", int(allg["significant"].sum())),
             ("…and HepG2/liver/hepatocyte\nagree (high confidence)", int((allg["confidence"] == "high").sum())),
             ("Level-1 liver genes", len(g))]
    for i, (s, n) in enumerate(steps):
        ax.barh(len(steps) - 1 - i, n, color=ACC[0] if i < 4 else UP, height=0.6)
        ax.text(n + 15, len(steps) - 1 - i, f"{n}", va="center", fontsize=6.6)
    ax.set_yticks(range(len(steps))[::-1]); ax.set_yticklabels([s for s, _ in steps], fontsize=6.3)
    ax.set_xlim(0, 1080); ax.set_xlabel("Count")
    ax.text(0.98, 0.04, f"UP {int((g.weighted_score > 0).sum())} · DOWN {int((g.weighted_score < 0).sum())}"
                        f"\nTFs {int(g.is_TF.sum())}", transform=ax.transAxes, ha="right", fontsize=6.4, color=MUTED)
    ax.set_title("From variants to Level-1 genes", loc="left"); tag(ax, "A", -0.62)
    # B top 25 genes by priority score
    ax = axs[1]
    top = pr.sort_values("priority_score", ascending=False).head(25).iloc[::-1]
    yy = np.arange(len(top))
    cols = [UP if s > 0 else DOWN for s in top["hepatic_score"]]
    ax.hlines(yy, 0, top["priority_score"], color=GRID, lw=1.6)
    ax.scatter(top["priority_score"], yy, s=26, c=cols, zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{gn}{' (TF)' if tf else ''}  {rs}" for gn, tf, rs in
                        zip(top["gene_name"], top["is_TF"], top["rsid"])], fontsize=5.9)
    for t in ax.get_yticklabels():
        if t.get_text().startswith("PROX1"):
            t.set_fontweight("bold")
    ax.set_xlabel("Priority score (7 equal-weight criteria, max 7)")
    ax.scatter([], [], c=UP, s=20, label="UP in diabetic"); ax.scatter([], [], c=DOWN, s=20, label="DOWN in diabetic")
    ax.legend(loc="center right", fontsize=6.2)
    ax.set_title("Top 25 of 108 variant–gene pairs", loc="left"); tag(ax, "B", -0.55)
    fig.tight_layout(w_pad=2)
    save(fig, "Fig3_atlas")


# ---------------------------------------------------------------- Figure 5 (manuscript): PROX1 mechanism
def tissue_class(name):
    n = name.lower()
    if n in ("hepg2", "hepatocyte", "huh-7", "huh-7.5") or ("liver" in n and "stellate" not in n):
        return "Liver / hepatocyte"
    if any(k in n for k in ("islet", "endocrine pancreas", "type b pancreatic")):
        return "Islet / endocrine"
    if "pancrea" in n:
        return "Whole pancreas"
    if any(k in n for k in ("intestin", "colon", "caco", "ht-29", "lovo", "sw480", "peyer", "stomach", "duoden",
                            "rectum", "gallbladder")):
        return "Gut epithelium"
    if any(k in n for k in ("kidney", "renal", "tubul", "rcc", "caki", "achn", "hk-2", "nephron", "glomerul")):
        return "Kidney"
    return "Other (n≈250)"


def fig4():
    loc = pd.read_csv("data/t2d_variants_for_alphagenome.csv")
    loc = loc[(loc.chrom == "chr1") & loc.pos.between(213_800_000, 214_300_000)].drop_duplicates("variant_id")
    ts = pd.read_csv("results_prox1/test2_tissue_specificity.csv")
    bio = pd.read_csv("results_prox1/test2_rs17712208_dnase_by_biosample.csv")
    ism = pd.read_csv("results_prox1/test4_ism_by_position.csv")
    enc = pd.read_csv("results_prox1/encode_selected_tf_peaks.csv")
    fig = plt.figure(figsize=(W, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[1, 1.15], hspace=0.62, wspace=0.55)

    # A locus: T2D effect vs AlphaGenome DNase (liver)
    ax = fig.add_subplot(gs[0, 0])
    dn = ts[ts.output == "DNASE"].set_index("rsid")["liver_mean"]
    loc = loc.assign(dn=loc.rsid.map(dn))
    for r in loc.itertuples():
        c = DOWN if r.dn < -0.5 else NEU
        ax.scatter(r.beta_alt * r.risk_sign, r.dn, s=18 + 60 * r.pip, color=c, edgecolor="white", lw=0.5, zorder=3)
    lab = {"rs17712208": (0.062, -1.35), "rs340874": (0.03, 0.33), "rs79687284": (0.004, -0.42),
           "rs340882": (0.048, -0.3), "rs12132228": (0.0005, 0.3)}
    for r in loc.itertuples():
        x, yv = r.beta_alt * r.risk_sign, r.dn
        ax.annotate(r.rsid, xy=(x, yv), xytext=lab[r.rsid], fontsize=5.8,
                    fontweight="bold" if r.rsid == "rs17712208" else "normal",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xlabel("T2D effect of risk allele (log OR)"); ax.set_ylabel("AlphaGenome DNase change\nliver (risk − ref)")
    ax.set_xlim(0, 0.115); ax.set_ylim(-1.9, 0.45)
    ax.set_title("Five fine-mapped variants at PROX1", loc="left"); tag(ax, "A", -0.3)

    # B tissue classes
    ax = fig.add_subplot(gs[0, 1])
    bio["cls"] = bio["biosample_name"].map(tissue_class)
    order = ["Gut epithelium", "Kidney", "Liver / hepatocyte", "Whole pancreas", "Islet / endocrine", "Other (n≈250)"]
    rng = np.random.default_rng(1)
    for i, c in enumerate(order):
        v = bio.loc[bio.cls == c, "diabetic"].values
        ax.scatter(rng.uniform(-0.18, 0.18, len(v)) + i, v, s=7, color=DOWN if c != "Other (n≈250)" else NEU,
                   alpha=0.85 if c != "Other (n≈250)" else 0.35, edgecolor="none")
        ax.hlines(np.median(v), i - 0.28, i + 0.28, color=INK, lw=1)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Gut", "Kidney", "Liver", "Pancreas\n(whole)", "Islet/\nendocrine", "Other"], fontsize=6)
    ax.set_ylabel("DNase change at rs17712208")
    ax.set_title("Enhancer closing across 305 biosamples", loc="left"); tag(ax, "B", -0.2)

    # C ISM
    ax = fig.add_subplot(gs[1, 0])
    seq = "GGATCGTTAATGGAGCTATGGTTAATTATTGACTGATTAGGG"
    pos0 = 213977102
    ism = ism.sort_values("pos")
    off = ism["pos"] - pos0
    z = (ism.iloc[:, 1] - ism.iloc[:, 1].median()) / ism.iloc[:, 1].std()
    ax.bar(off, ism.iloc[:, 1], color=[DOWN if zz <= -1.5 else NEU for zz in z], width=0.8)
    ax.axhline(0, color=GRID, lw=0.8)
    ref = "tggctgactggatcgttaatggagctatggTtaattattgactgattagggatttacctta".upper()
    for o in off:
        ch = ref[30 + int(o)]
        ax.text(o, ax.get_ylim()[0] if False else -0.0, "", fontsize=1)
    ymin = ism.iloc[:, 1].min()
    for o in off:
        ch = ref[30 + int(o)]
        ax.text(o, ymin * 1.18, ch, ha="center", va="center", fontsize=5.4, family="DejaVu Sans Mono",
                color=INK if (-1 <= o <= 11) else MUTED, fontweight="bold" if o == 0 else "normal")
    ax.add_patch(Rectangle((-1.5, ymin * 1.32), 13, ymin * -0.28, fill=False, ec=UP, lw=0.8))
    ax.text(5, ymin * 1.48, "HNF1 site  GTTAAT·n·ATTaaC", ha="center", fontsize=6, color=UP)
    ax.annotate("rs17712208\nT>A", xy=(0, ism.iloc[:, 1][off == 0].values[0]), xytext=(-14, ymin * 0.75),
                fontsize=6, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_ylim(ymin * 1.6, ism.iloc[:, 1].max() * 1.3)
    ax.set_xlabel("Position relative to rs17712208 (bp)")
    ax.set_ylabel("Mean DNase change\nif base is mutated (liver)")
    ax.set_title("In-silico mutagenesis (41 bp)", loc="left"); tag(ax, "C", -0.3)

    # D ENCODE ChIP-seq
    ax = fig.add_subplot(gs[1, 1])
    b = enc[(enc.site == "rs17712208") & enc.bound].copy()
    b["lab"] = b["tf"] + " (" + b["cell"] + ")"
    b = b.sort_values("summit_to_variant_bp", key=abs, ascending=False).reset_index(drop=True)
    for i, r in b.iterrows():
        ax.hlines(i, r.peak_start - pos0 + 1, r.peak_end - pos0 + 1, color=GRID, lw=4)
        ax.scatter(r.summit_to_variant_bp, i, s=10 + r.signal / 2, color=UP if r.tf == "HNF1A" else DOWN,
                   zorder=3, edgecolor="white", lw=0.5)
    ax.axvline(0, color=INK, lw=0.7, ls="--")
    ax.set_yticks(range(len(b))); ax.set_yticklabels(b["lab"], fontsize=6.2)
    ax.set_xlabel("Peak (bar) and summit (dot) relative to variant (bp)")
    ax.set_xlim(-300, 300)
    ax.set_title("ENCODE ChIP-seq peaks over the variant", loc="left"); tag(ax, "D", -0.42)
    save(fig, "Fig5_PROX1_mechanism")


# ---------------------------------------------------------------- Figure 6 (manuscript): two signals, colocalisation
def fig5():
    o = pd.read_csv("results_prox1/coloc_open_targets.csv")
    traits = ["alanine", "glutamine", "BCAA", "albumin", "GGT", "LDL", "HbA1c", "glucose"]
    names = {"BCAA": "BCAA (valine)", "GGT": "GGT", "LDL": "LDL-C", "HbA1c": "HbA1c"}
    m = o.groupby(["rsid", "group"])["clpp"].max().unstack().reindex(columns=traits).fillna(0)
    h4 = o.groupby(["rsid", "group"])["h4"].max().unstack().reindex(columns=traits)
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.45, 1.15]})
    ax = axs[0]
    x = np.arange(len(traits)); w = 0.38
    for k, (rs, c) in enumerate([("rs17712208", UP), ("rs340874", DOWN)]):
        v = m.loc[rs] if rs in m.index else pd.Series(0, index=traits)
        ax.bar(x + (k - 0.5) * w, v.values, w * 0.92, color=c, label=f"{rs} T2D signal")
    ax.axhline(0.1, color=MUTED, lw=0.6, ls="--"); ax.text(-0.45, 0.12, "CLPP 0.1", fontsize=5.8,
                                                            color=MUTED, ha="left")
    ax.set_xticks(x); ax.set_xticklabels([names.get(t, t.capitalize()) for t in traits], rotation=30, ha="right")
    ax.set_ylabel("Colocalisation (eCAVIAR CLPP)"); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", fontsize=6.2, ncol=2, bbox_to_anchor=(0.5, 1.13))
    tag(ax, "A", -0.12, 1.1)
    ax = axs[1]
    rows = [("Signal", "rs17712208", "rs340874"),
            ("Risk-allele freq. (EUR)", "≈5%", "≈53%"),
            ("T2D log OR", "0.087", "0.023"),
            ("LD (r²)", "< 0.05", ""),
            ("cCRE type", "distal enhancer", "promoter-like"),
            ("HNF1A ChIP peak", "yes (2/2)", "no"),
            ("AlphaGenome DNase", "−1.6 (closes)", "+0.02"),
            ("Colocalised traits", "alanine,\nglutamine,\nGGT, glucose,\nHbA1c, valine,\nalbumin, LDL-C", "glucose,\nHbA1c")]
    ax.axis("off")
    for i, r in enumerate(rows):
        yy = 1 - i * 0.12
        for j, t in enumerate(r):
            ax.text([0, 0.5, 0.8][j], yy, t, fontsize=6.0 if i else 6.4, fontweight="bold" if i == 0 else "normal",
                    color=INK if j else MUTED, transform=ax.transAxes, va="top")
    tag(ax, "B", -0.02, 1.1)
    fig.tight_layout(w_pad=1.5)
    save(fig, "Fig6_two_signals")


# ---------------------------------------------------------------- Figure 4 (manuscript): network propagation (negative)
def fig6():
    cc = pd.read_csv("results_hybrid/control_comparison.csv")
    gd = pd.read_csv("results_hybrid/genetic_direction_test.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.3))
    ax = axs[0]
    sub = cc[cc.test == "DIRECTION all DE genes"]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["A_AlphaGenome"], 0.38, color=ACC[0], label="Tree")
    ax.bar(x + 0.2, sub["C_shuffled_mean"], 0.38, color=NEU, label="Shuffled signs")
    for xi, p in zip(x, sub["p_A_vs_C"]):
        ax.text(xi, 0.66, f"p = {p:.2f}", ha="center", fontsize=6)
    ax.axhline(0.5, color=MUTED, lw=0.6, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(sub["yardstick"])
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("Direction agreement with\ndiabetic-liver expression")
    ax.legend(fontsize=6, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    ax.set_title("Patient liver biopsies", loc="left"); tag(ax, "A", -0.38)
    ax = axs[1]
    sub = cc[cc.test.str.contains("top-100")]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["A_AlphaGenome"], 0.38, color=ACC[0], label="Tree")
    ax.bar(x + 0.2, sub["random_mean"], 0.38, color=NEU, label="Matched random")
    for xi, p, v in zip(x, sub["p_A_vs_random"], sub["A_AlphaGenome"]):
        ax.text(xi, v + 0.5, f"p = {p:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(["Mendelian\ndiabetes genes", "Approved-drug\ntargets"])
    ax.set_ylabel("Gold genes in top-100"); ax.set_ylim(0, 19)
    ax.legend(fontsize=6, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    ax.set_title("Open Targets gene sets", loc="left"); tag(ax, "B", -0.3)
    ax = axs[2]
    r = gd.iloc[0]
    lo, hi = [float(v) for v in str(r["shuffled_95%"]).split("-")]
    ax.errorbar([1], [r["shuffled_mean"]], yerr=[[r["shuffled_mean"] - lo], [hi - r["shuffled_mean"]]], fmt="o",
                color=NEU, ms=4, capsize=2, lw=0.8, label="Shuffled (95% range)")
    ax.scatter([0], [r["tree_concordance"]], color=ACC[0], s=30, zorder=3, label="Tree")
    ax.text(0, r["tree_concordance"] + 0.04, f"{r['tree_concordance']:.0%}\np = {r['p_tree_vs_shuffled']:.2f}",
            ha="center", fontsize=6)
    ax.axhline(0.5, color=MUTED, lw=0.6, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Tree", "Null"]); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0.2, 0.85)
    ax.set_ylabel(f"Sign agreement with FinnGen\nWald ratio ({int(r['genes'])} genes)")
    ax.set_title("Genetic direction (SMR)", loc="left"); tag(ax, "C", -0.38)
    fig.tight_layout(w_pad=1.2)
    save(fig, "Fig4_network_negative")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
