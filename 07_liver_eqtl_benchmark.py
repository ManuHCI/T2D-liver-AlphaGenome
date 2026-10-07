"""
Step 7 - METHOD BENCHMARK: does AlphaGenome find the right liver gene and direction?

Uses ~1,000 GTEx liver eQTL variants that fine-mapping says are almost certainly causal
(SuSiE PIP >= 0.5), each with a known target gene (eGene) and a known effect (beta for the
ALT allele). These are the same kind of input as our T2D variants, but with a known answer.

Questions
  1. DIRECTION   : does the sign of AlphaGenome's predicted change for the eGene match the
                   sign of the real eQTL beta?                       (chance = 50 %)
  2. GENE CHOICE : which method names the real eGene more often?
                   AlphaGenome top gene   vs   nearest gene          (paired McNemar test)

Three commands (run on your machine):
  python 07_liver_eqtl_benchmark.py --prepare                 # picks the variants
  python 02_score_alphagenome_hepatic.py --input data/liver_eqtl_benchmark_variants.csv ^
         --chunks data/scores_eqtl_benchmark --expression-only   # AlphaGenome, ~15 min
  python 07_liver_eqtl_benchmark.py --analyse                 # results
Needs data/eqtl/QTD000266.credible_sets.tsv.gz (downloaded by step 6) and
data/gencode.basic.gtf.gz (downloaded by step 5).
"""
import argparse
import glob
import gzip
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
CS = "data/eqtl/QTD000266.credible_sets.tsv.gz"
GTF = "data/gencode.basic.gtf.gz"
VAR = "data/liver_eqtl_benchmark_variants.csv"
CHUNKS = "data/scores_eqtl_benchmark"
RES = "results"
HEP = ["HepG2", "liver", "hepatocyte"]


def gencode_genes():
    rows = []
    with gzip.open(GTF, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.split("\t", 8)
            if c[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^".]+)', c[8]).group(1)
            gtype = re.search(r'gene_type "([^"]+)"', c[8]).group(1)
            name = re.search(r'gene_name "([^"]+)"', c[8])
            rows.append((c[0], int(c[3]) if c[6] == "+" else int(c[4]), gid, gtype, name.group(1) if name else gid))
    return pd.DataFrame(rows, columns=["chrom", "tss", "gene_id", "gene_type", "gene_name"])


# ============================================================ prepare
def prepare(a):
    if not os.path.exists(CS):
        sys.exit(f"{CS} missing - run 06_liver_eqtl_comparison.py once first (it downloads it)")
    cs = pd.read_csv(CS, sep="\t")
    cs["gene_id"] = cs["gene_id"].astype(str).str.split(".").str[0]
    genes = gencode_genes()
    pc = set(genes.loc[genes["gene_type"] == "protein_coding", "gene_id"])
    v = cs["variant"].str.split("_", expand=True)
    cs["chrom"], cs["pos"], cs["ref"], cs["alt"] = v[0], v[1].astype(int), v[2], v[3]
    cs = cs[(cs["ref"].str.len() == 1) & (cs["alt"].str.len() == 1)]          # SNVs only
    cs = cs[cs["gene_id"].isin(pc) & (cs["pip"] >= a.min_pip)]
    # one variant per gene (highest PIP), and each variant used for only one gene
    cs = cs.sort_values("pip", ascending=False).drop_duplicates("gene_id").drop_duplicates("variant")
    if len(cs) > a.n:
        cs = cs.sample(a.n, random_state=1)
    cs["chrom"] = np.where(cs["chrom"].str.startswith("chr"), cs["chrom"], "chr" + cs["chrom"])
    cs["variant_id"] = cs["chrom"].str.replace("chr", "", regex=False) + "_" + cs["pos"].astype(str) + "_" \
        + cs["ref"] + "_" + cs["alt"]
    out = cs[["variant_id", "chrom", "pos", "ref", "alt", "gene_id", "beta", "pvalue", "pip"]] \
        .rename(columns={"gene_id": "eGene"})
    out.to_csv(VAR, index=False)
    print(f"Benchmark set: {len(out)} liver eQTL variants (SNV, protein-coding eGene, PIP >= {a.min_pip})")
    print(f"  PIP median {out['pip'].median():.2f}; beta > 0: {(out['beta'] > 0).mean():.0%}")
    print(f"Saved {VAR}\n\nNext, score them with AlphaGenome:\n"
          f"  python 02_score_alphagenome_hepatic.py --input {VAR} --chunks {CHUNKS} --expression-only")


# ============================================================ analyse
def load_ag():
    files = sorted(glob.glob(f"{CHUNKS}/chunk_*.parquet"))
    if not files:
        sys.exit(f"No AlphaGenome scores in {CHUNKS} - run step 2 with --chunks {CHUNKS}")
    cols = ["variant_id", "gene_id", "gene_name", "gene_type", "output_type", "variant_scorer",
            "cell", "raw_score", "quantile_score"]
    s = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    s = s[(s["output_type"] == "RNA_SEQ") & (s["gene_type"] == "protein_coding") & s["cell"].isin(HEP)]
    act = s[s["variant_scorer"].str.contains("Active")]
    lfc = s[~s["variant_scorer"].str.contains("Active")]
    ag = (lfc.groupby(["variant_id", "gene_id", "gene_name"])
             .agg(ag_score=("raw_score", "mean"), ag_q=("quantile_score", lambda x: x.abs().max()))
             .reset_index())
    lvl = act.groupby(["variant_id", "gene_id"])["raw_score"].mean().rename("ag_expr").reset_index()
    ag = ag.merge(lvl, on=["variant_id", "gene_id"], how="left")
    print(f"AlphaGenome scores: {ag['variant_id'].nunique()} variants, {len(ag):,} variant-gene pairs")
    return ag


def binom(k, n):
    return stats.binomtest(int(k), int(n), 0.5, "greater").pvalue if n else np.nan


def dir_row(d, label):
    ok = np.sign(d["ag_score"]) == np.sign(d["beta"])
    lo, hi = stats.binomtest(int(ok.sum()), len(d)).proportion_ci() if len(d) else (np.nan, np.nan)
    return {"subset": label, "variants": len(d), "AlphaGenome_correct": round(ok.mean(), 3) if len(d) else np.nan,
            "95%_CI": f"{lo:.2f}-{hi:.2f}" if len(d) else "", "p_vs_50%": binom(ok.sum(), len(d)),
            "spearman_r": round(stats.spearmanr(d["ag_score"], d["beta"])[0], 3) if len(d) > 5 else np.nan}


def mcnemar(a_hit, b_hit):
    b, c = int((a_hit & ~b_hit).sum()), int((~a_hit & b_hit).sum())
    return b, c, (stats.binomtest(b, b + c, 0.5).pvalue if b + c else np.nan)


def analyse(a):
    var = pd.read_csv(VAR)
    ag = load_ag()
    genes = gencode_genes()
    pc = genes[genes["gene_type"] == "protein_coding"]

    # ---------- 1. DIRECTION (AlphaGenome score of the TRUE eGene vs real beta)
    d = var.merge(ag, left_on=["variant_id", "eGene"], right_on=["variant_id", "gene_id"], how="inner")
    d = d[d["ag_score"].abs() > 0]
    print(f"  eGene scored by AlphaGenome for {len(d)} of {len(var)} variants")
    expr_cut = ag["ag_expr"].median()
    rng = np.random.default_rng(3)
    ctrl = []                                     # control: a random OTHER gene's AG sign vs the eGene beta
    for _, r in d.iterrows():
        o = ag[(ag["variant_id"] == r["variant_id"]) & (ag["gene_id"] != r["eGene"])]
        if len(o):
            ctrl.append({"ag_score": o["ag_score"].iloc[rng.integers(len(o))], "beta": r["beta"]})
    q_hi = d["ag_score"].abs().quantile(0.75)
    rows = [dir_row(d, "all benchmark eQTLs"),
            dir_row(d[d["pip"] >= 0.9], "eQTL PIP >= 0.9"),
            dir_row(d[d["ag_q"] >= 0.99], "AlphaGenome confident (|quantile| >= 0.99)"),
            dir_row(d[d["ag_score"].abs() >= q_hi], "largest 25% AlphaGenome effects"),
            dir_row(d[d["ag_expr"] >= expr_cut], "eGene well expressed in liver"),
            dir_row(d[d["beta"].abs() >= d["beta"].abs().median()], "stronger eQTLs (|beta| above median)"),
            dir_row(pd.DataFrame(ctrl), "CONTROL: random other gene (should be ~50%)")]
    dirt = pd.DataFrame(rows)

    # ---------- 2. GENE CHOICE
    near = {}
    for _, v in var.iterrows():
        sub = pc[pc["chrom"] == v["chrom"]]
        if len(sub):
            near[v["variant_id"]] = sub.loc[(sub["tss"] - v["pos"]).abs().idxmin(), "gene_id"]
    g = var.set_index("variant_id")[["eGene", "pip", "beta"]].copy()
    g["nearest"] = g.index.map(near)
    agv = ag[ag["variant_id"].isin(g.index)].assign(a=lambda x: x["ag_score"].abs())
    top = agv.sort_values("a", ascending=False).drop_duplicates("variant_id").set_index("variant_id")["gene_id"]
    agx = agv[agv["ag_expr"] >= expr_cut]
    top_x = agx.sort_values("a", ascending=False).drop_duplicates("variant_id").set_index("variant_id")["gene_id"]
    g["AG_top"] = g.index.map(top)
    g["AG_top_expressed"] = g.index.map(top_x)
    g["n_genes"] = g.index.map(agv.groupby("variant_id").size())
    g = g.dropna(subset=["AG_top", "nearest"])
    rank = agv.assign(r=agv.groupby("variant_id")["a"].rank(ascending=False))
    rk = rank.set_index(["variant_id", "gene_id"])["r"]
    g["eGene_rank_AG"] = [rk.get((v, e), np.nan) for v, e in zip(g.index, g["eGene"])]
    near_hit = g["nearest"] == g["eGene"]
    choice = []
    for col, label in [("AG_top", "AlphaGenome top gene"), ("AG_top_expressed", "AlphaGenome top expressed gene")]:
        hit = g[col] == g["eGene"]
        b, c, p = mcnemar(hit, near_hit)
        choice.append({"method": label, "variants": len(g), "AlphaGenome_correct": round(hit.mean(), 3),
                       "nearest_gene_correct": round(near_hit.mean(), 3), "only_AlphaGenome_right": b,
                       "only_nearest_right": c, "p_McNemar": p})
    # where nearest gene is NOT the eGene: can AlphaGenome rescue it?
    hard = g[~near_hit]
    hit_h = hard["AG_top_expressed"] == hard["eGene"]
    choice.append({"method": "AlphaGenome when nearest gene is WRONG", "variants": len(hard),
                   "AlphaGenome_correct": round(hit_h.mean(), 3), "nearest_gene_correct": 0.0,
                   "only_AlphaGenome_right": int(hit_h.sum()), "only_nearest_right": 0, "p_McNemar": np.nan})
    # combined rule: nearest gene unless AlphaGenome is confident about another gene
    conf = agx[agx["ag_q"] >= 0.99].sort_values("a", ascending=False).drop_duplicates("variant_id") \
        .set_index("variant_id")["gene_id"]
    comb = g["nearest"].copy()
    comb.loc[comb.index.isin(conf.index)] = conf.reindex(comb.index[comb.index.isin(conf.index)])
    hit_c = comb == g["eGene"]
    b, c, p = mcnemar(hit_c, near_hit)
    choice.append({"method": "COMBINED: nearest gene, overridden by confident AlphaGenome", "variants": len(g),
                   "AlphaGenome_correct": round(hit_c.mean(), 3), "nearest_gene_correct": round(near_hit.mean(), 3),
                   "only_AlphaGenome_right": b, "only_nearest_right": c, "p_McNemar": p})
    ch = pd.DataFrame(choice)

    os.makedirs(RES, exist_ok=True)
    dirt.to_csv(f"{RES}/benchmark_direction.csv", index=False)
    ch.to_csv(f"{RES}/benchmark_gene_choice.csv", index=False)
    g.to_csv(f"{RES}/benchmark_per_variant.csv")
    pd.set_option("display.width", 220)
    print("\n=== 1. DIRECTION: AlphaGenome's predicted change for the true eGene vs real eQTL beta ===")
    print(dirt.to_string(index=False))
    print("\n=== 2. GENE CHOICE: which method names the real eGene? ===")
    print(ch.to_string(index=False))
    print(f"\nMedian rank of the true eGene in AlphaGenome's list: {np.nanmedian(g['eGene_rank_AG']):.0f} "
          f"of {g['n_genes'].median():.0f} genes per variant; random pick would be right "
          f"{(1 / g['n_genes']).mean():.1%} of the time")
    print("\nReading it: DIRECTION well above 50% (control ~50%) = AlphaGenome predicts real effect direction."
          "\n            GENE CHOICE above nearest gene with small p = AlphaGenome adds value over the standard rule.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--min-pip", type=float, default=0.5)
    a = ap.parse_args()
    if a.prepare:
        prepare(a)
    elif a.analyse:
        analyse(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
