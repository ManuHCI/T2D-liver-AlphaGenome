"""
Step 6 - Test the NOVEL step alone (Level 0 -> Level 1) against real GENETIC data.

Question: when a T2D risk variant changes a gene in the human liver (a liver eQTL),
          does AlphaGenome pick the RIGHT gene and the RIGHT direction - and better than
          the "nearest gene" rule used in most GWAS papers?

Yardstick : GTEx liver eQTLs from the eQTL Catalogue (dataset QTD000266, n = 208 livers)
            SuSiE fine-mapped credible sets (FTP; the old REST API was retired):
            variant, gene, PIP (probability it is the causal variant), beta, p-value
Both the eQTL beta and the AlphaGenome score refer to the ALT allele, so their signs are
directly comparable (allele swaps are detected and corrected).

Tests
  1. DIRECTION   : sign(AlphaGenome score) == sign(eQTL beta)?   chance = 50 %
                   reported for all significant pairs, strong eQTLs, fine-mapped (PIP) pairs,
                   and pairs where AlphaGenome itself is confident
  2. GENE CHOICE : for variants that are liver eQTLs, which gene is the real eGene?
                   AlphaGenome's top gene  vs  the nearest gene   (paired McNemar test)

Run (on your machine, needs internet; ~10-20 min first time, cached afterwards):
  python 06_liver_eqtl_comparison.py
Options: --dataset QTD000266 --p-sig 1e-4
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import requests
from scipy import stats

warnings.filterwarnings("ignore")
API = "https://www.ebi.ac.uk/eqtl/api/v2"
FTP = "https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/susie"
DATA, RES, CACHE = "data", "results", "data/eqtl"
HEP = ["HepG2", "liver", "hepatocyte"]


# ================================================================ eQTL Catalogue
META = ("https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/master/"
        "data_tables/dataset_metadata.tsv")


def dataset_info(ds):
    """study id + labels for a dataset, from the eQTL Catalogue metadata table (GitHub)."""
    path = f"{CACHE}/dataset_metadata.tsv"
    if not os.path.exists(path):
        try:
            r = requests.get(META, timeout=120)
            r.raise_for_status()
            open(path, "w", encoding="utf-8").write(r.text)
        except Exception as e:
            print(f"  metadata table not reachable ({e}); assuming GTEx study QTS000015")
            return {"dataset_id": ds, "study_id": "QTS000015", "study_label": "GTEx", "tissue_label": "liver"}
    md = pd.read_csv(path, sep="\t")
    row = md[md["dataset_id"] == ds]
    if row.empty:
        sys.exit(f"{ds} not in the eQTL Catalogue metadata - send this to Claude")
    return row.iloc[0].to_dict()


def fetch_credible_sets(ds, study):
    path = f"{CACHE}/{ds}.credible_sets.tsv.gz"
    if not os.path.exists(path):
        url = f"{FTP}/{study}/{ds}/{ds}.credible_sets.tsv.gz"
        print(f"  downloading {url}")
        r = requests.get(url, timeout=900)
        if r.status_code != 200:
            sys.exit(f"Credible sets not found (HTTP {r.status_code}) at {url} - send this to Claude")
        open(path, "wb").write(r.content)
    cs = pd.read_csv(path, sep="\t")
    print(f"  liver credible sets: {len(cs):,} variant-gene rows, columns: {', '.join(cs.columns[:14])}")
    if "gene_id" not in cs.columns:
        cs["gene_id"] = cs["molecular_trait_id"]
    cs["gene_id"] = cs["gene_id"].astype(str).str.split(".").str[0]
    return cs


def eqtl_pairs(cs, var):
    """credible-set rows for OUR variants (either allele orientation)."""
    v = var.copy()
    v["c"] = v["chrom"].str.replace("chr", "", regex=False)
    ids = {}
    for _, r in v.iterrows():
        for pre in ("chr", ""):
            ids[f"{pre}{r['c']}_{int(r['pos'])}_{r['ref']}_{r['alt']}"] = (r["variant_id"], 1)
            ids[f"{pre}{r['c']}_{int(r['pos'])}_{r['alt']}_{r['ref']}"] = (r["variant_id"], -1)
    hit = cs[cs["variant"].isin(ids)].copy()
    hit["variant_id"] = hit["variant"].map(lambda x: ids[x][0])
    flip = hit["variant"].map(lambda x: ids[x][1])
    hit["beta"] = pd.to_numeric(hit["beta"], errors="coerce") * flip      # beta for OUR alt allele
    hit["pvalue"] = pd.to_numeric(hit.get("pvalue"), errors="coerce")
    print(f"  T2D variants inside a liver eQTL credible set: {hit['variant_id'].nunique()} "
          f"({len(hit)} variant-gene pairs; {int((flip < 0).sum())} with swapped alleles, corrected)")
    return hit


# ================================================================ AlphaGenome scores
def load_ag():
    files = sorted(glob.glob(f"{DATA}/scores_chunks/chunk_*.parquet"))
    if not files:
        sys.exit("No AlphaGenome score chunks found (data/scores_chunks).")
    cols = ["variant_id", "gene_id", "gene_name", "gene_type", "output_type", "variant_scorer",
            "cell", "raw_score", "quantile_score"]
    s = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    s = s[(s["output_type"] == "RNA_SEQ") & (s["gene_type"] == "protein_coding") & s["cell"].isin(HEP)]
    act = s[s["variant_scorer"].str.contains("Active")]
    lfc = s[~s["variant_scorer"].str.contains("Active")]
    ag = (lfc.groupby(["variant_id", "gene_id", "gene_name"])
             .agg(ag_score=("raw_score", "mean"), ag_q=("quantile_score", lambda x: x.abs().max()))
             .reset_index())
    lvl = act.groupby("gene_id")["raw_score"].mean()
    ag["ag_expr"] = ag["gene_id"].map(lvl)
    return ag


def nearest_gene(var):
    tss = []
    with gzip.open(f"{DATA}/gencode.basic.gtf.gz", "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.split("\t", 8)
            if c[2] != "gene" or 'gene_type "protein_coding"' not in c[8]:
                continue
            gid = re.search(r'gene_id "([^".]+)', c[8]).group(1)
            tss.append((c[0], int(c[3]) if c[6] == "+" else int(c[4]), gid))
    t = pd.DataFrame(tss, columns=["chrom", "tss", "gene_id"])
    out = {}
    for _, v in var.iterrows():
        sub = t[t["chrom"] == v["chrom"]]
        if len(sub):
            out[v["variant_id"]] = sub.loc[(sub["tss"] - v["pos"]).abs().idxmin(), "gene_id"]
    return out


# ================================================================ analysis
def binom(k, n):
    return stats.binomtest(int(k), int(n), 0.5, "greater").pvalue if n else np.nan


def direction_rows(m, label):
    ok = np.sign(m["ag_score"]) == np.sign(m["beta"])
    return {"subset": label, "pairs": len(m), "variants": m["variant_id"].nunique(),
            "AlphaGenome_correct": round(ok.mean(), 3) if len(m) else np.nan,
            "chance": 0.5, "p_binomial": binom(ok.sum(), len(m)),
            "spearman_r": round(stats.spearmanr(m["ag_score"], m["beta"])[0], 3) if len(m) > 5 else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="QTD000266")
    ap.add_argument("--p-sig", type=float, default=1e-4)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    info = dataset_info(a.dataset)
    print(f"eQTL dataset {a.dataset}: {info.get('study_label')} | {info.get('tissue_label')} | "
          f"{info.get('quant_method', 'ge')} | n = {info.get('sample_size', '?')}")
    var = pd.read_csv(f"{DATA}/t2d_variants_for_alphagenome.csv")
    cs = fetch_credible_sets(a.dataset, info["study_id"])
    eq = eqtl_pairs(cs, var)
    if eq.empty:
        sys.exit("None of the T2D variants is in a liver eQTL credible set - send this output to Claude.")

    print("Loading AlphaGenome liver scores ...")
    ag = load_ag()
    m = eq.merge(ag, on=["variant_id", "gene_id"], how="inner")
    m = m[m["ag_score"].abs() > 0]
    print(f"  {len(m):,} eQTL pairs also have an AlphaGenome score "
          f"(others: gene not protein-coding or outside AlphaGenome's scored genes)")

    # ---------------- 1. DIRECTION
    sig = m
    expr_cut = ag["ag_expr"].median()
    # control: AlphaGenome score of a RANDOM OTHER gene at the same variant vs the eGene beta
    rng = np.random.default_rng(11)
    ctrl = []
    for _, r in m.iterrows():
        others = ag[(ag["variant_id"] == r["variant_id"]) & (ag["gene_id"] != r["gene_id"])]
        if len(others):
            ctrl.append({"variant_id": r["variant_id"], "ag_score": others["ag_score"].iloc[rng.integers(len(others))],
                         "beta": r["beta"]})
    ctrl = pd.DataFrame(ctrl)
    rows = [direction_rows(sig, "all fine-mapped liver eQTL pairs"),
            direction_rows(sig[sig["pip"] >= 0.2], "eQTL PIP >= 0.2"),
            direction_rows(sig[sig["pip"] >= 0.5], "eQTL PIP >= 0.5 (likely causal)"),
            direction_rows(sig[sig["pvalue"] < 1e-8], "strong eQTL p < 1e-8"),
            direction_rows(sig[sig["ag_q"] >= 0.99], "AlphaGenome confident (|quantile| >= 0.99)"),
            direction_rows(sig[sig["ag_expr"] >= expr_cut], "gene expressed in liver (AlphaGenome level)"),
            direction_rows(ctrl, "CONTROL: random other gene at same variant")]
    dirt = pd.DataFrame(rows)

    # ---------------- 2. GENE CHOICE
    near = nearest_gene(var)
    eg = (sig.sort_values("pip", ascending=False).groupby("variant_id").first()[["gene_id", "pvalue", "pip"]]
             .rename(columns={"gene_id": "eGene"}))
    agc = ag[ag["variant_id"].isin(eg.index)].copy()
    pick_all = agc.loc[agc.groupby("variant_id")["ag_score"].apply(lambda x: x.abs().idxmax())] \
        .set_index("variant_id")["gene_id"]
    agx = agc[agc["ag_expr"] >= expr_cut]
    pick_expr = agx.loc[agx.groupby("variant_id")["ag_score"].apply(lambda x: x.abs().idxmax())] \
        .set_index("variant_id")["gene_id"]
    eg["AlphaGenome_top"] = eg.index.map(pick_all)
    eg["AlphaGenome_top_expressed"] = eg.index.map(pick_expr)
    eg["nearest_gene"] = eg.index.map(near)
    # rank of the true eGene in AlphaGenome's list
    def rank_of(v, gene):
        d = agc[agc["variant_id"] == v].assign(a=lambda x: x["ag_score"].abs()).sort_values("a", ascending=False)
        ids = list(d["gene_id"])
        return ids.index(gene) + 1 if gene in ids else np.nan
    eg["eGene_rank_in_AlphaGenome"] = [rank_of(v, r["eGene"]) for v, r in eg.iterrows()]
    eg["n_genes_scored"] = eg.index.map(agc.groupby("variant_id").size())

    def choice_row(col, label, sub=eg):
        hit_ag = sub[col] == sub["eGene"]
        hit_nb = sub["nearest_gene"] == sub["eGene"]
        b, c = int((hit_ag & ~hit_nb).sum()), int((~hit_ag & hit_nb).sum())
        p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else np.nan     # exact McNemar
        return {"subset": label, "variants": len(sub), "AlphaGenome_correct": round(hit_ag.mean(), 3),
                "nearest_gene_correct": round(hit_nb.mean(), 3), "only_AlphaGenome_right": b,
                "only_nearest_right": c, "p_McNemar": p,
                "random_gene_correct": round((1 / sub["n_genes_scored"]).mean(), 3)}
    ch = pd.DataFrame([choice_row("AlphaGenome_top", "all eQTL variants: AlphaGenome top gene"),
                       choice_row("AlphaGenome_top_expressed", "all eQTL variants: AlphaGenome top expressed gene"),
                       choice_row("AlphaGenome_top_expressed", "fine-mapped eQTL (PIP >= 0.2)", eg[eg["pip"] >= 0.2])])

    # ---------------- save + print
    pd.set_option("display.width", 220)
    dirt.to_csv(f"{RES}/eqtl_direction.csv", index=False)
    ch.to_csv(f"{RES}/eqtl_gene_choice.csv", index=False)
    eg.to_csv(f"{RES}/eqtl_gene_choice_per_variant.csv")
    sig.to_csv(f"{RES}/eqtl_pairs_significant.csv", index=False)
    print(f"\nT2D variants that are fine-mapped liver eQTLs: {len(eg)} of {len(var)}")
    print("\n=== 1. DIRECTION: does AlphaGenome's up/down match the real liver eQTL? ===")
    print(dirt.to_string(index=False))
    print("\n=== 2. GENE CHOICE: AlphaGenome top gene vs nearest gene (which one is the real eGene?) ===")
    print(ch.to_string(index=False))
    print(f"\nMedian rank of the true eGene in AlphaGenome's list: "
          f"{eg['eGene_rank_in_AlphaGenome'].median():.0f} of {eg['n_genes_scored'].median():.0f} genes")
    print("\nHow to read it:")
    print("  DIRECTION  : AlphaGenome_correct well above 0.5 with small p (and ~0.5 in the negative control) = real signal")
    print("  GENE CHOICE: AlphaGenome_correct > nearest_gene_correct with small p_McNemar = AlphaGenome picks genes better")
    print(f"\nSaved: {RES}/eqtl_direction.csv, eqtl_gene_choice.csv, eqtl_gene_choice_per_variant.csv")


if __name__ == "__main__":
    main()
