"""
Step 8 - HYBRID LEVEL 1: which gene from proximity, which direction from AlphaGenome.

Why (benchmark on 328 causal GTEx liver eQTLs, step 7):
  - nearest gene names the true target gene 62% of the time, AlphaGenome's top gene 42%
  - AlphaGenome's direction is right 82-95% of the time when it is confident
  - combined rule "nearest gene + confident AlphaGenome direction" got gene AND direction
    right 67% of the time, vs 36% for the old rule (AlphaGenome top gene)

Rule for every T2D variant
  1. gene      = nearest protein-coding gene (GENCODE TSS)
  2. direction = AlphaGenome's predicted change of THAT gene in liver (HepG2, liver, hepatocyte),
                 oriented as diabetic (risk allele) minus normal
  3. keep only if AlphaGenome is confident (|quantile| >= 0.99)
  No expression filter (it removed true target genes in the benchmark).

Writes the same files step 4 and step 5 read, into results_hybrid/ (old results kept):
  level0_genes.csv, level0_gene_variant_links.csv, level0_tf_binding.csv, level0_enhancers.csv
Run:
  python 08_hybrid_level1.py
  python 04_build_tree.py --res results_hybrid
  python 05_control_comparison.py --run --res results_hybrid
"""
import argparse
import glob
import gzip
import os
import re
import shutil

import numpy as np
import pandas as pd

HEP = ["HepG2", "liver", "hepatocyte"]


def nearest_genes(var, gtf):
    rows = []
    with gzip.open(gtf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.split("\t", 8)
            if c[2] != "gene" or 'gene_type "protein_coding"' not in c[8]:
                continue
            gid = re.search(r'gene_id "([^".]+)', c[8]).group(1)
            rows.append((c[0], int(c[3]) if c[6] == "+" else int(c[4]), gid))
    t = pd.DataFrame(rows, columns=["chrom", "tss", "gene_id"])
    out = []
    for _, v in var.iterrows():
        sub = t[t["chrom"] == v["chrom"]]
        if len(sub):
            d = (sub["tss"] - v["pos"]).abs()
            out.append({"variant_id": v["variant_id"], "gene_id": sub.loc[d.idxmin(), "gene_id"],
                        "distance_to_tss": int(d.min())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.99, help="AlphaGenome confidence cut-off (|quantile|)")
    ap.add_argument("--out", default="results_hybrid")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    var = pd.read_csv("data/t2d_variants_for_alphagenome.csv")
    tfs = {l.strip() for l in open("data/human_TF_list.txt") if l.strip()}
    near = nearest_genes(var, "data/gencode.basic.gtf.gz")

    cols = ["variant_id", "gene_id", "gene_name", "gene_type", "output_type", "variant_scorer",
            "cell", "raw_score", "quantile_score"]
    s = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/scores_chunks/chunk_*.parquet"))],
                  ignore_index=True)
    s = s[(s["output_type"] == "RNA_SEQ") & s["cell"].isin(HEP) & ~s["variant_scorer"].str.contains("Active")]
    s = s.merge(near, on=["variant_id", "gene_id"], how="inner")              # nearest gene only
    s = s.merge(var[["variant_id", "rsid", "pip", "risk_sign"]], on="variant_id")
    s["diabetic_score"] = s["raw_score"] * s["risk_sign"]
    per_cell = s.groupby(["variant_id", "gene_id", "gene_name", "cell"])["diabetic_score"].mean().unstack()
    g = (s.groupby(["variant_id", "rsid", "pip", "gene_id", "gene_name"])
          .agg(hepatic_score=("diabetic_score", "mean"),
               hepatic_q=("quantile_score", lambda x: x.abs().max()))
          .reset_index())
    sign_cells = np.sign(per_cell[[c for c in HEP if c in per_cell.columns]])
    agree = (sign_cells.nunique(axis=1) == 1) & (sign_cells.notna().sum(axis=1) >= 2)
    g = g.merge(agree.rename("cells_agree").reset_index()[["variant_id", "gene_id", "cells_agree"]],
                on=["variant_id", "gene_id"], how="left")
    g = g.merge(near[["variant_id", "distance_to_tss"]], on="variant_id")
    g["significant"] = g["hepatic_q"] >= a.q
    g["confidence"] = np.where(g["significant"] & g["cells_agree"], "high",
                               np.where(g["significant"], "medium", "low"))
    g["direction"] = np.where(g["hepatic_score"] > 0, "UP in diabetic", "DOWN in diabetic")
    g["is_TF"] = g["gene_name"].isin(tfs)
    links = g[g["significant"]].copy()

    sig = links.assign(w=links["hepatic_score"] * links["pip"])
    genes = (sig.groupby(["gene_id", "gene_name"])
                .agg(n_variants=("variant_id", "nunique"),
                     variants=("rsid", lambda x: ";".join(sorted(set(map(str, x))))),
                     weighted_score=("w", "sum"),
                     max_abs_score=("hepatic_score", lambda x: x.abs().max()),
                     n_high_conf=("confidence", lambda x: (x == "high").sum()))
                .reset_index())
    genes["direction"] = np.where(genes["weighted_score"] > 0, "UP in diabetic", "DOWN in diabetic")
    genes["is_TF"] = genes["gene_name"].isin(tfs)
    genes = genes.sort_values("weighted_score", key=abs, ascending=False)

    genes.to_csv(f"{a.out}/level0_genes.csv", index=False)
    links.to_csv(f"{a.out}/level0_gene_variant_links.csv", index=False)
    g.to_csv(f"{a.out}/level1_all_nearest_genes.csv", index=False)
    for f in ("level0_tf_binding.csv", "level0_enhancers.csv"):     # Level-0 mechanisms are unchanged
        if os.path.exists(f"results/{f}"):
            shutil.copy(f"results/{f}", f"{a.out}/{f}")

    old = pd.read_csv("results/level0_genes.csv") if os.path.exists("results/level0_genes.csv") else None
    print(f"T2D variants: {len(var)} | nearest gene scored by AlphaGenome: {g['variant_id'].nunique()}")
    print(f"Confident direction (|quantile| >= {a.q}): {links['variant_id'].nunique()} variants -> "
          f"{len(genes)} Level-1 genes (UP {(genes['weighted_score'] > 0).sum()}, "
          f"DOWN {(genes['weighted_score'] < 0).sum()}; TFs {int(genes['is_TF'].sum())})")
    if old is not None:
        ov = set(old["gene_name"]) & set(genes["gene_name"])
        print(f"Overlap with the OLD Level 1 ({len(old)} genes): {len(ov)} -> {', '.join(sorted(ov)[:25])}")
    print("\nTop 20 hybrid Level-1 genes:")
    print(genes.head(20)[["gene_name", "direction", "weighted_score", "n_variants", "variants", "is_TF"]]
          .to_string(index=False))
    for gn in ("PROX1", "HNF1A", "HNF4A", "CCND2", "PCBD1"):
        r = g[g["gene_name"] == gn]
        if len(r):
            print(f"  {gn}: " + "; ".join(f"{x.rsid} {x.direction} (q={x.hepatic_q:.3f}, "
                                          f"{'kept' if x.significant else 'not confident'})" for x in r.itertuples()))
        else:
            print(f"  {gn}: not the nearest gene to any T2D variant")
    print(f"\nSaved to {a.out}/  ->  next: python 04_build_tree.py --res {a.out}")


if __name__ == "__main__":
    main()
