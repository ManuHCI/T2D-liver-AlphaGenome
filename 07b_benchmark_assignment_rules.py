"""
Step 7b - Compare variant-to-gene ASSIGNMENT RULES on the liver eQTL benchmark (gene AND direction).

For each of the 328 causal GTEx liver eQTLs (step 7) a rule names one gene and a direction:
  hybrid           nearest protein-coding gene + AlphaGenome direction of that gene, kept if |quantile| >= 0.99
  AG top gene      gene with the largest AlphaGenome change, kept if |quantile| >= 0.99
  AG top expressed same, restricted to genes AlphaGenome predicts are expressed in liver (old Level-1 rule)
"Correct" = the named gene is the true eGene AND the predicted sign equals the sign of the eQTL beta.
Writes results/benchmark_rules.csv and results/benchmark_per_variant_scores.csv
Run:  python 07b_benchmark_assignment_rules.py
"""
import glob

import numpy as np
import pandas as pd

HEP = ["HepG2", "liver", "hepatocyte"]
cols = ["variant_id", "gene_id", "gene_type", "output_type", "variant_scorer", "cell", "raw_score", "quantile_score"]
s = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/scores_eqtl_benchmark/chunk_*.parquet"))])
s = s[(s.output_type == "RNA_SEQ") & (s.gene_type == "protein_coding") & s.cell.isin(HEP)
      & ~s.variant_scorer.str.contains("Active")]
ag = (s.groupby(["variant_id", "gene_id"])
      .agg(score=("raw_score", "mean"), q=("quantile_score", lambda x: x.abs().max())))
g = pd.read_csv("results/benchmark_per_variant.csv")
for col, name in [("nearest", "near"), ("AG_top", "top"), ("AG_top_expressed", "topx"), ("eGene", "egene")]:
    idx = list(zip(g["variant_id"], g[col]))
    g[f"{name}_score"] = [ag["score"].get(i, np.nan) for i in idx]
    g[f"{name}_q"] = [ag["q"].get(i, np.nan) for i in idx]
sb = np.sign(g["beta"])
rows = []
for label, gene, name in [("Hybrid\n(nearest + AG)", "nearest", "near"), ("AG top\ngene", "AG_top", "top"),
                          ("AG top\nexpressed", "AG_top_expressed", "topx")]:
    ok = (g[gene] == g["eGene"]) & (np.sign(g[f"{name}_score"]) == sb)
    kept = g[f"{name}_q"] >= 0.99
    rows.append({"label": label, "rule": gene, "variants": len(g), "correct_all": round(ok.mean(), 3),
                 "kept": int(kept.sum()), "correct_among_kept": round(ok[kept].mean(), 3)})
out = pd.DataFrame(rows)
out.to_csv("results/benchmark_rules.csv", index=False)
g.to_csv("results/benchmark_per_variant_scores.csv", index=False)
print(out.to_string(index=False))
