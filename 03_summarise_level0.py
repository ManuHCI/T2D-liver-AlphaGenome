"""
Step 3 - Turn AlphaGenome scores into the LEVEL-0 lists (roots of the diabetic tree).

Every score is re-oriented to read "DIABETIC (risk allele) minus NORMAL":
    diabetic_score = raw_score(ALT - REF) * risk_sign
so  > 0  means higher in the diabetic sequence,  < 0  lower.

Significance : |quantile_score| >= 0.99 (change is in the top 1 % of all
               variant effects AlphaGenome has seen for that track)
Magnitude    : raw_score (quantile saturates, so it is not used for ranking)
Robust call  : HepG2 and liver tissue agree in direction (hepatocyte too when present)
Liver-specific: the hepatic effect is clearly larger than in K562 (non-liver control)

Output : results/T2D_hepatocyte_level0.xlsx with sheets
   1_genes_per_variant   every variant -> protein-coding gene change
   2_genes_summary       one row per gene, combined over variants (PIP weighted)
   3_TF_expression       subset of sheet 2 that are transcription factors
   4_TF_binding          variant changes binding of a TF (HepG2 ChIP tracks)
   5_TF_binding_summary  one row per TF
   6_splicing            genes with splice changes
   7_chromatin           accessibility / histone change at each variant
   8_variants            the input variants with counts of effects
Run : python 03_summarise_level0.py            (add --test to use the test chunks)
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

VAR_FILE = "data/t2d_variants_for_alphagenome.csv"
CHUNK_DIR = "data/scores_chunks"
TF_FILE = "data/human_TF_list.txt"
TF_URL = "http://humantfs.ccbr.utoronto.ca/download/v_1.01/TF_names_v_1.01.txt"
OUT = "results/T2D_hepatocyte_level0.xlsx"
Q_CUT = 0.99
HEP_CELLS = ["HepG2", "liver", "hepatocyte"]


def load_tf_list():
    if not os.path.exists(TF_FILE):
        try:
            import requests
            r = requests.get(TF_URL, timeout=60)
            r.raise_for_status()
            open(TF_FILE, "w").write(r.text)
        except Exception as e:
            sys.exit(f"Could not download TF list ({e}).\nDownload {TF_URL} "
                     f"by browser and save it as {TF_FILE}")
    tfs = {l.strip() for l in open(TF_FILE) if l.strip()}
    print(f"Human TF list (Lambert 2018): {len(tfs)} TFs")
    return tfs


def direction(x):
    return np.where(x > 0, "UP in diabetic", np.where(x < 0, "DOWN in diabetic", "no change"))


def per_cell_wide(df, keys):
    """mean diabetic score and max |quantile| per key per cell -> wide table."""
    g = (df.groupby(keys + ["cell"])
           .agg(score=("diabetic_score", "mean"), q=("abs_q", "max"), n_tracks=("raw_score", "size"))
           .reset_index())
    wide = g.pivot_table(index=keys, columns="cell", values=["score", "q"])
    wide.columns = [f"{v}_{c}" for v, c in wide.columns]
    return wide.reset_index()


def add_calls(w, q_cut=Q_CUT):
    hep = [c for c in HEP_CELLS if f"score_{c}" in w.columns]
    w["hepatic_score"] = w[[f"score_{c}" for c in hep]].mean(axis=1)
    w["hepatic_q"] = w[[f"q_{c}" for c in hep]].max(axis=1)
    w["significant"] = w["hepatic_q"] >= q_cut
    signs = np.sign(w[[f"score_{c}" for c in hep]])
    # agreement needs at least two liver cell types with a score
    w["cells_agree"] = (signs.nunique(axis=1, dropna=True) == 1) & (signs.notna().sum(axis=1) >= 2)
    if "score_K562" in w.columns:
        w["liver_specific"] = w["hepatic_score"].abs() > 2 * w["score_K562"].abs()
    w["direction"] = direction(w["hepatic_score"])
    w["confidence"] = np.select(
        [w["significant"] & w["cells_agree"], w["significant"]], ["high", "medium"], "low")
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--q-tf", type=float, default=0.99, help="quantile cut-off for TF binding")
    ap.add_argument("--z-tf", type=float, default=4, help="TF-specific: |z| vs other TFs at same variant")
    ap.add_argument("--enh-min", type=int, default=100, help="TF tracks changed to call an enhancer-level effect")
    ap.add_argument("--expr-pct", type=float, default=50,
                    help="keep genes whose predicted liver expression is above this percentile")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{CHUNK_DIR}/{'test_' if a.test else 'chunk_'}*.parquet"))
    if not files:
        sys.exit("No score files found - run step 2 first.")
    s = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    var = pd.read_csv(VAR_FILE)
    s = s.merge(var[["variant_id", "rsid", "pip", "risk_sign", "risk_allele", "consequence",
                     "study_locus_id"]], on="variant_id", how="inner")
    s["diabetic_score"] = s["raw_score"] * s["risk_sign"]
    s["abs_q"] = s["quantile_score"].abs()
    # RNA_SEQ_ACTIVE rows = predicted expression LEVEL (not change); used only as a filter
    is_active = s["variant_scorer"].astype(str).str.contains("ActiveScorer")
    act, s = s[is_active], s[~is_active]
    print(f"{len(s):,} score rows, {s['variant_id'].nunique()} variants")
    tfs = load_tf_list()
    os.makedirs("results", exist_ok=True)

    # 1. gene expression per variant (RNA-seq, protein coding only)
    rna = s[(s["output_type"] == "RNA_SEQ") & (s["gene_type"] == "protein_coding")]
    gv = add_calls(per_cell_wide(rna, ["variant_id", "gene_id", "gene_name"]))
    gv = gv.merge(var[["variant_id", "rsid", "pip"]], on="variant_id")
    gv["is_TF"] = gv["gene_name"].isin(tfs)

    # expression filter: the fold-change scorer favours genes that are barely expressed
    # (tiny baseline -> large fold change). Keep genes AlphaGenome predicts are expressed in liver.
    a_hep = act[(act["output_type"] == "RNA_SEQ") & act["cell"].isin(HEP_CELLS)
                & (act["gene_type"] == "protein_coding")]
    if len(a_hep):
        level = a_hep.groupby("gene_id")["raw_score"].mean()
        cut = np.percentile(level, a.expr_pct)
        gv["liver_expression"] = gv["gene_id"].map(level)
        gv["expressed_in_liver"] = gv["liver_expression"] >= cut
        print(f"Expression filter: {int((level >= cut).sum())}/{len(level)} genes kept "
              f"(>= {a.expr_pct:.0f}th percentile of predicted liver expression)")
    else:
        print("WARNING: no RNA_SEQ_ACTIVE scores - expression filter skipped (re-run step 2)")
        gv["liver_expression"], gv["expressed_in_liver"] = np.nan, True
    gv = gv.sort_values(["significant", "hepatic_score"], key=lambda c: c.abs() if c.name == "hepatic_score" else c,
                        ascending=False)

    # 2. one row per gene: PIP-weighted over significant variants
    sig = gv[gv["significant"] & gv["expressed_in_liver"]].copy()
    sig["w"] = sig["hepatic_score"] * sig["pip"]
    gs = (sig.groupby(["gene_id", "gene_name"])
             .agg(n_variants=("variant_id", "nunique"),
                  variants=("rsid", lambda x: ";".join(sorted(set(map(str, x))))),
                  weighted_score=("w", "sum"),
                  max_abs_score=("hepatic_score", lambda x: x.abs().max()),
                  n_high_conf=("confidence", lambda x: (x == "high").sum()),
                  liver_expression=("liver_expression", "first"))
             .reset_index())
    gs["direction"] = direction(gs["weighted_score"])
    gs["is_TF"] = gs["gene_name"].isin(tfs)
    gs = gs.sort_values("weighted_score", key=abs, ascending=False)

    # 3. TF expression = TF genes from sheet 2
    tf_expr = gs[gs["is_TF"]].copy()

    # 4. TF binding change at the variant (ChIP-TF tracks)
    chip = s[s["output_type"] == "CHIP_TF"]
    tb = add_calls(per_cell_wide(chip, ["variant_id", "transcription_factor"]), q_cut=a.q_tf)
    tb = tb.merge(var[["variant_id", "rsid", "pip"]], on="variant_id")

    # A variant that opens/closes a whole regulatory element changes ALL TF tracks together
    # (87 % of hits in the first run came from 22 such variants). Separate the two:
    #   enhancer-level effect : variant changes many TF tracks (accessibility change)
    #   TF-specific effect    : this TF changes much more than the other TFs at the same variant
    hg = chip[chip["cell"] == "HepG2"].copy()
    hg["hit"] = hg["abs_q"] >= a.q_tf
    grp = hg.groupby("variant_id")["diabetic_score"]
    hg["z"] = (hg["diabetic_score"] - grp.transform("median")) / (grp.transform("std") + 1e-9)
    zt = (hg.loc[hg.groupby(["variant_id", "transcription_factor"])["z"].apply(lambda x: x.abs().idxmax())]
            [["variant_id", "transcription_factor", "z"]].rename(columns={"z": "z_within_variant"}))
    tb = tb.merge(zt, on=["variant_id", "transcription_factor"], how="left")
    tb["tf_specific"] = tb["significant"] & (tb["z_within_variant"].abs() >= a.z_tf)
    tb = tb.sort_values("hepatic_score", key=abs, ascending=False)

    enh = hg.groupby("variant_id").agg(n_TF_tracks_changed=("hit", "sum"), n_TF_tracks=("hit", "size"),
                                       mean_TF_change=("diabetic_score", "mean")).reset_index()
    dn = s[(s["output_type"] == "DNASE") & (s["cell"] == "HepG2")][["variant_id", "diabetic_score", "quantile_score"]]
    enh = enh.merge(dn.rename(columns={"diabetic_score": "dnase_change", "quantile_score": "dnase_q"}),
                    on="variant_id", how="left").merge(var[["variant_id", "rsid", "pip", "consequence"]], on="variant_id")
    enh["enhancer_level"] = enh["n_TF_tracks_changed"] >= a.enh_min
    enh["element"] = np.where(enh["mean_TF_change"] < 0, "CLOSED in diabetic", "OPENED in diabetic")
    sig_genes = gv[gv["significant"] & gv["expressed_in_liver"]]
    enh["genes_changed"] = enh["variant_id"].map(
        sig_genes.groupby("variant_id").apply(lambda d: ";".join(f"{g}({'+' if v > 0 else '-'})"
                                               for g, v in zip(d["gene_name"], d["hepatic_score"])))).fillna("")
    enh = enh.sort_values("n_TF_tracks_changed", ascending=False)

    tbs = (tb[tb["tf_specific"]].groupby("transcription_factor")
             .agg(n_variants=("variant_id", "nunique"),
                  n_gain=("hepatic_score", lambda x: (x > 0).sum()),
                  n_loss=("hepatic_score", lambda x: (x < 0).sum()),
                  mean_score=("hepatic_score", "mean"),
                  n_high_conf=("confidence", lambda x: (x == "high").sum()))
             .reset_index().sort_values("n_variants", ascending=False))

    # 6. splicing
    spl = s[s["output_type"].isin(["SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS"])
            & (s["gene_type"] == "protein_coding")]
    sp = (spl.groupby(["variant_id", "rsid", "gene_name", "output_type"])
             .agg(max_abs_raw=("raw_score", lambda x: x.abs().max()),
                  max_abs_q=("abs_q", "max")).reset_index())
    sp = sp[sp["max_abs_q"] >= Q_CUT].sort_values("max_abs_raw", ascending=False)

    # 7. chromatin at the variant
    chrom = s[s["output_type"].isin(["DNASE", "ATAC", "CHIP_HISTONE"])]
    ch = add_calls(per_cell_wide(chrom.assign(mark=chrom["histone_mark"].fillna(chrom["output_type"])),
                                 ["variant_id", "mark"]))

    # 8. variant overview
    vo = var.copy()
    vo["n_sig_genes"] = vo["variant_id"].map(gv[gv["significant"]].groupby("variant_id").size()).fillna(0).astype(int)
    vo["n_sig_TF_binding"] = vo["variant_id"].map(tb[tb["significant"]].groupby("variant_id").size()).fillna(0).astype(int)
    vo["scored"] = vo["variant_id"].isin(s["variant_id"])

    # small CSVs = roots for the tree (step 4)
    gs.to_csv("results/level0_genes.csv", index=False)
    tb[tb["tf_specific"]].to_csv("results/level0_tf_binding.csv", index=False)
    enh.to_csv("results/level0_enhancers.csv", index=False)
    gv[gv["significant"]].to_csv("results/level0_gene_variant_links.csv", index=False)

    with pd.ExcelWriter(OUT) as xw:
        gv.to_excel(xw, sheet_name="1_genes_per_variant", index=False)
        gs.to_excel(xw, sheet_name="2_genes_summary", index=False)
        tf_expr.to_excel(xw, sheet_name="3_TF_expression", index=False)
        tb.to_excel(xw, sheet_name="4_TF_binding", index=False)
        tbs.to_excel(xw, sheet_name="5_TF_binding_summary", index=False)
        enh[enh["n_TF_tracks_changed"] > 0].to_excel(xw, sheet_name="4b_enhancer_effects", index=False)
        sp.to_excel(xw, sheet_name="6_splicing", index=False)
        ch.to_excel(xw, sheet_name="7_chromatin", index=False)
        vo.to_excel(xw, sheet_name="8_variants", index=False)

    print("\n=== LEVEL 0 (roots of the diabetic hepatocyte tree) ===")
    print(f"Variants scored                    : {vo['scored'].sum()}")
    print(f"Variants with >=1 gene change      : {(vo['n_sig_genes'] > 0).sum()}")
    print(f"Protein-coding genes changed       : {len(gs)}  "
          f"(UP {(gs['weighted_score'] > 0).sum()}, DOWN {(gs['weighted_score'] < 0).sum()})")
    print(f"  of which transcription factors   : {len(tf_expr)}")
    e = enh[enh["enhancer_level"]]
    print(f"Enhancer-level variants (>= {a.enh_min} TF tracks change together): {len(e)} "
          f"(closed {(e['mean_TF_change'] < 0).sum()}, opened {(e['mean_TF_change'] > 0).sum()})")
    print(f"TF-specific binding changes        : {tb['tf_specific'].sum()} variant-TF pairs, "
          f"{tb[tb['tf_specific']]['variant_id'].nunique()} variants, {len(tbs)} TFs")
    print(f"High-confidence gene calls         : {(gv['confidence'] == 'high').sum()}")
    if "liver_specific" in gv.columns:
        sg = gv[gv["significant"]]
        print(f"Liver-specific (vs K562) gene calls: {sg['liver_specific'].mean():.0%} of significant")
    print(f"\nTop 15 genes:\n{gs.head(15)[['gene_name', 'direction', 'weighted_score', 'n_variants', 'is_TF']].to_string(index=False)}")
    print(f"\nTop 15 TF-specific binding changes:\n{tbs.head(15).to_string(index=False)}")
    print(f"\nTop enhancer-level variants:\n"
          f"{e.head(12)[['rsid', 'pip', 'n_TF_tracks_changed', 'element', 'genes_changed']].to_string(index=False)}")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
