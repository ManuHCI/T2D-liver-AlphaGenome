"""
Step 2 - Apply AlphaGenome to the T2D variants and keep the hepatocyte outputs.

For every variant AlphaGenome reads 1 Mb of hg38 DNA around it, predicts all
outputs for the NORMAL (REF) and the variant (ALT) sequence, and returns
ALT-minus-REF scores. Step 3 turns these into "diabetic minus normal".

Biosamples kept (found automatically from AlphaGenome metadata):
  HepG2 (EFO:0001187)        primary cell - most TF ChIP-seq tracks
  liver tissue (UBERON:0002107)
  hepatocyte (CL:0000182)
  K562 (EFO:0002067)         NON-liver control, used only to test liver specificity

Scorers (8): RNA_SEQ, CAGE, DNASE, ATAC, CHIP_TF, CHIP_HISTONE, SPLICE_SITES,
             SPLICE_SITE_USAGE, SPLICE_JUNCTIONS

Usage
  set ALPHAGENOME_API_KEY=your_key            (Windows)   or put the key in api_key.txt
  python 02_score_alphagenome_hepatic.py --tracks-only   # count tracks per cell, no scoring
  python 02_score_alphagenome_hepatic.py --test 5        # score 5 variants first
  python 02_score_alphagenome_hepatic.py                 # full run (resumes if stopped)
"""
import argparse
import glob
import os
import sys

import pandas as pd
from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers

IN_FILE = "data/t2d_variants_for_alphagenome.csv"
CHUNK_DIR = "data/scores_chunks"
CHUNK = 25
HEPATIC = {"EFO:0001187": "HepG2", "UBERON:0002107": "liver", "CL:0000182": "hepatocyte"}
CONTROL = {"EFO:0002067": "K562"}
SCORER_KEYS = ["RNA_SEQ", "RNA_SEQ_ACTIVE", "CAGE", "DNASE", "ATAC", "CHIP_TF", "CHIP_HISTONE",
               "SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS"]
KEEP_COLS = ["variant_id", "gene_id", "gene_name", "gene_type", "gene_strand",
             "output_type", "variant_scorer", "track_name", "track_strand",
             "ontology_curie", "biosample_name", "biosample_type",
             "transcription_factor", "histone_mark", "raw_score", "quantile_score"]


def api_key():
    key = os.environ.get("ALPHAGENOME_API_KEY")
    if not key and os.path.exists("api_key.txt"):
        key = open("api_key.txt").read().strip()
    if not key:
        sys.exit("No API key: set ALPHAGENOME_API_KEY or create api_key.txt")
    return key


def track_table(model):
    meta = model.output_metadata(dna_client.Organism.HOMO_SAPIENS)
    parts = []
    for name in ["rna_seq", "cage", "dnase", "atac", "chip_tf", "chip_histone",
                 "splice_sites", "splice_site_usage", "splice_junctions", "procap"]:
        df = getattr(meta, name, None)
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        df["output"] = name
        parts.append(df)
    allm = pd.concat(parts, ignore_index=True)
    counts = (allm.groupby(["ontology_curie", "biosample_name", "output"]).size()
              .unstack(fill_value=0))
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values("total", ascending=False)
    os.makedirs("data", exist_ok=True)
    counts.to_csv("data/alphagenome_tracks_per_biosample.csv")
    print("\nTop 15 biosamples by number of AlphaGenome tracks:")
    print(counts.head(15).to_string())
    wanted = list(HEPATIC) + list(CONTROL)
    print("\nOur chosen biosamples:")
    print(counts[counts.index.get_level_values(0).isin(wanted)].to_string())
    missing = [c for c in wanted if c not in counts.index.get_level_values(0)]
    if missing:
        print(f"\nWARNING: not found in AlphaGenome metadata: {missing}")
    return counts


def to_variant(r):
    return genome.Variant(chromosome=r["chrom"], position=int(r["pos"]),
                          reference_bases=r["ref"], alternate_bases=r["alt"],
                          name=r["variant_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks-only", action="store_true")
    ap.add_argument("--test", type=int, default=0, help="score only N variants")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--input", default=IN_FILE, help="variant list (default: the T2D variants)")
    ap.add_argument("--chunks", default=CHUNK_DIR, help="folder for the score files")
    ap.add_argument("--expression-only", action="store_true",
                    help="score only RNA expression (faster; used for the liver eQTL benchmark)")
    a = ap.parse_args()

    model = dna_client.create(api_key())
    track_table(model)
    if a.tracks_only:
        return

    var = pd.read_csv(a.input)
    if a.test:
        var = var.head(a.test)
    os.makedirs(a.chunks, exist_ok=True)
    keys = ["RNA_SEQ", "RNA_SEQ_ACTIVE"] if a.expression_only else SCORER_KEYS
    scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS[k] for k in keys]
    keep_curies = set(HEPATIC) | set(CONTROL)
    prefix = "test_" if a.test else "chunk_"

    chunks = [var.iloc[i:i + CHUNK] for i in range(0, len(var), CHUNK)]
    print(f"\nScoring {len(var)} variants in {len(chunks)} chunks of {CHUNK}")
    for ci, ch in enumerate(chunks):
        path = f"{a.chunks}/{prefix}{ci:04d}.parquet"
        if os.path.exists(path):
            continue                          # already done -> resume
        variants = [to_variant(r) for _, r in ch.iterrows()]
        intervals = [v.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB) for v in variants]
        try:
            res = model.score_variants(intervals, variants, scorers,
                                       organism=dna_client.Organism.HOMO_SAPIENS,
                                       max_workers=a.workers)
        except Exception as e:                # keep going; failed chunk is retried next run
            print(f"  chunk {ci} failed: {e}")
            continue
        tidy = variant_scorers.tidy_scores(res)
        if tidy is None:
            continue
        # SPLICE_SITES tracks have no cell type -> keep them as "all"
        tidy = tidy[tidy["ontology_curie"].isin(keep_curies) | (tidy["output_type"].astype(str) == "SPLICE_SITES")]
        tidy = tidy[[c for c in KEEP_COLS if c in tidy.columns]].copy()
        tidy["variant_id_ag"] = tidy["variant_id"].astype(str)
        # map AlphaGenome variant string back to Open Targets id via chunk order
        name_map = {str(v): v.name for v in variants}
        tidy["variant_id"] = tidy["variant_id_ag"].map(name_map).fillna(tidy["variant_id_ag"])
        tidy["cell"] = tidy["ontology_curie"].map({**HEPATIC, **CONTROL}).fillna("all")
        tidy.to_parquet(path, index=False)
        print(f"  chunk {ci + 1}/{len(chunks)}: {len(tidy):,} rows saved")

    done = len(glob.glob(f"{a.chunks}/{prefix}*.parquet"))
    print(f"\nFinished chunks: {done}/{len(chunks)}"
          + ("" if done == len(chunks) else "  -> run again to retry the missing ones"))


if __name__ == "__main__":
    main()
