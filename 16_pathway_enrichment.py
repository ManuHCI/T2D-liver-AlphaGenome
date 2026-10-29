"""
16_pathway_enrichment.py
GO / Reactome / KEGG over-representation analysis of the hybrid Level-1 genes, using the g:Profiler REST API.

Run from the T2D_liver folder:   python 16_pathway_enrichment.py
Needs: pandas, requests, openpyxl (pip install pandas requests openpyxl)
Writes: results_enrichment/
    enrichment_all_terms.csv   every term tested in every analysis (significant or not)
    enrichment_significant.csv terms with FDR < 0.05
    enrichment_tables.xlsx     one sheet per analysis (for Supplementary Table S10)
    enrichment_summary.txt     plain-text summary: database versions, gene counts, top terms

Three analyses (all FDR = Benjamini-Hochberg, threshold 0.05):
  A  91 Level-1 genes  vs  background = all nearest protein-coding genes scored for T2D variants
     (main test: does the confident-direction subset differ from T2D-variant genes in general?)
  B  91 Level-1 genes  vs  all annotated human genes (g:Profiler default)
     (descriptive: which pathways are represented in the atlas)
  C  all nearest genes of T2D variants  vs  all annotated human genes
     (context: pathways that come from T2D genetics itself, to interpret B)
"""
import json
import os
import sys
import time

import pandas as pd
import requests

BASE = "https://biit.cs.ut.ee/gprofiler/api"
SOURCES = ["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"]
OUT = "results_enrichment"
FDR = 0.05

os.makedirs(OUT, exist_ok=True)

# ---------- gene lists ----------
lvl1 = pd.read_csv("results_hybrid/level0_genes.csv")                 # the 91 Level-1 genes
allg = pd.read_csv("results_hybrid/level1_all_nearest_genes.csv")     # nearest gene of every scored variant
lvl1_ids = sorted(lvl1["gene_id"].dropna().unique())
all_ids = sorted(allg["gene_id"].dropna().unique())
sym = dict(zip(allg["gene_id"], allg["gene_name"]))
sym.update(dict(zip(lvl1["gene_id"], lvl1["gene_name"])))
print(f"Level-1 genes: {len(lvl1_ids)}   all nearest genes (background for A): {len(all_ids)}")
missing = set(lvl1_ids) - set(all_ids)
if missing:
    print("WARNING: Level-1 genes absent from background (added to background):", missing)
    all_ids = sorted(set(all_ids) | missing)


def gost(query_ids, background=None):
    body = {
        "organism": "hsapiens",
        "query": list(query_ids),
        "sources": SOURCES,
        "user_threshold": 1.0,                 # return every tested term; we filter by FDR ourselves
        "all_results": True,
        "significance_threshold_method": "fdr",
        "no_evidences": False,
        "no_iea": False,
        "ordered": False,
        "measure_underrepresentation": False,
    }
    if background is not None:
        body["domain_scope"] = "custom"
        body["background"] = list(background)
    else:
        body["domain_scope"] = "annotated"
    for attempt in range(3):
        try:
            r = requests.post(f"{BASE}/gost/profile/", json=body, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # network hiccup: retry
            print("  request failed:", e, "- retrying" if attempt < 2 else "")
            time.sleep(5)
    sys.exit("g:Profiler could not be reached. Check the internet connection and run again.")


def tidy(res, label):
    meta = res["meta"]
    qmeta = list(meta["genes_metadata"]["query"].values())[0]
    ensgs = qmeta["ensgs"]                       # order matches each term's 'intersections' list
    rows = []
    for t in res["result"]:
        genes = [sym.get(g, g) for g, ev in zip(ensgs, t["intersections"]) if ev]
        k, n, K, N = t["intersection_size"], t["query_size"], t["term_size"], t["effective_domain_size"]
        rows.append({
            "analysis": label,
            "source": t["source"],
            "term_id": t["native"],
            "term_name": t["name"],
            "FDR": t["p_value"],                 # already BH-adjusted because method = fdr
            "significant": t["p_value"] < FDR,
            "genes_in_term_and_list": k,
            "list_size_annotated": n,
            "term_size": K,
            "background_size": N,
            "enrichment_ratio": round((k / n) / (K / N), 2) if n and K else None,
            "genes": ";".join(sorted(genes)),
        })
    df = pd.DataFrame(rows).sort_values(["FDR", "source"]).reset_index(drop=True)
    mapped = sum(1 for g in ensgs if g)
    return df, meta.get("version", "unknown"), len(ensgs), mapped


analyses = [
    ("A_level1_vs_T2D_nearest_genes", lvl1_ids, all_ids),
    ("B_level1_vs_genome", lvl1_ids, None),
    ("C_T2D_nearest_genes_vs_genome", all_ids, None),
]
tables, lines = {}, []
try:
    versions = requests.get(f"{BASE}/util/data_versions", timeout=60).json()
except Exception:
    versions = {}

for label, q, bg in analyses:
    print("Running", label, "...")
    res = gost(q, bg)
    df, ver, nq, nm = tidy(res, label)
    tables[label] = df
    sig = df[df.significant]
    lines.append(f"\n=== {label} ===")
    lines.append(f"query genes {nq}; background {'custom, ' + str(len(bg)) + ' genes' if bg else 'all annotated genes'}")
    lines.append(f"terms tested {len(df)}; significant at FDR < {FDR}: {len(sig)}")
    for src in SOURCES:
        lines.append(f"  {src}: {int((sig.source == src).sum())} significant")
    show = sig if len(sig) else df.head(10)
    lines.append("top terms (significant)" if len(sig) else "no significant term; 10 lowest FDR shown")
    for _, r in show.head(15).iterrows():
        lines.append(f"  {r.source:6s} {r.term_id:14s} {r.term_name[:60]:60s} FDR={r.FDR:.2e} "
                     f"k={r.genes_in_term_and_list}/{r.list_size_annotated} ER={r.enrichment_ratio} [{r.genes[:80]}]")

allres = pd.concat(tables.values(), ignore_index=True)
allres.to_csv(f"{OUT}/enrichment_all_terms.csv", index=False)
allres[allres.significant].to_csv(f"{OUT}/enrichment_significant.csv", index=False)
with pd.ExcelWriter(f"{OUT}/enrichment_tables.xlsx") as xw:
    for label, df in tables.items():
        df.to_excel(xw, sheet_name=label[:31], index=False)

head = ["g:Profiler over-representation analysis (hypergeometric test, Benjamini-Hochberg FDR)",
        f"g:Profiler data version: {ver}",
        f"sources: {', '.join(SOURCES)}",
        f"database versions: {json.dumps(versions)[:1500]}",
        f"run date: {time.strftime('%Y-%m-%d')}"]
txt = "\n".join(head + lines)
open(f"{OUT}/enrichment_summary.txt", "w", encoding="utf-8").write(txt)
print(txt)
print(f"\nDone. Files written to {OUT}/")
