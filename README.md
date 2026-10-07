# T2D-liver-AlphaGenome

Variant-to-function analysis of type 2 diabetes (T2D) risk variants in human hepatocytes with the
AlphaGenome sequence-to-function model, including a liver eQTL benchmark, a hybrid gene-assignment rule,
an atlas of 108 variant–gene pairs, an interactive regulatory tree, and a case study of an independent
PROX1 signal (rs17712208) in an HNF1A-bound liver enhancer.

**Interactive tree:** https://manuhci.github.io/T2D-liver-AlphaGenome/  (served from `docs/index.html`)

Author: Manu Kumar Shetty, Department of Pharmacology, Maulana Azad Medical College, New Delhi
(ORCID 0000-0002-9767-5092). Code: MIT. Derived data: CC BY 4.0.

---

## What is in this repository

| Folder / file | Content |
|---|---|
| `01`–`15` `*.py` | The complete pipeline (Python 3.11). Run in order; every step reads the previous step's files. |
| `data/` | Inputs and AlphaGenome scores (`data/scores_chunks/`, `data/scores_eqtl_benchmark/`, `data/prox1/*.parquet`). Large public downloads are not stored; the scripts fetch them. |
| `results/` | Level 0 tables, liver eQTL benchmark, first-version tree, literature check |
| `results_hybrid/` | Hybrid Level 1 (108 variants → 91 genes), network tree, validation tests |
| `results_prox1/` | PROX1 locus tests, ENCODE ChIP-seq check, colocalisation |
| `figures/` | Main figures 1–6 (300 dpi) |
| `supplementary/` | Supplementary Tables S1–S9 (Excel) |
| `docs/` | Interactive tree (`index.html`) and its data (`tree_data.json`) |
| `tree_template.html` | Page template used by step 13 |

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `01_collect_t2d_variants.py` | Open Targets: SuSiE credible sets of two T2D GWAS (GCST90475667, FINNGEN_R12_T2D); PIP ≥ 0.2; risk allele orientation |
| 2 | `02_score_alphagenome_hepatic.py` | AlphaGenome variant scoring (1 Mb, HepG2 / liver / hepatocyte; K562 control) |
| 3 | `03_summarise_level0.py` | Level 0: gene, TF-binding and enhancer-level effects |
| 4 | `04_build_tree.py` | Network propagation over CollecTRI + OmniPath (liver-expressed genes), matched empirical null, BH FDR |
| 5 | `05_control_comparison.py` | Validation against diabetic-liver expression (GSE23343, GSE15653) and Open Targets gene sets |
| 6 | `06_liver_eqtl_comparison.py` | Overlap of T2D variants with GTEx liver eQTL credible sets |
| 7 | `07_liver_eqtl_benchmark.py` | Benchmark on 328 causal GTEx liver eQTLs: direction and target-gene choice |
| 7b | `07b_benchmark_assignment_rules.py` | Gene AND direction accuracy of three assignment rules |
| 8 | `08_hybrid_level1.py` | Hybrid Level 1: nearest protein-coding gene + confident AlphaGenome direction |
| 9 | `09_genetic_direction_test.py` | FinnGen Wald-ratio (SMR-style) direction test of downstream genes |
| 10 | `10_prox1_case_study.py` | PROX1 locus: LD, tissue specificity, GTEx eQTL, in-silico mutagenesis, motifs, PheWAS |
| 11 | `11_encode_chip_check.py` | ENCODE ChIP-seq peaks and cCREs over rs17712208 |
| 12 | `12_colocalization.py` | Colocalisation of the T2D signal with metabolite / glycaemic traits |
| 13 | `13_build_tree_page.py` | Builds the interactive tree (`docs/index.html`) |
| 14 | `14_make_figures.py` | Figures 1–6 |
| 15 | `15_supplementary_tables.py` | Supplementary Tables S1–S9 |

AlphaGenome steps (2, 7, 10) need an API key: set `ALPHAGENOME_API_KEY` or put the key in `api_key.txt`
(this file is excluded by `.gitignore`). All other steps run from the stored scores.

```
pip install -r requirements.txt
python 01_collect_t2d_variants.py --study GCST90475667 --study FINNGEN_R12_T2D
python 02_score_alphagenome_hepatic.py
python 03_summarise_level0.py
python 08_hybrid_level1.py
python 04_build_tree.py --res results_hybrid
python 05_control_comparison.py --fetch
python 05_control_comparison.py --run --res results_hybrid
python 07_liver_eqtl_benchmark.py --prepare
python 02_score_alphagenome_hepatic.py --input data/liver_eqtl_benchmark_variants.csv --chunks data/scores_eqtl_benchmark --expression-only
python 07_liver_eqtl_benchmark.py --analyse
python 07b_benchmark_assignment_rules.py
python 09_genetic_direction_test.py --prepare
python 09_genetic_direction_test.py --run --res results_hybrid
python 10_prox1_case_study.py
python 11_encode_chip_check.py
python 12_colocalization.py
python 13_build_tree_page.py
python 14_make_figures.py
python 15_supplementary_tables.py
```

## Key results

* **Benchmark (328 causal GTEx liver eQTLs).** AlphaGenome predicted the direction of effect for 68% of
  eQTLs, 82.5% when confident (|quantile| ≥ 0.99) and 95% for the largest 25% of effects (control 51%).
  The nearest protein-coding gene named the true eGene more often than AlphaGenome's top gene (62.5% vs 42.1%).
* **Hybrid rule.** Nearest gene + confident AlphaGenome direction got gene *and* direction right for 66.7%
  of the variants it kept (159/328).
* **Atlas.** 108 T2D variants → 91 liver genes (45 up, 46 down, 14 transcription factors).
* **Network tree (Levels 2+).** Not validated by three independent yardsticks; reported as hypothesis-generating only.
* **PROX1.** rs17712208 is independent of the known rs340874 signal (r² < 0.05), lies in an ENCODE distal
  enhancer bound by HNF1A (2/2 HepG2 experiments, summits 13–31 bp away), is predicted to close it and lower
  PROX1, and colocalises with plasma alanine, glutamine and branched-chain amino acids (COLOC H4 ≈ 0.999).

## Data sources

Open Targets Platform (GraphQL API); FinnGen R12; GCST90475667 (Verma et al., 2024); GTEx v8 / eQTL Catalogue
(QTD000266); GENCODE v46; CollecTRI and OmniPath; GEO GSE23343 and GSE15653; ENCODE (portal and UCSC
`encRegTfbsClustered`, `encodeCcreCombined`); JASPAR 2024; Ensembl REST; GWAS Catalog; AlphaGenome API.

## Publishing the interactive tree (GitHub Pages)

Repository → Settings → Pages → Source: *Deploy from a branch* → Branch `main`, folder `/docs` → Save.
After about a minute the tree is live at `https://<user>.github.io/T2D-liver-AlphaGenome/`.
