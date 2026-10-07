"""
Step 10 - PROX1 CASE STUDY: does rs17712208 act through a LIVER enhancer?

Already known (literature)
  - PROX1 is a T2D / fasting-glucose locus (Dupuis 2010).
  - The studied variant rs340874 lowers a reporter signal; the authors proposed a BETA-CELL mechanism (Lecompte 2013).
  - Liver Prox1 knockout -> insulin resistance; PROX1 represses LRH-1 (NR5A2) -> CYP7A1 (bile acids).
Not known (what we test)
  - rs17712208 (larger effect, zero PubMed papers): AlphaGenome predicts that the risk allele closes a liver
    enhancer and lowers PROX1. Is this a real, separate, liver-acting signal?

Five tests - each one can FAIL, and the final summary says which passed.
  1  Independent signal?   LD between rs17712208 and rs340874 (and the other locus variants), 1000 Genomes EUR
  2  Liver or pancreas?    AlphaGenome chromatin/expression change in ALL tissues; where does liver rank?
  3  Real human tissue?    GTEx eQTL (Liver, Pancreas): does the risk allele lower PROX1?
  4  Which TF site breaks? AlphaGenome in-silico mutagenesis (41 bp) + JASPAR motif scan + AlphaGenome ChIP-TF drops
  5  Which phenotype?      Open Targets + GWAS Catalog: insulin-resistance / lipid / liver traits (liver story)
                           versus insulin-secretion traits (beta-cell story)

Run (from E:\\Drug_Design\\Alpha_Chain\\T2D_liver; API key as in step 2)
  python 10_prox1_case_study.py                 # all five tests
  python 10_prox1_case_study.py --steps 1 3 5   # only the tests that need no AlphaGenome key
Results: results_prox1/  (CSV files + prox1_case_study_summary.txt). Downloads are cached in data/prox1/.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

DATA, OUT = "data/prox1", "results_prox1"
TARGET, KNOWN = "rs17712208", "rs340874"
PROX1_ENSG = "ENSG00000117707"
LOCUS = ("chr1", 213_800_000, 214_300_000)
HEPATIC = {"EFO:0001187": "HepG2", "UBERON:0002107": "liver", "CL:0000182": "hepatocyte"}
LIVER_RX = r"liver|hepat|HepG2"
PANC_RX = r"pancrea"                                   # whole / exocrine pancreas
ISLET_RX = r"islet|endocrine pancreas|type B pancreatic|beta cell"   # endocrine (beta-cell) samples
ENSEMBL = "https://rest.ensembl.org"
GTEX = "https://gtexportal.org/api/v2"
OT = "https://api.platform.opentargets.org/api/v4/graphql"
GWASCAT = "https://www.ebi.ac.uk/gwas/rest/api"
JASPAR_URLS = [
    "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
    "https://jaspar.genereg.net/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
]
SUMMARY = []          # one line per test verdict


def say(line=""):
    print(line)
    SUMMARY.append(line)


# ------------------------------------------------------------------ helpers
def get_json(url, params=None, retries=3, method="GET", body=None):
    """GET/POST with retries and a JSON cache on disk."""
    raw = url + json.dumps(params or {}, sort_keys=True) + json.dumps(body or {}, sort_keys=True)
    key = re.sub(r"[^A-Za-z0-9]+", "_", url.split("//")[-1])[:60] + "_" + hashlib.md5(raw.encode()).hexdigest()[:12]
    cache = f"{DATA}/cache_{key}.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    for i in range(retries):
        try:
            if method == "POST":
                r = requests.post(url, json=body, timeout=60)
            else:
                r = requests.get(url, params=params, timeout=60, headers={"Accept": "application/json"})
            if r.status_code == 429:
                time.sleep(3 * (i + 1))
                continue
            if r.status_code >= 400:
                print(f"    HTTP {r.status_code} from {url}: {r.text[:200]}")
                return None
            d = r.json()
            if not (isinstance(d, dict) and "errors" in d):        # never cache an error answer
                json.dump(d, open(cache, "w"))
            return d
        except Exception as e:
            print(f"    attempt {i + 1} failed ({e})")
            time.sleep(2)
    return None


def load_locus():
    v = pd.read_csv("data/t2d_variants_for_alphagenome.csv")
    v = v[(v["chrom"] == LOCUS[0]) & v["pos"].between(LOCUS[1], LOCUS[2])]
    v = v.sort_values("pip", ascending=False).drop_duplicates("variant_id")
    if TARGET not in set(v["rsid"]):
        sys.exit(f"{TARGET} not in data/t2d_variants_for_alphagenome.csv")
    print("PROX1-locus T2D variants (fine-mapped):")
    print(v[["rsid", "variant_id", "pip", "beta_alt", "risk_allele", "studies_supporting"]].to_string(index=False))
    return v


def api_key():
    key = os.environ.get("ALPHAGENOME_API_KEY")
    if not key and os.path.exists("api_key.txt"):
        key = open("api_key.txt").read().strip()
    if not key:
        sys.exit("No API key: set ALPHAGENOME_API_KEY or create api_key.txt (needed for tests 2 and 4)")
    return key


# ------------------------------------------------------------------ test 1: LD
def test1_ld(loc):
    say("\n=== TEST 1  Is rs17712208 a separate signal from the studied rs340874? (LD, 1000G EUR) ===")
    rows = []
    for rs in loc["rsid"]:
        if rs == TARGET:
            continue
        d = get_json(f"{ENSEMBL}/ld/human/pairwise/{TARGET}/{rs}",
                     {"population_name": "1000GENOMES:phase_3:EUR"})
        r2 = dp = np.nan
        if d:
            d = d[0] if isinstance(d, list) and d else d
            r2, dp = float(d.get("r2", np.nan)), float(d.get("d_prime", np.nan))
        rows.append({"variant": rs, "r2_with_rs17712208": r2, "d_prime": dp})
    freq = []
    for rs in loc["rsid"]:
        d = get_json(f"{ENSEMBL}/variation/human/{rs}", {"pops": 1})
        f = {}
        for p in (d or {}).get("populations", []):
            if p["population"] in ("1000GENOMES:phase_3:EUR", "1000GENOMES:phase_3:SAS", "gnomADg:nfe"):
                f[f"{p['population'].split(':')[-1]}_{p['allele']}"] = p["frequency"]
        freq.append({"variant": rs, **f})
    ld = pd.DataFrame(rows)
    fr = pd.DataFrame(freq)
    ld.to_csv(f"{OUT}/test1_ld.csv", index=False)
    fr.to_csv(f"{OUT}/test1_allele_frequencies.csv", index=False)
    print(ld.to_string(index=False))
    print("\nAllele frequencies:\n" + fr.to_string(index=False))
    known = ld.loc[ld["variant"] == KNOWN, "r2_with_rs17712208"]
    # Ensembl returns a pair only when r2 >= 0.05, so an empty answer means r2 < 0.05.
    # Independent check: the maximum r2 two SNPs can have is fixed by their allele frequencies.
    def maf(rs):
        r = fr[fr["variant"] == rs].filter(like="EUR_")
        return float(r.min(axis=1).iloc[0]) if len(r) and r.notna().any(axis=None) else np.nan
    p1, p2 = sorted([maf(TARGET), maf(KNOWN)])
    r2max = (p1 * (1 - p2)) / ((1 - p1) * p2) if p1 > 0 and p2 > 0 else np.nan
    say(f"  allele-frequency ceiling: r2 between {TARGET} (MAF {p1:.3f}) and {KNOWN} (MAF {p2:.3f}) "
        f"cannot exceed {r2max:.3f}")
    proxies = ld.loc[ld["r2_with_rs17712208"] >= 0.8, "variant"].tolist()
    if proxies:
        say(f"  perfect/strong proxies of {TARGET} (r2 >= 0.8): {proxies} -> genetics alone cannot separate them")
    if (known.empty or known.isna().all()) and r2max < 0.2:
        say(f"  Test 1: PASS - Ensembl reports no LD (r2 < 0.05) and the ceiling is {r2max:.3f} -> separate signal")
    elif known.empty or known.isna().all():
        say("  Test 1: NOT TESTED (Ensembl LD not returned)")
    elif known.iloc[0] < 0.2:
        say(f"  Test 1: PASS - r2 with {KNOWN} = {known.iloc[0]:.3f} -> separate signal, not a proxy")
    else:
        say(f"  Test 1: FAIL - r2 with {KNOWN} = {known.iloc[0]:.3f} -> may be the same signal")


# ------------------------------------------------------------------ test 2: tissue
def test2_tissue(loc):
    say("\n=== TEST 2  Liver or pancreas? AlphaGenome effect in every tissue ===")
    from alphagenome.data import genome
    from alphagenome.models import dna_client, variant_scorers

    path = f"{DATA}/tissue_scores.parquet"
    if not os.path.exists(path):
        model = dna_client.create(api_key())
        keys = ["DNASE", "ATAC", "CHIP_HISTONE", "RNA_SEQ"]
        scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS[k] for k in keys]
        variants = [genome.Variant(chromosome=r.chrom, position=int(r.pos), reference_bases=r.ref,
                                   alternate_bases=r.alt, name=r.rsid) for r in loc.itertuples()]
        intervals = [v.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB) for v in variants]
        print(f"  scoring {len(variants)} variants in all tissues (a few minutes)...")
        res = model.score_variants(intervals, variants, scorers,
                                   organism=dna_client.Organism.HOMO_SAPIENS, max_workers=4)
        t = variant_scorers.tidy_scores(res)
        name = {str(v): v.name for v in variants}
        t["rsid"] = t["variant_id"].astype(str).map(name)
        keep = ["rsid", "output_type", "variant_scorer", "ontology_curie", "biosample_name", "biosample_type",
                "histone_mark", "gene_name", "raw_score", "quantile_score"]
        t = t[[c for c in keep if c in t.columns]].copy()
        for c in t.columns:
            if t[c].dtype == object or str(t[c].dtype) == "category":
                t[c] = t[c].astype(str)
        t.to_parquet(path, index=False)
    t = pd.read_parquet(path)
    t = t.merge(loc[["rsid", "risk_sign"]], on="rsid")
    t["diabetic"] = t["raw_score"] * t["risk_sign"]
    bn = t["biosample_name"].astype(str)
    is_liver = (t["ontology_curie"].isin(HEPATIC) | bn.str.contains(LIVER_RX, case=False)) & \
        ~bn.str.contains("stellate|endothel|kupffer", case=False)
    is_islet = bn.str.contains(ISLET_RX, case=False)
    is_panc = bn.str.contains(PANC_RX, case=False) & ~is_islet
    t["group"] = np.select([is_liver, is_islet, is_panc], ["LIVER", "ISLET", "PANCREAS"], "other")
    for grp in ("LIVER", "PANCREAS", "ISLET"):
        print(f"  {grp} biosamples:", sorted(t.loc[t["group"] == grp, "biosample_name"].unique()) or "none")

    rows = []
    for (rs, out), d in t.groupby(["rsid", "output_type"]):
        if out == "RNA_SEQ":
            d = d[d["gene_name"] == "PROX1"]
            if d.empty:
                continue
        if out == "CHIP_HISTONE":
            d = d[d["histone_mark"].isin(["H3K27ac", "H3K4me1"])]       # active-enhancer marks
        per = d.groupby(["biosample_name", "group"])["diabetic"].mean().reset_index()
        per["rank_pct"] = per["diabetic"].rank(pct=True)                 # low = strongest decrease
        g = per.groupby("group")["diabetic"].mean()
        liver_rank = per.loc[per["group"] == "LIVER", "rank_pct"].mean()
        rows.append({"rsid": rs, "output": out, "n_biosamples": len(per),
                     "liver_mean": g.get("LIVER", np.nan), "islet_mean": g.get("ISLET", np.nan),
                     "pancreas_mean": g.get("PANCREAS", np.nan),
                     "other_mean": g.get("other", np.nan), "liver_rank_pct": liver_rank})
        if rs == TARGET and out == "DNASE":
            per.sort_values("diabetic").to_csv(f"{OUT}/test2_rs17712208_dnase_by_biosample.csv", index=False)
            print("\n  rs17712208 DNase change, 10 strongest-closing biosamples:")
            print(per.sort_values("diabetic").head(10).to_string(index=False))
            print("  liver / pancreas / islet biosamples:")
            print(per[per["group"] != "other"].sort_values("diabetic").to_string(index=False))
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/test2_tissue_specificity.csv", index=False)
    print("\n  Mean diabetic-minus-normal change (liver_rank_pct near 0 = liver among the strongest decreases):")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    tpm = pd.read_csv(gzip.open("data/gtex_median_tpm.gct.gz", "rt"), sep="\t", skiprows=2)
    p = tpm[tpm["Description"] == "PROX1"]
    if len(p):
        print(f"\n  PROX1 expression (GTEx median TPM): liver {p['Liver'].max():.1f}, "
              f"pancreas {p['Pancreas'].max():.1f}")

    r = tab[(tab["rsid"] == TARGET) & (tab["output"] == "DNASE")]
    if r.empty:
        say("  Test 2: NOT TESTED")
        return
    # liver (hepatocyte) versus BETA CELL / islet, judged on every output that has islet samples
    tt = tab[(tab["rsid"] == TARGET)].dropna(subset=["liver_mean", "islet_mean"])
    calls = []
    for x in tt.itertuples():
        if x.liver_mean < 0 and abs(x.liver_mean) > 2 * abs(x.islet_mean):
            calls.append((x.output, "liver"))
        elif x.islet_mean < 0 and abs(x.islet_mean) > 2 * abs(x.liver_mean):
            calls.append((x.output, "islet"))
        else:
            calls.append((x.output, "both"))
    say("  per output (liver vs islet/endocrine): " + ", ".join(
        f"{o} {c} ({lv:.2f} vs {il:.2f})" for (o, c), lv, il in zip(calls, tt["liver_mean"], tt["islet_mean"])))
    n_liver = sum(c == "liver" for _, c in calls)
    if calls and n_liver > len(calls) / 2:
        verdict = "PASS - effect is liver-dominant in most outputs"
    elif calls and sum(c == "islet" for _, c in calls) > len(calls) / 2:
        verdict = "FAIL - effect is islet-dominant"
    else:
        verdict = "SHARED - the enhancer closes in liver AND in endocrine pancreas (beta-cell route not excluded)"
    r = r.iloc[0]
    say(f"  Test 2: {verdict}; DNase liver rank among {int(r['n_biosamples'])} biosamples = top "
        f"{100 * r['liver_rank_pct']:.0f}%")
    k = tab[(tab["rsid"] == KNOWN) & (tab["output"] == "DNASE")]
    if len(k):
        say(f"          for comparison {KNOWN}: DNase liver {k.iloc[0]['liver_mean']:.3f}, "
            f"islet/endocrine {k.iloc[0]['islet_mean']:.3f}")
    print("  NOTE: whole pancreas is mostly exocrine; ISLET = endocrine / beta-cell samples.")


# ------------------------------------------------------------------ test 3: eQTL
def test3_eqtl(loc):
    say("\n=== TEST 3  Does the risk allele lower PROX1 in real human tissue? (GTEx v8) ===")
    g = get_json(f"{GTEX}/reference/gene", {"geneId": "PROX1", "gencodeVersion": "v26",
                                            "genomeBuild": "GRCh38/hg38"})
    gencode = g["data"][0]["gencodeId"] if g and g.get("data") else PROX1_ENSG + ".11"
    rows = []
    for r in loc.itertuples():
        vid = f"{r.chrom}_{r.pos}_{r.ref}_{r.alt}_b38"
        for tissue in ("Liver", "Pancreas"):
            d = get_json(f"{GTEX}/association/dyneqtl", {"gencodeId": gencode, "variantId": vid,
                                                         "tissueSiteDetailId": tissue, "datasetId": "gtex_v8"})
            if not d:
                rows.append({"rsid": r.rsid, "tissue": tissue, "note": "not in GTEx / no result"})
                continue
            nes = d.get("nes")
            rows.append({"rsid": r.rsid, "tissue": tissue, "nes_alt": nes, "p": d.get("pValue"),
                         "maf": d.get("maf"), "risk_allele": r.risk_allele, "alt": r.alt,
                         "nes_risk": (nes * r.risk_sign) if nes is not None else None})
    e = pd.DataFrame(rows)
    e.to_csv(f"{OUT}/test3_gtex_eqtl.csv", index=False)
    print(e.to_string(index=False))
    t = e[(e["rsid"] == TARGET) & (e["tissue"] == "Liver")]
    if t.empty or "nes_risk" not in t or t["nes_risk"].isna().all():
        say("  Test 3: NOT TESTED (rs17712208 not available in GTEx liver - low frequency / not genotyped)")
    else:
        t = t.iloc[0]
        if t["nes_risk"] < 0 and t["p"] < 0.05:
            say(f"  Test 3: PASS - risk allele lowers PROX1 in liver (NES {t['nes_risk']:.2f}, p {t['p']:.2g})")
        elif t["nes_risk"] < 0:
            say(f"  Test 3: SAME DIRECTION, NOT SIGNIFICANT (NES {t['nes_risk']:.2f}, p {t['p']:.2g}; "
                f"GTEx liver n~208, low power for a low-frequency variant)")
        else:
            say(f"  Test 3: FAIL - opposite direction (NES {t['nes_risk']:.2f}, p {t['p']:.2g})")


# ------------------------------------------------------------------ test 4: mechanism
def read_jaspar():
    path = f"{DATA}/jaspar2024_core_vertebrates.txt"
    if not os.path.exists(path):
        for u in JASPAR_URLS:
            try:
                r = requests.get(u, timeout=120)
                if r.ok and r.text.startswith(">"):
                    open(path, "w").write(r.text)
                    break
            except Exception as e:
                print(f"    JASPAR download failed from {u}: {e}")
    if not os.path.exists(path):
        return {}
    motifs, name, mat = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name:
                motifs[name] = np.array(mat)
            parts = line[1:].split()
            name, mat = f"{parts[1] if len(parts) > 1 else parts[0]} ({parts[0]})", []
        elif line:
            mat.append([float(x) for x in re.findall(r"[\d.]+", line.split("[", 1)[-1])])
    if name:
        motifs[name] = np.array(mat)
    pwm = {}
    for n, m in motifs.items():                      # rows A,C,G,T -> log-odds, positions x 4
        p = (m + 0.8 * 0.25) / (m.sum(axis=0) + 0.8)
        pwm[n] = np.log2(p / 0.25).T
    return pwm


def best_hit(pwm, seq, must_cover):
    """best relative score (0-1) over windows that cover position `must_cover`, both strands."""
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    L = len(pwm)
    lo, hi = pwm.min(axis=1).sum(), pwm.max(axis=1).sum()
    rc = pwm[::-1, ::-1]
    best = -np.inf
    for s in range(max(0, must_cover - L + 1), min(len(seq) - L, must_cover) + 1):
        w = seq[s:s + L]
        if any(c not in idx for c in w):
            continue
        cols = [idx[c] for c in w]
        sc = max(pwm[np.arange(L), cols].sum(), rc[np.arange(L), cols].sum())
        best = max(best, sc)
    return (best - lo) / (hi - lo) if np.isfinite(best) else np.nan


def test4_mechanism(loc):
    say("\n=== TEST 4  Which transcription-factor site does rs17712208 break? ===")
    v = loc[loc["rsid"] == TARGET].iloc[0]
    pos, ref, alt = int(v["pos"]), v["ref"], v["alt"]
    risk = v["risk_allele"]
    flank = 30
    d = get_json(f"{ENSEMBL}/sequence/region/human/1:{pos - flank}..{pos + flank}:1",
                 {"content-type": "application/json"})
    seq = (d or {}).get("seq", "").upper()
    if len(seq) != 2 * flank + 1 or seq[flank] != ref:
        say(f"  could not fetch reference sequence correctly (got '{seq[flank:flank + 1]}', expected {ref})")
        return
    seq_alt = seq[:flank] + alt + seq[flank + 1:]
    print(f"  reference : {seq[:flank].lower()}[{ref}]{seq[flank + 1:].lower()}")
    print(f"  alternate : {seq[:flank].lower()}[{alt}]{seq[flank + 1:].lower()}   (risk allele = {risk})")

    # 4a  AlphaGenome in-silico mutagenesis: which bases around the variant hold the enhancer open?
    ism_path = f"{DATA}/ism_dnase_rs17712208.parquet"
    if not os.path.exists(ism_path):
        try:
            from alphagenome.data import genome
            from alphagenome.models import dna_client, variant_scorers
            model = dna_client.create(api_key())
            var = genome.Variant(chromosome="chr1", position=pos, reference_bases=ref, alternate_bases=alt)
            interval = var.reference_interval.resize(dna_client.SEQUENCE_LENGTH_100KB)
            ism_iv = genome.Interval("chr1", pos - 21, pos + 20)          # 41 bp, variant in the centre
            print("  AlphaGenome in-silico mutagenesis: 41 bp x 3 bases = 123 mutations ...")
            res = model.score_ism_variants(interval, ism_iv, [variant_scorers.RECOMMENDED_VARIANT_SCORERS["DNASE"]],
                                           organism=dna_client.Organism.HOMO_SAPIENS)
            rows = []                      # ISM scores carry no scorer label -> read the matrices directly
            for group in res:
                for ad in (group if isinstance(group, (list, tuple)) else [group]):
                    mask = ad.var["ontology_curie"].astype(str).isin(list(HEPATIC)).values
                    if mask.any():
                        rows.append({"variant_id": str(ad.uns["variant"]),
                                     "raw_score": float(np.asarray(ad.X)[:, mask].mean())})
            t = pd.DataFrame(rows).groupby("variant_id")["raw_score"].mean().reset_index()
            print(f"  ISM: {len(t)} mutations scored")
            t.to_parquet(ism_path, index=False)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  ISM failed: {e}")
    ism_pos = None
    if os.path.exists(ism_path):
        t = pd.read_parquet(ism_path)
        p = t["variant_id"].str.extract(r":(\d+):(\w)>(\w)")
        t["pos"], t["alt"] = p[0].astype(int), p[2]
        ism_pos = t.groupby("pos")["raw_score"].mean()                 # mean of the 3 possible mutations
        ism_pos.rename("mean_dnase_change_if_mutated").to_csv(f"{OUT}/test4_ism_by_position.csv")
        z = (ism_pos - ism_pos.median()) / (ism_pos.std() + 1e-9)
        core = sorted(ism_pos[z <= -1.5].index)
        bases = "".join(seq[flank + (q - pos)].upper() if q in core else seq[flank + (q - pos)].lower()
                        for q in range(pos - 20, pos + 21))
        print(f"  ISM - bases whose mutation closes the enhancer (CAPITALS, z <= -1.5):\n    {bases}")
        own = t[(t["pos"] == pos) & (t["alt"] == alt)]["raw_score"]
        print(f"  ISM effect of the real variant {ref}>{alt}: {own.mean():.3f} "
              f"(rank among 123 mutations: {int((t['raw_score'] < own.mean()).sum()) + 1})")
        say(f"  ISM: critical bases (z <= -1.5) at offsets {[q - pos for q in core]} from the variant")

    # 4b  JASPAR: which motifs lose most when ref -> risk allele?
    pwm = read_jaspar()
    hits = pd.DataFrame()
    if pwm:
        rows = []
        for n, m in pwm.items():
            rs, as_ = best_hit(m, seq, flank), best_hit(m, seq_alt, flank)
            rows.append({"motif": n, "ref_score": rs, "alt_score": as_, "loss": rs - as_})
        hits = pd.DataFrame(rows).dropna()
        hits = hits[hits["ref_score"] >= 0.80].sort_values("loss", ascending=False)
        hits.to_csv(f"{OUT}/test4_motif_changes.csv", index=False)
        print("\n  Motifs that match the normal sequence (score >= 0.80) and lose most with the risk allele:")
        print(hits.head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 4c  AlphaGenome ChIP-TF tracks (already scored in step 2): which TFs drop most?
    import glob
    chip = []
    for f in sorted(glob.glob("data/scores_chunks/chunk_*.parquet")):
        c = pd.read_parquet(f, columns=["variant_id", "output_type", "transcription_factor", "cell", "raw_score"])
        chip.append(c[(c["variant_id"] == v["variant_id"]) & (c["output_type"] == "CHIP_TF")
                      & c["cell"].isin(list(HEPATIC.values()))])
    chip = pd.concat(chip) if chip else pd.DataFrame()
    if len(chip):
        chip["diabetic"] = chip["raw_score"] * v["risk_sign"]
        tf = chip.groupby("transcription_factor")["diabetic"].mean().sort_values()
        tf.rename("mean_change").to_csv(f"{OUT}/test4_alphagenome_chip_tf.csv")
        print(f"\n  AlphaGenome ChIP-TF: {int((tf < 0).sum())} of {len(tf)} TFs predicted to lose binding. Top 15:")
        print(tf.head(15).to_string())
        if len(hits):
            top_tf = set(tf.head(40).index.str.upper())
            agree = [m for m in hits.head(20)["motif"] if m.split(" (")[0].upper().split("::")[0] in top_tf]
            say(f"  Motif + ChIP agreement (motif loss in top 20 AND TF in top 40 ChIP drops): {agree or 'none'}")
    if len(hits):
        say(f"  Test 4: top motif losses -> {', '.join(hits.head(5)['motif'])}")
    say("  Test 4 is descriptive: it names candidate TFs for the wet-lab test, it does not pass/fail.")


# ------------------------------------------------------------------ test 5: phenotype pattern
TRAIT_GROUPS = [
    ("beta-cell / insulin secretion", r"HOMA-?B|insulin secretion|proinsulin|disposition|insulinogenic|acute insulin"),
    ("insulin resistance", r"HOMA-?IR|insulin resistance|insulin sensitivity|fasting insulin|insulin level"),
    ("glycaemia / diabetes", r"glucose|HbA1c|glycated|hemoglobin a1c|diabetes"),
    ("lipids / bile", r"cholesterol|LDL|HDL|triglycer|apolipoprotein|lipoprotein|bile|gallstone|cholelith"),
    ("liver", r"alanine aminotransferase|aspartate aminotransferase|\bALT\b|\bAST\b|gamma.?glutamyl|GGT|"
              r"fatty liver|NAFLD|MASLD|bilirubin|alkaline phosphatase|liver"),
    ("adiposity", r"body mass|BMI|waist|fat|obes|adipos"),
]


def trait_group(t):
    for g, rx in TRAIT_GROUPS:
        if re.search(rx, str(t), re.I):
            return g
    return "other"


Q_VAR = """
query($id:String!, $index:Int!){
  variant(variantId:$id){
    id rsIds
    credibleSets(page:{index:$index, size:100}){
      count
      rows{
        studyLocusId
        study{ id studyType traitFromSource projectId target{ approvedSymbol } biosample{ biosampleName } }
        locus(variantIds:[$id]){ rows{ beta posteriorProbability pValueMantissa pValueExponent } }
      }
    }
  }
}"""
Q_VAR_SIMPLE = Q_VAR.replace(" target{ approvedSymbol } biosample{ biosampleName }", "")
Q_VAR_MIN = Q_VAR_SIMPLE.replace(
    "\n        locus(variantIds:[$id]){ rows{ beta posteriorProbability pValueMantissa pValueExponent } }", "")


def ot_credible_sets(vid):
    out, index = [], 0
    for q in (Q_VAR, Q_VAR_SIMPLE, Q_VAR_MIN):
        out, index, ok = [], 0, True
        while True:
            d = get_json(OT, method="POST", body={"query": q, "variables": {"id": vid, "index": index}})
            if not d or "errors" in d or not d.get("data", {}).get("variant"):
                ok = False
                if d and "errors" in d:
                    print(f"    Open Targets: {d['errors'][0].get('message', '')[:150]}")
                break
            cs = d["data"]["variant"]["credibleSets"]
            for r in cs["rows"]:
                s = r["study"] or {}
                loc = (r.get("locus") or {}).get("rows") or [{}]
                l = loc[0]
                out.append({"study": s.get("id"), "type": s.get("studyType"), "trait": s.get("traitFromSource"),
                            "project": s.get("projectId"),
                            "qtl_gene": (s.get("target") or {}).get("approvedSymbol"),
                            "qtl_tissue": (s.get("biosample") or {}).get("biosampleName"),
                            "beta_alt": l.get("beta"), "pip": l.get("posteriorProbability"),
                            "p": (l["pValueMantissa"] * 10.0 ** l["pValueExponent"])
                            if l.get("pValueMantissa") is not None else None})
            index += 1
            if index * 100 >= cs["count"]:
                break
        if ok:
            return out
    return out


def gwas_catalog(rs):
    d = get_json(f"{GWASCAT}/singleNucleotidePolymorphisms/{rs}/associations",
                 {"projection": "associationBySnp"})
    rows = []
    for a in ((d or {}).get("_embedded") or {}).get("associations", []):
        traits = "; ".join(t.get("trait", "") for t in a.get("efoTraits", []))
        risk = "; ".join(ra.get("riskAlleleName", "") for l in a.get("loci", [])
                         for ra in l.get("strongestRiskAlleles", []))
        rows.append({"trait": traits, "p": a.get("pvalue"), "beta": a.get("betaNum"),
                     "direction": a.get("betaDirection"), "or": a.get("orPerCopyNum"), "risk_allele": risk})
    return rows


def test5_phenotype(loc):
    say("\n=== TEST 5  Liver pattern or beta-cell pattern? (PheWAS of fine-mapped signals) ===")
    allr = []
    for r in loc[loc["rsid"].isin([TARGET, KNOWN])].itertuples():
        cs = ot_credible_sets(r.variant_id)
        for c in cs:
            c.update(rsid=r.rsid, risk_sign=r.risk_sign, source="Open Targets credible set")
        gc = gwas_catalog(r.rsid)
        for c in gc:
            c.update(rsid=r.rsid, type="gwas", source="GWAS Catalog")
        allr += cs + gc
    if not allr:
        say("  Test 5: NOT TESTED (no data returned)")
        return
    p = pd.DataFrame(allr)
    for c in ("type", "trait", "beta_alt", "pip", "p", "qtl_gene", "qtl_tissue", "project", "risk_sign"):
        if c not in p.columns:
            p[c] = np.nan
    p["risk_sign"] = p["risk_sign"].fillna(1)
    p["group"] = p["trait"].map(trait_group)
    p["beta_risk"] = pd.to_numeric(p["beta_alt"], errors="coerce") * p["risk_sign"]
    p.to_csv(f"{OUT}/test5_phewas.csv", index=False)
    gw = p[p["type"].astype(str).str.lower() == "gwas"]
    qtl = p[p["type"].astype(str).str.lower() != "gwas"]
    for rs in (TARGET, KNOWN):
        g = gw[gw["rsid"] == rs]
        say(f"\n  {rs}: {g['trait'].nunique()} GWAS traits")
        if len(g):
            cnt = g.drop_duplicates("trait")["group"].value_counts()
            say("    by group: " + ", ".join(f"{k} {v}" for k, v in cnt.items()))
            print(g.drop_duplicates("trait")[["trait", "group", "beta_risk", "pip", "p", "source"]]
                  .sort_values("group").to_string(index=False))
        q = qtl[qtl["rsid"] == rs]
        if len(q):
            print(f"    molecular QTL credible sets ({len(q)}):")
            print(q[["type", "qtl_gene", "qtl_tissue", "project", "beta_alt", "pip"]].to_string(index=False))
            pq = q[q["qtl_gene"] == "PROX1"]
            if len(pq):
                say(f"    PROX1 QTL in: {', '.join(sorted(set(map(str, pq['qtl_tissue']))))} "
                    f"(beta for risk allele: {', '.join(f'{b * r:.2f}' for b, r in zip(pq['beta_alt'], pq['risk_sign']) if pd.notna(b))})")
    for rs in (TARGET, KNOWN):
        g = gw[(gw["rsid"] == rs) & (gw["group"] == "other") & (pd.to_numeric(gw["pip"], errors="coerce") >= 0.5)]
        g = g.assign(p=pd.to_numeric(g["p"], errors="coerce")).sort_values("p").drop_duplicates("trait")
        if len(g):
            say(f"  {rs}: strongest non-diabetes signals (PIP >= 0.5): " + "; ".join(
                f"{t} (beta {b:+.3f}, p {pv:.1g})" for t, b, pv in zip(g["trait"], g["beta_risk"], g["p"])
                if pd.notna(pv))[:600])
    gw = gw[pd.to_numeric(gw["pip"], errors="coerce").fillna(1) >= 0.1]   # drop weak credible-set members
    g = gw[gw["rsid"] == TARGET].drop_duplicates("trait")
    liver_like = g["group"].isin(["insulin resistance", "lipids / bile", "liver"]).sum()
    beta_like = (g["group"] == "beta-cell / insulin secretion").sum()
    if liver_like == beta_like == 0:
        say("  Test 5: INCONCLUSIVE - rs17712208 has no fine-mapped insulin, lipid, liver or beta-cell traits")
    elif beta_like == 0:
        say(f"  Test 5: WEAK SUPPORT - liver-type traits {liver_like}, beta-cell traits 0 "
            f"(but HOMA-B GWAS rarely cover a 3-5% variant, so absence is not evidence)")
    elif liver_like > beta_like:
        say(f"  Test 5: PASS - liver-type traits {liver_like} vs beta-cell traits {beta_like}")
    else:
        say(f"  Test 5: FAIL - beta-cell traits {beta_like} >= liver-type traits {liver_like}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    loc = load_locus()
    tests = {1: test1_ld, 2: test2_tissue, 3: test3_eqtl, 4: test4_mechanism, 5: test5_phenotype}
    for s in a.steps:
        try:
            tests[s](loc)
        except SystemExit:
            raise
        except Exception as e:
            say(f"  Test {s}: ERROR - {type(e).__name__}: {e}  (send this line to Claude)")
    say("\n=== VERDICTS ===")
    for line in [l for l in SUMMARY if l.strip().startswith("Test ")]:
        print(line)
    tag = "_steps" + "".join(map(str, a.steps))
    open(f"{OUT}/prox1_case_study_summary{tag}.txt", "w").write("\n".join(SUMMARY))
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
