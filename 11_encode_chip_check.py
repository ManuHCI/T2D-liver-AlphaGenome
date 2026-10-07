"""
Step 11 - ENCODE check: do REAL ChIP-seq experiments show HNF1 sitting on rs17712208?

Steps 10 used AlphaGenome PREDICTIONS. Here we look only at LABORATORY data (ENCODE ChIP-seq peaks).

Part A  UCSC "ENCODE TF clusters" track (one request): every TF with a peak over the variant,
        across all ENCODE cell types. Also the ENCODE cCRE track (is it a registered enhancer?).
        The same is done at rs340874 as a comparison.
Part B  ENCODE portal: for selected TFs (HNF1A, HNF1B + other candidates) download every GRCh38
        peak file and ask: is there a peak over the variant? In which cell? How far is the
        peak summit (strongest point) from the variant?

How to read the result
  HNF1A/HNF1B peak over the variant in HepG2/liver, summit within ~50 bp -> HNF1 SUPPORTED by lab data
  HNF1 experiments exist in HepG2 but show no peak here                   -> HNF1 idea WEAKENED
  Many (>50) TFs have peaks here                                          -> "HOT" region: binding of any
                                                                             single TF is less specific
Run
  python 11_encode_chip_check.py
  python 11_encode_chip_check.py --tfs HNF1A HNF1B HNF4A FOXA2      # choose TFs
Results: results_prox1/encode_*.csv and encode_summary.txt (downloads cached in data/prox1/encode/)
"""
import argparse
import gzip
import io
import json
import os
import time

import pandas as pd
import requests

CACHE, OUT = "data/prox1/encode", "results_prox1"
SITES = {"rs17712208": ("chr1", 213977102), "rs340874": ("chr1", 213985913)}   # 1-based hg38
TARGET = "rs17712208"
UCSC = "https://api.genome.ucsc.edu/getData/track"
ENCODE = "https://www.encodeproject.org"
DEFAULT_TFS = ["HNF1A", "HNF1B", "HNF4A", "FOXA1", "FOXA2", "CEBPA", "CEBPB", "ONECUT1",
               "NR2F2", "MNX1", "BARX1", "LHX8", "HOXA1", "SOX5", "YY1"]
LIVER_LIKE = ("HepG2", "liver", "hepatocyte", "Hep G2")
PEAK_TYPES = ["conservative IDR thresholded peaks", "optimal IDR thresholded peaks",
              "IDR thresholded peaks", "pseudoreplicated peaks", "replicated peaks"]
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def get_json(url, params=None, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=90, headers={"Accept": "application/json"})
            if r.status_code == 404:
                return None
            if r.ok:
                return r.json()
            print(f"    HTTP {r.status_code} {r.url[:120]}")
        except Exception as e:
            print(f"    attempt {i + 1}: {e}")
        time.sleep(2 * (i + 1))
    return None


# ------------------------------------------------------------------ Part A: UCSC tracks
def ucsc(track, chrom, pos, flank):
    d = get_json(UCSC, {"genome": "hg38", "track": track, "chrom": chrom,
                        "start": pos - 1 - flank, "end": pos + flank})
    time.sleep(1.1)                                       # UCSC asks for <= 1 request / second
    if not d:
        return pd.DataFrame()
    items = d.get(track)
    if isinstance(items, dict):                           # some tracks are returned per chromosome
        items = items.get(chrom, [])
    return pd.DataFrame(items or [])


def part_a():
    say("\n=== PART A  ENCODE TF clusters + cCRE (UCSC, all ENCODE cell types) ===")
    rows = []
    for rs, (chrom, pos) in SITES.items():
        tf = ucsc("encRegTfbsClustered", chrom, pos, 0)
        if tf.empty:
            say(f"  {rs}: no TF peaks over the variant (or the UCSC request failed)")
        else:
            tf["site"] = rs
            rows.append(tf)
            names = sorted(tf["name"].astype(str).unique())
            say(f"  {rs}: {len(names)} TFs have a ChIP-seq peak over the variant")
            hnf = [n for n in names if n.upper().startswith("HNF1")]
            say(f"     HNF1 factors among them: {hnf or 'NONE'}")
            print("     all: " + ", ".join(names))
        cc = ucsc("encodeCcreCombined", chrom, pos, 0)
        if cc.empty:
            say(f"  {rs}: not inside an ENCODE candidate regulatory element (cCRE)")
        else:
            lab = [c for c in ("ucscLabel", "encodeLabel", "name", "description") if c in cc.columns]
            say(f"  {rs}: inside ENCODE cCRE -> " + "; ".join(str(cc.iloc[0][c]) for c in lab))
    if rows:
        pd.concat(rows).to_csv(f"{OUT}/encode_tf_clusters.csv", index=False)


# ------------------------------------------------------------------ Part B: ENCODE portal peak files
def experiments(tf):
    d = get_json(f"{ENCODE}/search/", {"type": "Experiment", "assay_title": "TF ChIP-seq",
                                        "target.label": tf, "status": "released",
                                        "format": "json", "limit": "all"})
    out = []
    for e in (d or {}).get("@graph", []):
        b = e.get("biosample_ontology") or {}
        cell = b.get("term_name", "?") if isinstance(b, dict) else str(b).strip("/").split("/")[-1]
        out.append((e["accession"], cell))
    return out


def best_peak_file(acc):
    d = get_json(f"{ENCODE}/search/", {"type": "File", "dataset": f"/experiments/{acc}/", "assembly": "GRCh38",
                                        "file_format": "bed", "status": "released",
                                        "format": "json", "limit": "all"})
    files = [f for f in (d or {}).get("@graph", []) if f.get("output_type") in PEAK_TYPES and f.get("href")]
    if not files:
        return None
    files.sort(key=lambda f: (not f.get("preferred_default", False), PEAK_TYPES.index(f["output_type"])))
    return files[0]


def peaks_over(fobj, chrom, pos, window=200):
    acc = fobj["accession"]
    path = f"{CACHE}/{acc}.bed.gz"
    if not os.path.exists(path):
        r = requests.get(ENCODE + fobj["href"], timeout=300)
        r.raise_for_status()
        open(path, "wb").write(r.content)
    hits = []
    with gzip.open(path, "rt") as f:
        for line in f:
            c = line.split("\t")
            if c[0] != chrom:
                continue
            s, e = int(c[1]), int(c[2])
            if s - window <= pos - 1 < e + window:
                summit = s + int(c[9]) if len(c) > 9 and c[9].strip() not in ("-1", "") else (s + e) // 2
                hits.append({"peak_start": s, "peak_end": e, "covers_variant": s <= pos - 1 < e,
                             "summit_to_variant_bp": pos - 1 - summit,
                             "signal": float(c[6]) if len(c) > 6 else None})
    return hits


def part_b(tfs):
    say("\n=== PART B  ENCODE peak files for selected TFs (any cell type) ===")
    rows = []
    for tf in tfs:
        exps = experiments(tf)
        print(f"  {tf}: {len(exps)} experiments found, downloading peak files (cached after the first run)...")
        if not exps:
            say(f"  {tf}: no released ENCODE TF ChIP-seq experiment")
            continue
        n_hit = 0
        for acc, cell in exps:
            fobj = best_peak_file(acc)
            if not fobj:
                continue
            for rs, (chrom, pos) in SITES.items():
                try:
                    hits = peaks_over(fobj, chrom, pos)
                except Exception as e:
                    print(f"    {tf} {acc}: download failed ({e})")
                    hits = None
                if hits is None:
                    continue
                best = min(hits, key=lambda h: abs(h["summit_to_variant_bp"])) if hits else {}
                bound = bool(best) and best["covers_variant"]
                n_hit += bound and rs == TARGET
                rows.append({"tf": tf, "experiment": acc, "cell": cell, "file": fobj["accession"],
                             "peak_type": fobj["output_type"], "site": rs, "bound": bound, **best})
        sub = [r for r in rows if r["tf"] == tf and r["site"] == TARGET]
        liver = [r for r in sub if any(k.lower() in str(r["cell"]).lower() for k in LIVER_LIKE)]
        say(f"  {tf}: {len(sub)} experiments; peak over {TARGET} in {sum(r['bound'] for r in sub)} "
            f"(liver/HepG2: {sum(r['bound'] for r in liver)} of {len(liver)})")
        for r in sub:
            if r["bound"]:
                say(f"      BOUND  {r['cell']:<28} {r['experiment']}  summit {r['summit_to_variant_bp']:+d} bp "
                    f"from variant, signal {r['signal']}")
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(f"{OUT}/encode_selected_tf_peaks.csv", index=False)
    return df


def verdict(df):
    say("\n=== VERDICT (laboratory ChIP-seq data) ===")
    if df is None or df.empty:
        say("  no ENCODE peak data obtained - check the internet connection and send the output to Claude")
        return
    t = df[df["site"] == TARGET]
    h = t[t["tf"].str.startswith("HNF1")]
    hl = h[h["cell"].astype(str).str.contains("|".join(LIVER_LIKE), case=False)]
    close = hl[hl["bound"] & (hl["summit_to_variant_bp"].abs() <= 50)]
    if len(close):
        say(f"  HNF1 SUPPORTED: real HNF1 ChIP peak over rs17712208 in liver/HepG2, summit within 50 bp "
            f"({', '.join(sorted(set(close['tf'])))})")
    elif hl["bound"].any():
        say("  HNF1 PARTLY SUPPORTED: HNF1 peak covers the variant in liver/HepG2, but its summit is > 50 bp away")
    elif len(hl):
        say(f"  HNF1 WEAKENED: {len(hl)} HNF1 experiments in liver/HepG2 show NO peak at rs17712208")
    elif h["bound"].any():
        say("  HNF1 bound only in non-liver cells: " + ", ".join(sorted(set(h[h['bound']]['cell']))))
    else:
        say("  HNF1 NOT TESTABLE: no HNF1 experiment found")
    other = t[t["bound"] & ~t["tf"].str.startswith("HNF1")]
    if len(other):
        say("  other selected TFs with a peak here: " + ", ".join(
            f"{a} ({b})" for a, b in sorted(set(zip(other["tf"], other["cell"])))))
    c = df[(df["site"] == "rs340874") & df["bound"]]
    say(f"  comparison rs340874: {c['tf'].nunique()} of the selected TFs bound there")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", nargs="+", default=DEFAULT_TFS)
    ap.add_argument("--skip-a", action="store_true")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    if not a.skip_a:
        part_a()
    df = part_b(a.tfs)
    verdict(df)
    open(f"{OUT}/encode_summary.txt", "w").write("\n".join(LINES))
    print(f"\nSaved to {OUT}/encode_*.csv and encode_summary.txt")


if __name__ == "__main__":
    main()
