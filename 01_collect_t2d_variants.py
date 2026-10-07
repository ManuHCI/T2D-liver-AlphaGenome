"""
Step 1 - Collect the "diabetic" DNA variants for the hepatocyte study.

Source : Open Targets Platform (GraphQL API), type 2 diabetes GWAS studies,
         SuSiE fine-mapped credible sets.
Output : data/t2d_variants_all.csv          every credible-set variant
         data/t2d_variants_for_alphagenome.csv  likely-causal, non-coding (to score)
         data/t2d_variants_coding.csv       likely-causal, protein-altering (protein route later)
         data/t2d_studies.csv               list of T2D studies found

Logic
- "Normal" sequence   = reference genome base (REF)
- "Diabetic" sequence = the T2D risk allele. Open Targets reports beta for the ALT allele,
  so risk allele = ALT when beta > 0, REF when beta < 0. Column `risk_sign` stores +1 / -1;
  step 3 multiplies AlphaGenome scores by it so every score reads "diabetic minus normal".
- Keep only likely-causal variants (PIP >= 0.2) - lesson from the liver positive control
  (LD dilution: low-PIP variants carry the beta but not the sequence effect).

Usage
  python 01_collect_t2d_variants.py                 # auto-picks the largest T2D GWAS
  python 01_collect_t2d_variants.py --list-only     # just show the studies, then choose
  python 01_collect_t2d_variants.py --study GCST...  # use a specific study (repeatable)
  python 01_collect_t2d_variants.py --min-pip 0.1
"""
import argparse
import os
import sys
import time

import pandas as pd
import requests

API = "https://api.platform.opentargets.org/api/v4/graphql"
DISEASE_IDS = ["MONDO_0005148", "EFO_0001360"]  # type 2 diabetes mellitus (new id, old id)
OUT = "data"

PROTEIN_ALTERING = {
    "missense_variant", "stop_gained", "stop_lost", "start_lost",
    "frameshift_variant", "inframe_insertion", "inframe_deletion",
    "protein_altering_variant", "incomplete_terminal_codon_variant",
}


Q_TYPE = 'query($n:String!){ __type(name:$n){ name fields{ name args{ name } } } }'


def show_schema():
    """print the fields Open Targets currently accepts for the types we use."""
    print("\nCurrent Open Targets fields (send this to Claude):")
    for t in ["CredibleSet", "Locus", "Loci", "Variant", "CredibleSets"]:
        r = requests.post(API, json={"query": Q_TYPE, "variables": {"n": t}}, timeout=60)
        typ = (r.json().get("data") or {}).get("__type")
        if not typ:
            print(f"  {t}: (type not found)")
            continue
        names = [f["name"] + (f"({','.join(a['name'] for a in f['args'])})" if f["args"] else "")
                 for f in typ["fields"]]
        print(f"  {t}: {', '.join(names)}")


class TooExpensive(Exception):
    pass


def gql(query, variables, retries=4):
    for attempt in range(retries):
        try:
            r = requests.post(API, json={"query": query, "variables": variables}, timeout=120)
            if r.status_code == 400 and "too expensive" in r.text.lower():
                raise TooExpensive()
            if 400 <= r.status_code < 500:          # query problem: retrying will not help
                print(f"\nOpen Targets rejected the query (HTTP {r.status_code}). Server message:")
                try:
                    for e in r.json().get("errors", []):
                        print("  ", e.get("message"))
                except ValueError:
                    print("  ", r.text[:1500])
                show_schema()
                sys.exit(1)
            r.raise_for_status()
            js = r.json()
            if "errors" in js:
                print("\nGraphQL error - please send this message to Claude:")
                for e in js["errors"]:
                    print("  ", e.get("message"))
                show_schema()
                sys.exit(1)
            return js["data"]
        except requests.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f"  network problem ({e}); retry in {wait}s")
            time.sleep(wait)
    sys.exit("Open Targets API not reachable.")


Q_DISEASE = "query($id:String!){ disease(efoId:$id){ id name } }"

Q_STUDIES = """
query($ids:[String!], $index:Int!, $size:Int!){
  studies(diseaseIds:$ids, page:{index:$index, size:$size}){
    count
    rows{ id studyType traitFromSource projectId publicationFirstAuthor
          publicationDate nSamples nCases nControls }
  }
}"""

Q_CREDSETS = """
query($studyIds:[String!], $index:Int!, $li:Int!, $ls:Int!){
  credibleSets(studyIds:$studyIds, page:{index:$index, size:1}){
    count
    rows{
      studyLocusId studyId finemappingMethod
      variant{ id }
      locus(page:{index:$li, size:$ls}){
        count
        rows{
          posteriorProbability is95CredibleSet beta
          variant{ id chromosome position referenceAllele alternateAllele rsIds
                   mostSevereConsequence{ label } }
        }
      }
    }
  }
}"""


Q_CS_LIGHT = """
query($studyIds:[String!], $index:Int!, $size:Int!){
  credibleSets(studyIds:$studyIds, page:{index:$index, size:$size}){
    count  rows{ finemappingMethod }
  }
}"""

COMBINED_WORDS = (" or ", " and ", "pleiotrop", "adjusted", "|", "/", "complication",
                  "nephropathy", "retinopathy", "neuropathy")


def is_pure_t2d(trait):
    t = str(trait).lower().strip()
    if any(w in t for w in COMBINED_WORDS):
        return False
    return ("type 2 diabetes" in t or "type ii diabetes" in t or t in {"t2d", "diabetes mellitus type 2"})


def finemapping_summary(study_id):
    """number of credible sets and how many are SuSiE."""
    methods, index, size = [], 0, 100
    while True:
        d = gql(Q_CS_LIGHT, {"studyIds": [study_id], "index": index, "size": size})["credibleSets"]
        methods += [str(r["finemappingMethod"]) for r in d["rows"]]
        if len(methods) >= d["count"] or not d["rows"]:
            break
        index += 1
    n_susie = sum("susi" in m.lower() for m in methods)
    return len(methods), n_susie, ",".join(sorted(set(methods)))


def get_disease_ids():
    found = []
    for d in DISEASE_IDS:
        data = gql(Q_DISEASE, {"id": d})
        if data.get("disease"):
            print(f"Disease found: {d} = {data['disease']['name']}")
            found.append(d)
    if not found:
        sys.exit("Type 2 diabetes id not found in Open Targets.")
    return found


def get_studies(disease_ids):
    rows, index, size = [], 0, 100
    while True:
        data = gql(Q_STUDIES, {"ids": disease_ids, "index": index, "size": size})["studies"]
        rows += data["rows"]
        if len(rows) >= data["count"] or not data["rows"]:
            break
        index += 1
    df = pd.DataFrame(rows).drop_duplicates("id")
    df = df[df["studyType"].astype(str).str.lower() == "gwas"]
    return df.sort_values("nCases", ascending=False, na_position="last")


def fetch_one(study_id, index, li, ls):
    """one credible set, one page of its variants; shrinks page size if too expensive."""
    while True:
        try:
            return gql(Q_CREDSETS, {"studyIds": [study_id], "index": index,
                                    "li": li, "ls": ls})["credibleSets"], ls
        except TooExpensive:
            if ls <= 10:
                sys.exit("Open Targets refuses even 10 variants per request - send this to Claude.")
            ls //= 2
            li = 0       # restart paging with the smaller size (caller resets too)
            print(f"    page too large, reducing to {ls} variants per request")


def get_credible_sets(study_id, page_size=200):
    out, index, n_total, seen = [], 0, None, set()
    while n_total is None or index < n_total:
        li, ls, rows_cs, cs_info = 0, page_size, [], None
        while True:
            data, new_ls = fetch_one(study_id, index, li, ls)
            if new_ls != ls:                       # page size changed: restart this set
                ls, li, rows_cs, page_size = new_ls, 0, [], new_ls
                continue
            n_total = data["count"]
            if not data["rows"]:
                break
            cs = data["rows"][0]
            cs_info = cs
            if "susi" not in str(cs["finemappingMethod"]).lower():
                break                              # skip PICS sets - no need for variants
            rows_cs += cs["locus"]["rows"]
            if len(rows_cs) >= cs["locus"]["count"] or not cs["locus"]["rows"]:
                break
            li += 1
        if cs_info and cs_info["studyLocusId"] not in seen:
            seen.add(cs_info["studyLocusId"])
            for v in rows_cs:
                var = v["variant"]
                out.append({
                    "study_id": cs_info["studyId"],
                    "study_locus_id": cs_info["studyLocusId"],
                    "finemapping_method": cs_info["finemappingMethod"],
                    "lead_variant": (cs_info.get("variant") or {}).get("id"),
                    "variant_id": var["id"],
                    "rsid": ";".join(var.get("rsIds") or []),
                    "chrom": "chr" + str(var["chromosome"]),
                    "pos": var["position"],
                    "ref": var["referenceAllele"],
                    "alt": var["alternateAllele"],
                    "consequence": (var.get("mostSevereConsequence") or {}).get("label"),
                    "pip": v["posteriorProbability"],
                    "in_95_cs": v["is95CredibleSet"],
                    "beta_alt": v.get("beta"),
                })
        index += 1
        if index % 50 == 0 or index == n_total:
            print(f"  {study_id}: {index}/{n_total} credible sets, {len(out):,} variant rows")
    if len(seen) < (n_total or 0):
        print(f"  note: {n_total - len(seen)} sets returned twice or empty (PICS sets are skipped on purpose)")
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", action="append", help="study id(s) to use")
    ap.add_argument("--min-pip", type=float, default=0.2)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--n-check", type=int, default=12, help="how many pure-T2D studies to check")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    studies = get_studies(get_disease_ids())
    studies["pure_t2d"] = studies["traitFromSource"].apply(is_pure_t2d)
    studies.to_csv(f"{OUT}/t2d_studies.csv", index=False)
    pure = studies[studies["pure_t2d"]].copy()
    print(f"\n{len(studies)} T2D GWAS studies; {len(pure)} are pure T2D "
          f"(combined / pleiotropy traits removed).")

    if not a.study:
        print(f"\nChecking fine-mapping for the {a.n_check} largest pure-T2D studies ...")
        info = [finemapping_summary(s) for s in pure.head(a.n_check)["id"]]
        top = pure.head(a.n_check).copy()
        top["n_credible_sets"] = [i[0] for i in info]
        top["n_susie"] = [i[1] for i in info]
        top["methods"] = [i[2] for i in info]
        top.to_csv(f"{OUT}/t2d_pure_studies_checked.csv", index=False)
        print(top[["id", "publicationFirstAuthor", "publicationDate", "nCases", "nSamples",
                   "n_credible_sets", "n_susie", "methods"]].to_string(index=False))
        if a.list_only:
            return
        usable = top[top["n_susie"] >= 20]
        if usable.empty:
            sys.exit("No pure-T2D study with >= 20 SuSiE credible sets - send this output to Claude.")
        chosen = [usable.iloc[0]["id"]]
    else:
        chosen = a.study
    print(f"\nUsing study: {chosen}")
    parts = []
    for s in chosen:
        cache = f"{OUT}/cache_{s}.csv"
        if os.path.exists(cache):
            print(f"  {s}: loaded from {cache}")
            parts.append(pd.read_csv(cache))
            continue
        d = get_credible_sets(s)
        d.to_csv(cache, index=False)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        sys.exit("No credible sets returned for that study - try another with --study.")

    df["finemapping_method"] = df["finemapping_method"].astype(str)
    n0 = df["study_locus_id"].nunique()
    df = df[df["finemapping_method"].str.contains("SuSi", case=False)]
    print(f"\nCredible sets: {n0} total, {df['study_locus_id'].nunique()} SuSiE")
    df.to_csv(f"{OUT}/t2d_variants_all.csv", index=False)

    # replication: in how many of the chosen studies is the variant likely causal?
    hi = df[df["pip"] >= a.min_pip]
    df["n_studies_supporting"] = df["variant_id"].map(hi.groupby("variant_id")["study_id"].nunique()).fillna(0).astype(int)
    df["studies_supporting"] = df["variant_id"].map(
        hi.groupby("variant_id")["study_id"].agg(lambda x: ";".join(sorted(set(x))))).fillna("")
    # one row per variant: keep the credible set where it has the highest PIP
    df = df.sort_values("pip", ascending=False).drop_duplicates("variant_id")
    df = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)]   # SNVs only
    df = df[df["pip"] >= a.min_pip]
    df = df[df["beta_alt"].notna() & (df["beta_alt"] != 0)]
    df["risk_allele"] = df.apply(lambda r: r["alt"] if r["beta_alt"] > 0 else r["ref"], axis=1)
    df["risk_sign"] = df["beta_alt"].apply(lambda b: 1 if b > 0 else -1)
    df["protein_altering"] = df["consequence"].isin(PROTEIN_ALTERING)

    coding = df[df["protein_altering"]]
    noncoding = df[~df["protein_altering"]]
    coding.to_csv(f"{OUT}/t2d_variants_coding.csv", index=False)
    noncoding.to_csv(f"{OUT}/t2d_variants_for_alphagenome.csv", index=False)

    print(f"Likely-causal SNVs (PIP >= {a.min_pip}): {len(df)}")
    print(f"  to AlphaGenome (non-coding / synonymous / splice): {len(noncoding)}")
    print(f"  protein-altering (kept aside for protein tools):  {len(coding)}")
    print(f"  risk allele = ALT: {(df['risk_sign'] == 1).sum()},  = REF: {(df['risk_sign'] == -1).sum()}")
    if len(chosen) > 1:
        print("  variants likely causal in >1 study (replicated):",
              (noncoding["n_studies_supporting"] > 1).sum())
        print("  per study:\n", noncoding["study_id"].value_counts().to_string())
    print("\nTop consequences:\n", noncoding["consequence"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
