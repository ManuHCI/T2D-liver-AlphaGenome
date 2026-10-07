"""
Step 12 - COLOCALIZATION: is the alanine / glutamine signal the SAME causal variant as the T2D signal?

Why: rs17712208 is linked to T2D and to blood alanine/glutamine. Two different causal variants that sit
close together could give the same picture. Colocalization asks: do both traits point to the SAME variant?

Method (uses fine-mapped credible sets from Open Targets)
  Each credible set gives every candidate variant a probability of being causal (PIP).
  Part A  Open Targets' own pre-computed colocalisation results for our T2D credible sets
          (COLOC H4 = probability of one shared causal variant; eCAVIAR CLPP).
  Part B  Our own check: CLPP = sum over variants of PIP(T2D) x PIP(trait).
          CLPP >= 0.1 -> the same variant very likely drives both traits.
  Control the same calculation for the OLD signal rs340874 (should NOT colocalise with alanine).
  Direction  does the T2D-risk allele raise or lower each trait?

Run:  python 12_colocalization.py
Results: results_prox1/coloc_*.csv and coloc_summary.txt (cached in data/prox1/coloc/)
"""
import hashlib
import json
import os
import re
import sys
import time

import pandas as pd
import requests

API = "https://api.platform.opentargets.org/api/v4/graphql"
CACHE, OUT = "data/prox1/coloc", "results_prox1"
VARS = {"rs17712208": "1_213977102_T_A", "rs340874": "1_213985913_T_C"}
RISK_SIGN = {"rs17712208": 1, "rs340874": 1}          # risk allele = ALT for both (step 1)
T2D_RX = r"type 2 diabetes|diabetes mellitus|^diabetes$|non-insulin-dependent"
TRAITS = {                                             # traits of interest -> regex on traitFromSource
    "alanine": r"^alanine", "glutamine": r"^glutamine", "BCAA": r"branched-chain|^valine|^leucine|^isoleucine",
    "HbA1c": r"hba1c|glycated", "glucose": r"glucose", "LDL": r"LDL|low density lipoprotein",
    "GGT": r"gamma glutamyl", "albumin": r"albumin",
}
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def gql(query, variables):
    key = hashlib.md5((query + json.dumps(variables, sort_keys=True)).encode()).hexdigest()[:16]
    path = f"{CACHE}/{key}.json"
    if os.path.exists(path):
        return json.load(open(path))
    for i in range(4):
        try:
            r = requests.post(API, json={"query": query, "variables": variables}, timeout=120)
            js = r.json()
            if "errors" in js:
                print("  Open Targets error:", js["errors"][0].get("message", "")[:300])
                return None
            json.dump(js["data"], open(path, "w"))
            return js["data"]
        except Exception as e:
            print(f"  retry {i + 1}: {e}")
            time.sleep(5 * (i + 1))
    return None


def show_type(name):
    d = gql('query($n:String!){ __type(name:$n){ fields{ name } } }', {"n": name})
    if d and d.get("__type"):
        print(f"  fields of {name}: " + ", ".join(f["name"] for f in d["__type"]["fields"]))


# ------------------------------------------------------------------ credible sets containing a variant
Q_VAR = """
query($id:String!, $index:Int!){
  variant(variantId:$id){
    credibleSets(page:{index:$index, size:100}){
      count
      rows{
        studyLocusId
        study{ id studyType traitFromSource projectId }
        locus(variantIds:[$id]){ rows{ beta posteriorProbability pValueMantissa pValueExponent } }
      }
    }
  }
}"""


def sets_with(rs):
    out, index = [], 0
    while True:
        d = gql(Q_VAR, {"id": VARS[rs], "index": index})
        if not d or not d.get("variant"):
            break
        cs = d["variant"]["credibleSets"]
        for r in cs["rows"]:
            s = r["study"] or {}
            l = ((r.get("locus") or {}).get("rows") or [{}])[0]
            out.append({"rsid": rs, "studyLocusId": r["studyLocusId"], "study": s.get("id"),
                        "type": s.get("studyType"), "trait": s.get("traitFromSource"),
                        "pip_here": l.get("posteriorProbability"),
                        "beta_risk": (l["beta"] * RISK_SIGN[rs]) if l.get("beta") is not None else None,
                        "p": (l["pValueMantissa"] * 10.0 ** l["pValueExponent"])
                        if l.get("pValueMantissa") is not None else None})
        index += 1
        if index * 100 >= cs["count"]:
            break
    return pd.DataFrame(out)


# ------------------------------------------------------------------ full PIP list of one credible set
Q_LOCUS = """
query($ids:[String!], $li:Int!){
  credibleSets(studyLocusIds:$ids, page:{index:0, size:1}){
    rows{ studyLocusId
      locus(page:{index:$li, size:500}){ count rows{ posteriorProbability variant{ id } } } }
  }
}"""


def pips(sl_id):
    out, li = {}, 0
    while True:
        d = gql(Q_LOCUS, {"ids": [sl_id], "li": li})
        if d is None:
            show_type("Query")
            sys.exit("credibleSets(studyLocusIds:...) not accepted - send this output to Claude")
        rows = ((d or {}).get("credibleSets") or {}).get("rows") or []
        if not rows:
            break
        loc = rows[0]["locus"]
        for r in loc["rows"]:
            out[r["variant"]["id"]] = r["posteriorProbability"] or 0.0
        li += 1
        if li * 500 >= loc["count"]:
            break
    if not out:
        print(f"  WARNING: no variants returned for credible set {sl_id}")
    return out


def clpp(a, b):
    return sum(p * b.get(v, 0.0) for v, p in a.items())


# ------------------------------------------------------------------ Part A: Open Targets coloc
Q_COLOC = """
query($ids:[String!], $index:Int!){
  credibleSets(studyLocusIds:$ids, page:{index:0, size:1}){
    rows{ studyLocusId
      colocalisation(page:{index:$index, size:100}){
        count
        rows{ colocalisationMethod h3 h4 clpp numberColocalisingVariants betaRatioSignAverage
              otherStudyLocus{ studyLocusId study{ id studyType traitFromSource } variant{ id rsIds } } }
      }
    }
  }
}"""


def ot_coloc(sl_id):
    out, index = [], 0
    while True:
        d = gql(Q_COLOC, {"ids": [sl_id], "index": index})
        if d is None:
            print("  (Open Targets colocalisation query not accepted - schema below, send to Claude)")
            show_type("CredibleSet")
            show_type("Colocalisation")
            return pd.DataFrame()
        rows = (d.get("credibleSets") or {}).get("rows") or []
        if not rows:
            break
        c = rows[0]["colocalisation"]
        for r in c["rows"]:
            o = r.get("otherStudyLocus") or {}
            st = o.get("study") or {}
            v = o.get("variant") or {}
            out.append({"method": r.get("colocalisationMethod"), "h3": r.get("h3"), "h4": r.get("h4"),
                        "clpp": r.get("clpp"), "n_shared_variants": r.get("numberColocalisingVariants"),
                        "beta_ratio_sign": r.get("betaRatioSignAverage"),
                        "other_type": st.get("studyType"), "other_trait": st.get("traitFromSource"),
                        "other_study": st.get("id"), "other_lead": ";".join(v.get("rsIds") or [v.get("id", "")])})
        index += 1
        if index * 100 >= c["count"]:
            break
    return pd.DataFrame(out)


def trait_group(t):
    for g, rx in TRAITS.items():
        if re.search(rx, str(t), re.I):
            return g
    return None


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    say("=== Credible sets that contain each variant ===")
    cs = pd.concat([sets_with(rs) for rs in VARS], ignore_index=True)
    if cs.empty:
        sys.exit("No credible sets returned - check the internet connection.")
    cs["group"] = cs["trait"].map(trait_group)
    cs["is_t2d"] = cs["trait"].str.contains(T2D_RX, case=False, regex=True, na=False) & \
        (cs["type"].astype(str).str.lower() == "gwas")
    cs.to_csv(f"{OUT}/coloc_credible_sets.csv", index=False)
    for rs in VARS:
        c = cs[cs["rsid"] == rs]
        say(f"  {rs}: {len(c)} credible sets ({int(c['is_t2d'].sum())} T2D)")

    # main T2D credible set per variant = the one where the variant has the highest PIP
    t2d = cs[cs["is_t2d"]].sort_values("pip_here", ascending=False).drop_duplicates("rsid")
    if t2d.empty:
        sys.exit("No T2D credible set found")
    say("\n  T2D credible sets used:")
    for r in t2d.itertuples():
        say(f"    {r.rsid}: {r.study} '{r.trait}' (PIP of the variant {r.pip_here:.2f})")

    # ---------------- Part A
    say("\n=== PART A  Open Targets pre-computed colocalisation with the T2D credible set ===")
    allA = []
    for r in t2d.itertuples():
        a = ot_coloc(r.studyLocusId)
        if a.empty:
            continue
        a["rsid"] = r.rsid
        a["group"] = a["other_trait"].map(trait_group)
        allA.append(a)
        g = a[a["group"].notna()].copy()
        g["score"] = g[["h4", "clpp"]].max(axis=1, skipna=True)
        g = g.sort_values("score", ascending=False).drop_duplicates(["group", "method"])
        say(f"  {r.rsid}: {len(a)} colocalisation tests; traits of interest:")
        for x in g.itertuples():
            say(f"    {x.group:<9} {str(x.other_trait)[:45]:<45} {x.method:<8} "
                f"H4={x.h4 if pd.notna(x.h4) else '-':<6} CLPP={x.clpp if pd.notna(x.clpp) else '-'}")
    if allA:
        pd.concat(allA).to_csv(f"{OUT}/coloc_open_targets.csv", index=False)

    # ---------------- Part B
    say("\n=== PART B  Own CLPP = sum PIP(T2D) x PIP(trait) ===")
    trait_sets = cs[cs["group"].notna() & (cs["rsid"] == "rs17712208")]
    trait_sets = trait_sets.sort_values("pip_here", ascending=False).drop_duplicates("group")
    rows = []
    for t in t2d.itertuples():
        pt = pips(t.studyLocusId)
        for s in trait_sets.itertuples():
            ps = pips(s.studyLocusId)
            v = clpp(pt, ps)
            rows.append({"t2d_signal": t.rsid, "trait": s.group, "trait_study": s.study,
                         "trait_name": s.trait, "clpp": v, "beta_risk_on_trait": s.beta_risk,
                         "n_t2d_variants": len(pt), "n_trait_variants": len(ps)})
    b = pd.DataFrame(rows)
    b.to_csv(f"{OUT}/coloc_own_clpp.csv", index=False)
    if len(b):
        print(b.pivot(index="trait", columns="t2d_signal", values="clpp").round(3).to_string())

    # ---------------- verdict
    say("\n=== VERDICT ===")
    for tr in ("alanine", "glutamine", "BCAA", "HbA1c"):
        x = b[(b["trait"] == tr)]
        new = x[x["t2d_signal"] == "rs17712208"]["clpp"]
        old = x[x["t2d_signal"] == "rs340874"]["clpp"]
        if new.empty:
            say(f"  {tr}: no credible set containing rs17712208")
            continue
        beta = pd.to_numeric(x["beta_risk_on_trait"], errors="coerce").iloc[0]
        call = "COLOCALISES" if new.iloc[0] >= 0.1 else "does NOT colocalise"
        direction = (f"risk allele {'raises' if beta > 0 else 'lowers'} {tr} (beta {beta:+.3f})"
                     if pd.notna(beta) else "direction not available")
        old_txt = f"{old.iloc[0]:.2f}" if len(old) else "n/a"
        say(f"  {tr}: {call} with the rs17712208 T2D signal (CLPP {new.iloc[0]:.2f}); "
            f"rs340874 T2D signal CLPP {old_txt}; {direction}")
    say("  CLPP >= 0.1: same causal variant likely; ~0: different variants. Open Targets H4 >= 0.8 = colocalised.")
    open(f"{OUT}/coloc_summary.txt", "w").write("\n".join(LINES))
    print(f"\nSaved to {OUT}/coloc_*.csv and coloc_summary.txt")


if __name__ == "__main__":
    main()
