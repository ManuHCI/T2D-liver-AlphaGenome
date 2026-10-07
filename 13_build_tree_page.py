"""
Step 13 - Build the interactive regulatory tree (revised, hybrid Level 1) as one HTML page.

Reads   results_hybrid/  (step 8 + step 4 outputs) and data/net_collectri.tsv, data/net_omnipath.tsv
Writes  docs/index.html          -> open in any browser; GitHub Pages serves the docs/ folder
        docs/tree_data.json      -> the same data as a file (for re-use)

Run:    python 13_build_tree_page.py            (needs tree_template.html next to this script)
"""
import json
import os

import numpy as np
import pandas as pd

RES, DATA, OUT = "results_hybrid", "data", "docs"
TEMPLATE = "tree_template.html"
MAX_TARGETS = 60


def clean(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def load_network():
    out = []
    for ds, kind in [("collectri", "TF->target"), ("omnipath", "signalling")]:
        d = pd.read_csv(f"{DATA}/net_{ds}.tsv", sep="\t")
        if kind == "signalling" and "consensus_direction" in d.columns:
            d = d[d["consensus_direction"].astype(str).isin(["1", "True", "true"])]
        st = d["is_stimulation"].astype(str).isin(["1", "True", "true"]).astype(int)
        inh = d["is_inhibition"].astype(str).isin(["1", "True", "true"]).astype(int)
        d["sign"] = np.where((st == 1) & (inh == 0), 1, np.where((inh == 1) & (st == 0), -1, 0))
        d = d[d["sign"] != 0].rename(columns={"source_genesymbol": "s", "target_genesymbol": "t"})
        d = d.assign(s=d["s"].str.split("_")).explode("s").assign(t=lambda x: x["t"].str.split("_")).explode("t")
        d["type"] = kind
        out.append(d[["s", "t", "sign", "type"]])
    net = pd.concat(out).drop_duplicates()
    agg = (net.groupby(["s", "t"]).agg(sign=("sign", "mean"), type=("type", lambda x: "+".join(sorted(set(x)))))
           .reset_index())
    agg = agg[(agg["sign"].abs() == 1) & (agg["s"] != agg["t"])]
    agg["sign"] = agg["sign"].astype(int)
    corr = f"{DATA}/edge_corrections.csv"
    if os.path.exists(corr):
        for r in pd.read_csv(corr).itertuples():
            m = (agg["s"] == r.source) & (agg["t"] == r.target)
            if r.sign == 0:
                agg = agg[~m]
            elif m.any():
                agg.loc[m, "sign"] = int(r.sign)
            else:
                agg = pd.concat([agg, pd.DataFrame([{"s": r.source, "t": r.target, "sign": int(r.sign),
                                                     "type": "literature"}])])
    return agg


def main():
    nodes = pd.read_csv(f"{RES}/tree_nodes.csv")
    edges = pd.read_csv(f"{RES}/tree_edges.csv")
    l1 = pd.read_csv(f"{RES}/level0_genes.csv")
    links = pd.read_csv(f"{RES}/level0_gene_variant_links.csv")
    mech = pd.read_csv(f"{RES}/tree_level0_mechanisms.csv")
    net = load_network()
    etype = {(r.s, r.t): r.type for r in net.itertuples()}

    nd = nodes.set_index("gene")
    l1score = dict(zip(l1["gene_name"], l1["weighted_score"]))
    l1tf = dict(zip(l1["gene_name"], l1["is_TF"]))
    shown = set(edges["parent"]) | set(edges["child"]) | set(l1["gene_name"])

    def node(g, extra=False):
        r = nd.loc[g] if g in nd.index else None
        lvl = 1 if g in l1score else (int(min(r["level"], 3)) if r is not None else 3)
        push = clean(r["net_push"]) if r is not None else None
        d = {"id": g, "level": lvl,
             "dir": int(np.sign(l1score[g])) if g in l1score else (int(np.sign(push)) if push else 0),
             "push": push, "nL1": clean(r["n_level1_genes"]) if r is not None else None,
             "agree": clean(round(r["agreement"], 3)) if r is not None else None,
             "p": clean(r["p_push"]) if r is not None else None,
             "fdr": clean(r["fdr_push"]) if r is not None and lvl > 1 else None,
             "tf": bool(l1tf.get(g, r["is_TF"] if r is not None else False)),
             "tpm": clean(round(r["liver_TPM_normal"], 1)) if r is not None else None,
             "explo": bool(r["exploratory"]) if r is not None else False,
             "l1score": clean(l1score.get(g))}
        if extra:
            d["extra"] = True
        return d

    out_nodes, seen = [], set()
    # Level 0: variants
    mech_by_var = mech.groupby("variant")
    for v, grp in links.groupby("rsid"):
        m = mech_by_var.get_group(v) if v in mech_by_var.groups else pd.DataFrame()
        enh = next((x for x in m.get("enhancer", pd.Series()).dropna()), None) if len(m) else None
        tfb = next((x for x in m.get("TF_binding_change", pd.Series()).dropna()), None) if len(m) else None
        out_nodes.append({"id": v, "level": 0, "pip": clean(grp["pip"].iloc[0]), "enh": enh, "tfb": tfb,
                          "genes": sorted(grp["gene_name"].unique().tolist())})
        seen.add(v)
    # Level 1-3 shown nodes
    for g in sorted(shown):
        if g not in seen:
            out_nodes.append(node(g))
            seen.add(g)
    # direct-target lists for shown genes (lineage view), adding hidden target nodes as "extra"
    by_src = {s: grp for s, grp in net.groupby("s")}
    for n in list(out_nodes):
        if n["level"] in (1, 2) and n["id"] in by_src:
            tg = by_src[n["id"]]
            tg = tg[tg["t"].isin(nd.index) | tg["t"].isin(shown)]
            if tg.empty:
                continue
            tl = []
            for r in tg.itertuples():
                tn = nd.loc[r.t] if r.t in nd.index else None
                push = clean(tn["net_push"]) if tn is not None else None
                tl.append({"t": r.t, "sign": int(r.sign), "type": r.type, "dir": int(np.sign(push)) if push else 0,
                           "push": push, "level": int(min(tn["level"], 3)) if tn is not None else 3})
                if r.t not in seen:
                    out_nodes.append(node(r.t, extra=True))
                    seen.add(r.t)
            n["targets"] = sorted(tl, key=lambda z: -abs(z["push"] or 0))[:MAX_TARGETS]

    out_edges = [{"s": r.rsid, "t": r.gene_name, "sign": int(np.sign(r.hepatic_score)), "type": "AlphaGenome",
                  "ch": clean(r.hepatic_score)} for r in links.itertuples()]
    for r in edges.itertuples():
        out_edges.append({"s": r.parent, "t": r.child, "sign": int(r.sign),
                          "type": etype.get((r.parent, r.child), "literature")})

    data = {"nodes": out_nodes, "edges": out_edges}
    os.makedirs(OUT, exist_ok=True)
    json.dump(data, open(f"{OUT}/tree_data.json", "w"), separators=(",", ":"))
    page = open(TEMPLATE, encoding="utf-8").read().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    open(f"{OUT}/tree_fragment.html", "w", encoding="utf-8").write(page)       # body-only version
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n' + page + "\n</html>\n")
    open(f"{OUT}/index.html", "w", encoding="utf-8").write(html)
    cnt = pd.Series([n["level"] for n in out_nodes if not n.get("extra")]).value_counts().sort_index()
    print("shown nodes per level:", cnt.to_dict(), "| edges:", len(out_edges),
          "| hidden target nodes:", sum(1 for n in out_nodes if n.get("extra")))
    print(f"Saved {OUT}/index.html and {OUT}/tree_data.json")


if __name__ == "__main__":
    main()
