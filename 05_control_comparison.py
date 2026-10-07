"""
Step 5 - CONTROL COMPARISON: does the AlphaGenome tree beat simpler trees on real data?

Three trees, built from the same T2D variants, the same liver network and the same method;
only the Level 0 -> Level 1 step differs:
  A  AlphaGenome tree   : gene AND direction (up/down) predicted by AlphaGenome
  B  nearest-gene tree  : the nearest liver-expressed protein-coding gene to each variant
                          (the usual GWAS approach; gives a gene but no direction)
  C  shuffled tree      : same genes as A, directions randomly flipped (1,000 times)

Yardsticks (real data, never used to build the trees)
  1. Human liver biopsies, T2D vs non-diabetic (GEO: GSE23343, GSE15653 by default)
       DIRECTION test   A vs C : do A's predicted up/down genes move that way in diabetic liver?
       GENE-CHOICE test A vs B : do A's strongly-pushed genes overlap the genes that really
                                 change in diabetic liver more than B's do?   (AUC)
  2. Known diabetes genes from Open Targets, independent of common GWAS variants:
       "Mendelian/rare-variant" genes and "approved-drug target" genes
       GENE-CHOICE test A vs B : enrichment among each tree's top-ranked genes

Run in two steps (on your machine - needs internet):
  python 05_control_comparison.py --fetch     downloads data, prints sample groups to CHECK
  python 05_control_comparison.py --run       builds trees B and C, scores A, B, C
Labels: data/geo/<GSE>_labels.csv is written by --fetch. Open it, check the "group" column
(T2D / control / exclude), correct if needed, then run --run.
"""
import argparse
import warnings
warnings.filterwarnings("ignore")
import gzip
import importlib.util
import io
import os
import re
import sys

import numpy as np
import pandas as pd
import requests
from scipy import sparse, stats

RES, DATA, GEO = "results", "data", "data/geo"
GSE_DEFAULT = ["GSE23343", "GSE15653"]
GENCODE = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/"
           "gencode.v46.basic.annotation.gtf.gz")
OT_API = "https://api.platform.opentargets.org/api/v4/graphql"
DISEASES = {"MONDO_0005148": "type 2 diabetes", "MONDO_0005015": "diabetes mellitus"}
MENDELIAN_SOURCES = {"gene_burden", "eva", "genomics_england", "gene2phenotype",
                     "orphanet", "clingen", "uniprot_literature", "uniprot_variants"}
DRUG_SOURCES = {"chembl", "clinical_precedence"}   # Open Targets renamed ChEMBL drug evidence

# reuse the exact tree code from step 4
_spec = importlib.util.spec_from_file_location("tree", "04_build_tree.py")
tree = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tree)


# ============================================================== downloads
def get(url, path):
    if os.path.exists(path):
        return path
    print(f"  downloading {url}")
    r = requests.get(url, timeout=600)
    if r.status_code != 200:
        return None
    open(path, "wb").write(r.content)
    return path


def geo_dir(acc):            # GSE23343 -> GSE23nnn ; GPL570 -> GPLnnn
    p, n = re.match(r"(GSE|GPL)(\d+)", acc).groups()
    return f"{p}{n[:-3]}nnn"


def parse_series_matrix(path):
    meta, chars, rows, in_table = {}, [], [], False
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                rows.append(line.rstrip("\n").split("\t"))
                continue
            parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
            key = parts[0]
            if key == "!Sample_characteristics_ch1":
                chars.append(parts[1:])
            elif key.startswith("!Sample_") or key.startswith("!Series_"):
                meta.setdefault(key, parts[1:])
    expr = pd.DataFrame(rows[1:], columns=[c.strip('"') for c in rows[0]])
    expr = expr.rename(columns={expr.columns[0]: "ID"})
    expr["ID"] = expr["ID"].str.strip('"')
    expr = expr.set_index("ID").apply(pd.to_numeric, errors="coerce")
    samples = pd.DataFrame({"sample": meta.get("!Sample_geo_accession", []),
                            "title": meta.get("!Sample_title", []),
                            "source": meta.get("!Sample_source_name_ch1", [])})
    for i, c in enumerate(chars):
        samples[f"char{i + 1}"] = c[:len(samples)]
    return meta, samples, expr


def platform_symbols(gpl):
    d = geo_dir(gpl)
    p = get(f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{d}/{gpl}/annot/{gpl}.annot.gz", f"{GEO}/{gpl}.annot.gz")
    if p:
        txt = gzip.open(p, "rt", errors="replace").read()
        start = txt.find("!platform_table_begin")
        tab = pd.read_csv(io.StringIO(txt[start:].split("\n", 1)[1].split("!platform_table_end")[0]), sep="\t",
                          dtype=str)
        col = next(c for c in tab.columns if c.lower() in ("gene symbol", "gene_symbol", "symbol"))
        return tab.set_index("ID")[col].dropna()
    p = get(f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{d}/{gpl}/soft/{gpl}_family.soft.gz",
            f"{GEO}/{gpl}_family.soft.gz")
    if not p:
        sys.exit(f"No annotation for platform {gpl}")
    txt = gzip.open(p, "rt", errors="replace").read()
    start = txt.find("!platform_table_begin")
    tab = pd.read_csv(io.StringIO(txt[start:].split("\n", 1)[1].split("!platform_table_end")[0]), sep="\t",
                      dtype=str)
    col = next((c for c in tab.columns if re.fullmatch(r"(gene[ _]?symbol|symbol)", c, re.I)), None)
    if col is None:
        sys.exit(f"{gpl}: no gene-symbol column in {list(tab.columns)[:15]} - send this to Claude")
    return tab.set_index("ID")[col].dropna()


NON_RX = re.compile(r"(non.?diabet|without diabet|no.?dm(?![a-z])|non.?dm(?![a-z])|non.?t2d|\bngt\b|normal glucose|"
                    r"normoglyc|healthy|(?<![-\w])control\b)", re.I)       # "well-controlled" is NOT a control
T2D_RX = re.compile(r"(type ?2 diabet|\bt2dm?\b|_dm_|\bdm\b|diabetic|diabetes|hyperglyc)", re.I)
EXCLUDE_RX = re.compile(r"\blean\b", re.I)    # GSE15653: compare obese-T2D vs obese-non-diabetic (BMI-matched)


def auto_group(row):
    txt = " ".join(str(v) for k, v in row.items() if k != "sample")
    if EXCLUDE_RX.search(txt):
        return "exclude"
    if NON_RX.search(txt):
        return "control"
    if T2D_RX.search(txt):
        return "T2D"
    return "exclude"


def fetch(gses):
    os.makedirs(GEO, exist_ok=True)
    for g in gses:
        print(f"\n=== {g} ===")
        p = get(f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_dir(g)}/{g}/matrix/{g}_series_matrix.txt.gz",
                f"{GEO}/{g}_series_matrix.txt.gz")
        if not p:
            print("  not found (maybe several platforms: check the GEO page)")
            continue
        meta, samples, expr = parse_series_matrix(p)
        print("  title   :", (meta.get("!Series_title") or ["?"])[0])
        print("  platform:", (meta.get("!Series_platform_id") or ["?"])[0], f"| {expr.shape[0]:,} probes x {expr.shape[1]} samples")
        samples["group"] = samples.apply(auto_group, axis=1)
        lab = f"{GEO}/{g}_labels.csv"
        if not os.path.exists(lab):
            samples.to_csv(lab, index=False)
        print(samples.drop(columns=["sample"]).head(40).to_string(index=False, max_colwidth=45))
        print(f"  groups: {samples['group'].value_counts().to_dict()}   -> CHECK {lab}")
    get(GENCODE, f"{DATA}/gencode.basic.gtf.gz")
    get_gold()
    print("\nNext: open each data/geo/*_labels.csv, fix the 'group' column if needed, then run --run")


# ============================================================== gold-standard genes
Q_ASSOC = """query($id:String!,$i:Int!){ disease(efoId:$id){ associatedTargets(page:{index:$i,size:200}){
  count rows{ target{ approvedSymbol } datasourceScores{ id score } } } } }"""


Q_DRUGS = """query($id:String!){ disease(efoId:$id){ knownDrugs(size:1000){ count
  rows{ approvedSymbol prefName phase } } } }"""


def get_gold():
    path = f"{DATA}/gold_diabetes_genes.csv"
    if os.path.exists(path):
        g = pd.read_csv(path)
        if g["drug_target"].sum() > 0:
            return g
        print("  cached gold list has no drug targets - fetching again")
    rows, seen_ids = [], set()
    for dz in DISEASES:
        i = 0
        while True:
            r = requests.post(OT_API, json={"query": Q_ASSOC, "variables": {"id": dz, "i": i}}, timeout=120)
            js = r.json()
            if "errors" in js or r.status_code != 200:
                print("  Open Targets error:", js.get("errors", r.text[:300]))
                break
            at = js["data"]["disease"]["associatedTargets"]
            for t in at["rows"]:
                ids = {d["id"] for d in t["datasourceScores"] if d["score"] > 0}
                seen_ids |= ids
                rows.append({"gene": t["target"]["approvedSymbol"], "disease": dz,
                             "mendelian": bool(ids & MENDELIAN_SOURCES), "drug_target": bool(ids & DRUG_SOURCES)})
            i += 1
            if i * 200 >= at["count"] or not at["rows"]:
                break
    # drug targets: diabetes drugs that reached clinical trials (phase >= 2), from Open Targets/ChEMBL
    for dz in DISEASES:
        r = requests.post(OT_API, json={"query": Q_DRUGS, "variables": {"id": dz}}, timeout=120)
        js = r.json()
        if "errors" in js or r.status_code != 200:
            print("  knownDrugs error:", js.get("errors", r.text[:300]))
            continue
        kd = js["data"]["disease"]["knownDrugs"]["rows"]
        for k in kd:
            if k.get("approvedSymbol") and (k.get("phase") or 0) >= 2:
                rows.append({"gene": k["approvedSymbol"], "disease": dz, "mendelian": False, "drug_target": True})
    print(f"  Open Targets evidence types seen: {sorted(seen_ids)}")
    g = pd.DataFrame(rows).groupby("gene")[["mendelian", "drug_target"]].any().reset_index()
    g.to_csv(path, index=False)
    print("  drug-target genes:", ", ".join(sorted(g.loc[g["drug_target"], "gene"])[:40]))
    print(f"  gold genes: Mendelian/rare {int(g['mendelian'].sum())}, drug targets {int(g['drug_target'].sum())}")
    return g


# ============================================================== nearest gene (tree B)
def load_tss():
    rows = []
    with gzip.open(f"{DATA}/gencode.basic.gtf.gz", "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.split("\t", 8)
            if c[2] != "gene" or 'gene_type "protein_coding"' not in c[8]:
                continue
            name = re.search(r'gene_name "([^"]+)"', c[8]).group(1)
            tss = int(c[3]) if c[6] == "+" else int(c[4])
            rows.append((c[0], tss, name))
    return pd.DataFrame(rows, columns=["chrom", "tss", "gene"])


def nearest_genes(var, tss, allowed):
    t = tss[tss["gene"].isin(allowed)]
    out = []
    for _, v in var.iterrows():
        sub = t[t["chrom"] == v["chrom"]]
        if sub.empty:
            continue
        j = (sub["tss"] - v["pos"]).abs().idxmin()
        out.append({"variant_id": v["variant_id"], "gene": sub.loc[j, "gene"],
                    "distance": abs(int(sub.loc[j, "tss"]) - int(v["pos"]))})
    return pd.DataFrame(out)


# ============================================================== yardstick: diabetic vs normal liver
def de_table(g):
    meta, samples, expr = parse_series_matrix(f"{GEO}/{g}_series_matrix.txt.gz")
    lab = pd.read_csv(f"{GEO}/{g}_labels.csv")
    gpl = (meta.get("!Series_platform_id") or [None])[0]
    sym = platform_symbols(gpl)
    if np.nanmax(expr.values) > 100:              # not yet log-scale
        expr = np.log2(expr.clip(lower=1))
    expr = expr.join(sym.rename("symbol"), how="inner")
    expr["symbol"] = expr["symbol"].str.split(" /// ").str[0]
    gexp = expr.groupby("symbol").mean()          # mean of probes per gene
    t2d = lab.loc[lab["group"] == "T2D", "sample"]
    ctl = lab.loc[lab["group"] == "control", "sample"]
    t2d, ctl = [s for s in t2d if s in gexp.columns], [s for s in ctl if s in gexp.columns]
    if len(t2d) < 3 or len(ctl) < 3:
        print(f"  {g}: only {len(t2d)} T2D / {len(ctl)} control samples - skipped (fix labels?)")
        return None
    tt, pp = stats.ttest_ind(gexp[t2d], gexp[ctl], axis=1, equal_var=False, nan_policy="omit")
    de = pd.DataFrame({"gene": gexp.index, "log2FC": gexp[t2d].mean(1) - gexp[ctl].mean(1),
                       "t": tt, "p": pp}).dropna()
    print(f"  {g}: {len(t2d)} T2D vs {len(ctl)} control, {len(de):,} genes, "
          f"{int((de['p'] < 0.05).sum()):,} with p < 0.05")
    return de.set_index("gene")


# ============================================================== scoring
def push_vector(roots, idx, W, depth, decay, n):
    X0 = np.zeros((len(roots), n))
    for i, (g, v) in enumerate(roots):
        X0[i, idx[g]] = v
    C, _ = tree.propagate(X0, W, depth, decay)
    return C.sum(axis=0)


def direction_score(push, nodes, de, top=None, p_cut=0.05):
    """fraction of tree genes (changed in real data) whose predicted sign matches; |push|-weighted."""
    d = pd.DataFrame({"gene": nodes, "push": push})
    d = d[d["push"].abs() > 1e-9].merge(de, left_on="gene", right_index=True)
    d = d[d["p"] < p_cut]
    if top:
        d = d.reindex(d["push"].abs().sort_values(ascending=False).index).head(top)
    if d.empty:
        return np.nan, 0
    ok = np.sign(d["push"]) == np.sign(d["log2FC"])
    return float(np.average(ok, weights=d["push"].abs())), len(d)


def auc(score, positive):
    """probability a positive gene is ranked above a negative one (ties = 0.5)."""
    s, y = np.asarray(score), np.asarray(positive, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def topk_enrichment(score, positive, k=100):
    order = np.argsort(-np.asarray(score))[:k]
    hit = int(np.asarray(positive)[order].sum())
    exp = k * np.mean(positive)
    return hit, exp


def run(a):
    rng = np.random.default_rng(a.seed)
    print("Loading network, liver expression, roots")
    net = tree.load_network()
    tpm = tree.load_liver_tpm()
    liver = set(tpm[tpm >= a.min_tpm].index)
    net = net[net["source"].isin(liver) & net["target"].isin(liver)]
    net = net[~((net["source"] == "LPA") & (net["target"] == "LPAR2"))]       # name-clash artefact
    nodes = sorted(set(net["source"]) | set(net["target"]))
    W, outdeg, idx = tree.build_matrix(net, nodes)
    Wabs = abs(W)
    n = len(nodes)

    # ---- tree A (AlphaGenome) ----
    g = pd.read_csv(f"{RES}/level0_genes.csv")
    links = pd.read_csv(f"{RES}/level0_gene_variant_links.csv")
    g = g[g["gene_name"].isin(idx)]
    g["root_value"] = np.sign(g["weighted_score"]) * np.where(g["n_high_conf"] > 0, 1.0, 0.6)
    rootsA = list(zip(g["gene_name"], g["root_value"]))
    pushA = push_vector(rootsA, idx, W, a.depth, a.decay, n)
    absA = push_vector([(x, abs(v)) for x, v in rootsA], idx, Wabs, a.depth, a.decay, n)

    # ---- tree B (nearest gene) - same variants that gave tree A its genes ----
    var = pd.read_csv(f"{DATA}/t2d_variants_for_alphagenome.csv")
    used = var[var["variant_id"].isin(links["variant_id"])]
    tss = load_tss()
    nb = nearest_genes(used, tss, liver)
    nb_all = nearest_genes(var, tss, liver)
    rootsB = [(x, 1.0) for x in sorted(set(nb["gene"]) & set(idx))]
    absB = push_vector(rootsB, idx, Wabs, a.depth, a.decay, n)
    rootsB_all = [(x, 1.0) for x in sorted(set(nb_all["gene"]) & set(idx))]
    absB_all = push_vector(rootsB_all, idx, Wabs, a.depth, a.decay, n)
    overlap = len(set(x for x, _ in rootsA) & set(x for x, _ in rootsB))
    print(f"  tree A roots: {len(rootsA)} | tree B roots (same {len(used)} variants): {len(rootsB)} | "
          f"overlap {overlap} | tree B (all {len(var)} variants): {len(rootsB_all)}")

    # ---- yardstick 1: diabetic liver transcriptomes ----
    results = []
    for gse in a.gse:
        if not os.path.exists(f"{GEO}/{gse}_labels.csv"):
            print(f"  {gse}: no labels file - run --fetch first")
            continue
        de = de_table(gse)
        if de is None:
            continue
        # DIRECTION: A vs C (shuffled root signs, same genes)
        for top in (None, 50):
            obs, m = direction_score(pushA, nodes, de, top)
            null = []
            for _ in range(a.perm):
                sh = [(x, v * rng.choice([-1, 1])) for x, v in rootsA]
                null.append(direction_score(push_vector(sh, idx, W, a.depth, a.decay, n), nodes, de, top)[0])
            null = np.array(null, float)
            p = (np.sum(null >= obs) + 1) / (np.sum(~np.isnan(null)) + 1)
            results.append({"yardstick": gse, "test": f"DIRECTION {'top-50' if top else 'all DE'} genes",
                            "A_AlphaGenome": obs, "B_nearest_gene": np.nan,
                            "C_shuffled_mean": np.nanmean(null), "n_genes": m, "p_A_vs_C": p})
        # GENE CHOICE: A vs B (AUC: |push| ranks the genes that change in diabetic liver)
        # only genes actually measured in this dataset
        meas = np.array([x in de.index for x in nodes])
        inde = np.array([x in de.index and de.loc[x, "p"] < 0.05 for x in nodes])[meas]
        sA, sB, sBall = absA[meas], absB[meas], absB_all[meas]
        aA, aB, aBall = auc(sA, inde), auc(sB, inde), auc(sBall, inde)
        # paired bootstrap over genes: is A better than B?
        diffs = []
        for _ in range(500):
            b = rng.integers(0, len(inde), len(inde))
            diffs.append(auc(sA[b], inde[b]) - auc(sB[b], inde[b]))
        diffs = np.array(diffs)
        p_AB = (np.sum(diffs <= 0) + 1) / (len(diffs) + 1)
        null = []
        pool = np.array(nodes)
        for _ in range(a.perm // 4):                     # random root sets, same size as A
            pick = rng.choice(pool, len(rootsA), replace=False)
            null.append(auc(push_vector([(x, 1.0) for x in pick], idx, Wabs, a.depth, a.decay, n)[meas], inde))
        null = np.array(null)
        results.append({"yardstick": gse, "test": "GENE CHOICE AUC (DE genes in diabetic liver)",
                        "A_AlphaGenome": aA, "B_nearest_gene": aB, "B_nearest_all_variants": aBall,
                        "random_mean": null.mean(), "n_genes": int(inde.sum()),
                        "p_A_vs_random": (np.sum(null >= aA) + 1) / (len(null) + 1),
                        "p_A_vs_B": p_AB, "A_minus_B_95CI": f"{np.percentile(diffs, 2.5):.3f} to {np.percentile(diffs, 97.5):.3f}"})

    # ---- yardstick 2: known diabetes genes ----
    gold = get_gold()
    tpm_n = np.array([tpm.get(x, 0) for x in nodes])
    strata_m = np.digitize(outdeg, [1, 3, 11, 51]) * 10 + \
        pd.qcut(pd.Series(tpm_n).rank(method="first"), 5, labels=False).values
    pool_m = {s_: np.where(strata_m == s_)[0] for s_ in np.unique(strata_m)}
    for col, label in [("mendelian", "Mendelian / rare-variant diabetes genes"),
                       ("drug_target", "approved-drug targets for diabetes")]:
        pos = np.isin(nodes, gold.loc[gold[col], "gene"])
        # STRICT test: Level-1 genes themselves excluded; null = random trees started from genes
        # matched for network connectivity and liver expression (1,000 draws)
        def topk_down(push, root_idx, k=100):
            sc = push.copy()
            sc[list(root_idx)] = -np.inf
            return int(pos[np.argsort(-sc)[:k]].sum())
        ridxA = [idx[x] for x, _ in rootsA]
        hA = topk_down(absA, ridxA)
        hB = topk_down(absB, [idx[x] for x, _ in rootsB])
        hBa = topk_down(absB_all, [idx[x] for x, _ in rootsB_all])
        null = []
        for _ in range(a.perm):
            pick = [rng.choice(pool_m[strata_m[i]]) for i in ridxA]
            null.append(topk_down(push_vector([(nodes[j], 1.0) for j in pick], idx, Wabs, a.depth, a.decay, n), pick))
        null = np.array(null)
        results.append({"yardstick": "Open Targets", "test": f"GENE CHOICE: {label}, top-100 downstream (roots excluded)",
                        "A_AlphaGenome": hA, "B_nearest_gene": hB, "B_nearest_all_variants": hBa,
                        "random_mean": round(null.mean(), 1), "n_genes": int(pos.sum()),
                        "p_A_vs_random": (np.sum(null >= hA) + 1) / (len(null) + 1),
                        "level1_genes_in_list": int(pos[ridxA].sum())})
        diffs = []
        for _ in range(500):
            b = rng.integers(0, n, n)
            diffs.append(auc(absA[b], pos[b]) - auc(absB[b], pos[b]))
        diffs = np.array(diffs, float)
        results.append({"yardstick": "Open Targets", "test": f"GENE CHOICE AUC: {label}",
                        "A_AlphaGenome": auc(absA, pos), "B_nearest_gene": auc(absB, pos),
                        "B_nearest_all_variants": auc(absB_all, pos), "random_mean": 0.5,
                        "n_genes": int(pos.sum()),
                        "p_A_vs_B": ((np.sum(diffs <= 0) + 1) / (np.sum(~np.isnan(diffs)) + 1)
                                     if pos.sum() >= 5 else np.nan),   # too few genes for a p-value
                        "A_minus_B_95CI": f"{np.nanpercentile(diffs, 2.5):.3f} to {np.nanpercentile(diffs, 97.5):.3f}"})

    out = pd.DataFrame(results)
    os.makedirs(RES, exist_ok=True)
    out.to_csv(f"{RES}/control_comparison.csv", index=False)
    nb.to_csv(f"{RES}/treeB_nearest_genes.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== CONTROL COMPARISON ===")
    print(out.round(4).to_string(index=False))
    print("\nHow to read it:")
    print("  DIRECTION      : A > C_shuffled with small p  -> AlphaGenome's up/down predictions carry real information")
    print("  GENE CHOICE    : A > B_nearest_gene          -> AlphaGenome picks more relevant genes than 'nearest gene'")
    print("  AUC 0.5 = chance; top-100 counts vs random_mean = expected by chance")
    try:
        plot(out)
    except Exception as e:
        print(f"(figure skipped: {e})")
    print(f"\nSaved: {RES}/control_comparison.csv")


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = out[out["test"].str.contains("AUC|DIRECTION")].copy()
    d["label"] = d["yardstick"] + "\n" + d["test"].str.replace("GENE CHOICE AUC", "AUC").str.slice(0, 48)
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(d) + 1.5))
    y = np.arange(len(d))
    ax.barh(y - 0.25, d["A_AlphaGenome"], 0.25, label="A  AlphaGenome tree", color="#1f7a5c")
    ax.barh(y, d["B_nearest_gene"].fillna(0), 0.25, label="B  nearest-gene tree", color="#8a96a0")
    ref = d["C_shuffled_mean"].fillna(d.get("random_mean"))
    ax.barh(y + 0.25, ref.astype(float), 0.25, label="C shuffled / random (chance)", color="#d9b38c")
    ax.axvline(0.5, color="#555", lw=0.8, ls="--")
    ax.set_yticks(y, d["label"], fontsize=8)
    ax.set_xlabel("score (0.5 = chance)")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{RES}/control_comparison.png", dpi=200)
    print(f"Figure: {RES}/control_comparison.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--gse", nargs="+", default=GSE_DEFAULT)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--min-tpm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--res", default="results", help="results folder (e.g. results_hybrid)")
    a = ap.parse_args()
    global RES
    RES = a.res
    if a.fetch:
        fetch(a.gse)
    elif a.run:
        run(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
