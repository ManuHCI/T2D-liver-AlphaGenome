"""
Step 4 - Build the diabetic hepatocyte tree (Level 0 -> Level 5) and find convergence.

Levels
  Level 0  variant + its mechanism (TF-specific binding change / enhancer opened or closed)
  Level 1  liver genes changed in cis by the variant (AlphaGenome, step 3 roots)
  Level 2-5  network spread from Level-1 genes:
             TF  -> its target genes          (CollecTRI, signed: activates / represses)
             protein -> downstream protein    (OmniPath signalling, directed and signed)

Rules (so the tree is biology, not a hairball)
  - only nodes expressed in liver (GTEx liver median TPM >= 1)
  - sign carried along every path (activation x repression = repression ...)
  - each step weakens the signal (decay 0.5) and hubs are down-weighted
    (edge weight = sign / sqrt(out-degree x in-degree))
  - CONVERGENCE = a node that receives the same-direction push from several Level-1 genes
  - PERMUTATION TEST: random Level-1 gene sets (same size, same up/down signs, matched for
    network degree and liver expression) are pushed through the same network 1,000 times;
    a node is reported only if its diabetic push is bigger than chance (FDR < 0.1)

Normal tree   = the same network in the healthy hepatocyte (baseline GTEx liver expression,
                no push).  Diabetic tree = the predicted change on top of it.

Inputs  results/level0_genes.csv, level0_gene_variant_links.csv, level0_tf_binding.csv,
        level0_enhancers.csv  (written by step 3)
Downloads (once, cached in data/): CollecTRI + OmniPath from omnipathdb.org, GTEx liver TPM
Output  results/T2D_tree.xlsx  and  results/tree_nodes.csv, tree_edges.csv, tree_paths.csv
Run     python 04_build_tree.py            (options: --depth 4 --perm 1000 --decay 0.5)
"""
import argparse
import gzip
import io
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
import requests
from scipy import sparse

RES = "results"
DATA = "data"
OMNI = "https://omnipathdb.org/interactions?datasets={ds}&genesymbols=yes"
GTEX = ("https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
        "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz")


# ------------------------------------------------------------------ downloads
def download(url, path, binary=False):
    if os.path.exists(path):
        return
    print(f"  downloading {url[:90]} ...")
    try:
        r = requests.get(url, timeout=300)
        r.raise_for_status()
    except Exception as e:
        sys.exit(f"Download failed ({e}).\nOpen this link in a browser and save the file as {path}:\n{url}")
    open(path, "wb").write(r.content)


def load_network():
    out = []
    for ds, kind in [("collectri", "TF->target"), ("omnipath", "signalling")]:
        path = f"{DATA}/net_{ds}.tsv"
        download(OMNI.format(ds=ds), path)
        d = pd.read_csv(path, sep="\t")
        need = {"source_genesymbol", "target_genesymbol", "is_stimulation", "is_inhibition"}
        if not need.issubset(d.columns):
            sys.exit(f"{path}: unexpected columns {list(d.columns)[:12]} - send this to Claude")
        if kind == "signalling" and "consensus_direction" in d.columns:
            d = d[d["consensus_direction"].astype(int) == 1]
        st, inh = d["is_stimulation"].astype(int), d["is_inhibition"].astype(int)
        d["sign"] = np.where((st == 1) & (inh == 0), 1, np.where((inh == 1) & (st == 0), -1, 0))
        d = d[d["sign"] != 0]
        d = d.drop(columns=[c for c in ("source", "target") if c in d.columns])   # UniProt ids
        d = d.rename(columns={"source_genesymbol": "source", "target_genesymbol": "target"})
        d["edge_type"] = kind
        # complexes are written "A_B": split into members
        d = d.assign(source=d["source"].str.split("_")).explode("source")
        d = d.assign(target=d["target"].str.split("_")).explode("target")
        out.append(d[["source", "target", "sign", "edge_type"]])
        print(f"  {ds}: {len(d):,} signed edges")
    net = pd.concat(out).drop_duplicates()
    # one edge per pair; drop pairs whose sources disagree on the sign
    agg = net.groupby(["source", "target"]).agg(sign=("sign", "mean"),
                                                edge_type=("edge_type", lambda x: "+".join(sorted(set(x)))))
    agg = agg[agg["sign"].abs() == 1].reset_index()
    agg["sign"] = agg["sign"].astype(int)
    agg = agg[agg["source"] != agg["target"]]
    # literature-verified corrections of database edges (see results/T2D_tree_literature_check.docx)
    corr_path = f"{DATA}/edge_corrections.csv"
    if os.path.exists(corr_path):
        corr = pd.read_csv(corr_path)
        for r in corr.itertuples():
            m = (agg["source"] == r.source) & (agg["target"] == r.target)
            if r.sign == 0:
                agg = agg[~m]
            elif m.any():
                agg.loc[m, "sign"] = int(r.sign)
            else:
                agg = pd.concat([agg, pd.DataFrame([{"source": r.source, "target": r.target,
                                                     "sign": int(r.sign), "edge_type": "literature"}])])
        print(f"  applied {len(corr)} literature corrections from {corr_path}")
    return agg


def load_liver_tpm():
    path = f"{DATA}/gtex_median_tpm.gct.gz"
    download(GTEX, path, binary=True)
    with gzip.open(path, "rt") as f:
        d = pd.read_csv(f, sep="\t", skiprows=2)
    d = d.groupby("Description")["Liver"].max()
    print(f"  GTEx liver: {len(d):,} genes, {int((d >= 1).sum()):,} with TPM >= 1")
    return d


# ------------------------------------------------------------------ propagation
def build_matrix(edges, nodes):
    idx = {g: i for i, g in enumerate(nodes)}
    s = edges["source"].map(idx).values
    t = edges["target"].map(idx).values
    outdeg = np.bincount(s, minlength=len(nodes))
    indeg = np.bincount(t, minlength=len(nodes))
    w = edges["sign"].values / np.sqrt(outdeg[s] * indeg[t])
    W = sparse.csr_matrix((w, (s, t)), shape=(len(nodes), len(nodes)))
    return W, outdeg, idx


def propagate(X0, W, depth, decay):
    """X0: roots x nodes. returns total contribution (roots x nodes) and first level reached."""
    X = sparse.csr_matrix(X0)
    total = sparse.csr_matrix(X.shape)
    reach = (abs(X) > 0).astype(np.int8)
    first = np.full(X.shape[1], -1)
    for k in range(1, depth + 1):
        X = decay * (X @ W)
        X.data[np.abs(X.data) < 1e-12] = 0
        X.eliminate_zeros()
        total = total + X
        newly = np.asarray((abs(X) > 0).sum(axis=0)).ravel() > 0
        first[(first < 0) & newly] = k
    return total.toarray(), first


def robust_push(C):
    """net push after removing the single largest contributor at each node.
    A node fed by only one Level-1 gene scores ~0 -> tests CONVERGENCE, not proximity."""
    if C.shape[0] < 2:
        return np.zeros(C.shape[1])
    top = np.abs(C).argmax(axis=0)
    return C.sum(axis=0) - C[top, np.arange(C.shape[1])]


def bfs_path(adj, src, dst, depth):
    q, prev = deque([(src, 0)]), {src: None}
    while q:
        u, d = q.popleft()
        if u == dst:
            path = [u]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return path[::-1]
        if d < depth:
            for v in adj.get(u, ()):
                if v not in prev:
                    prev[v] = u
                    q.append((v, d + 1))
    return None


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=4, help="network steps after Level 1 (Level 2..5)")
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--min-tpm", type=float, default=1.0)
    ap.add_argument("--fdr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--res", default="results", help="results folder (e.g. results_hybrid)")
    ap.add_argument("--explore", type=int, default=30,
                    help="also trace paths for the top-N nodes by p-value (labelled exploratory)")
    a = ap.parse_args()
    global RES
    RES = a.res
    rng = np.random.default_rng(a.seed)

    print("Loading networks and liver expression")
    net = load_network()
    tpm = load_liver_tpm()
    liver = set(tpm[tpm >= a.min_tpm].index)
    net = net[net["source"].isin(liver) & net["target"].isin(liver)]
    nodes = sorted(set(net["source"]) | set(net["target"]))
    W, outdeg, idx = build_matrix(net, nodes)
    print(f"  liver network: {len(nodes):,} genes, {len(net):,} edges "
          f"({(net['edge_type'].str.contains('TF')).sum():,} TF->target)")

    # ---------------- Level 0 / Level 1
    g = pd.read_csv(f"{RES}/level0_genes.csv")
    links = pd.read_csv(f"{RES}/level0_gene_variant_links.csv")
    tfb = pd.read_csv(f"{RES}/level0_tf_binding.csv")
    enh = pd.read_csv(f"{RES}/level0_enhancers.csv")
    links = links[links["gene_name"].isin(g["gene_name"])]
    mech = []
    for _, r in links.iterrows():
        tf = tfb[tfb["variant_id"] == r["variant_id"]]
        tf_txt = ", ".join(f"{t}({'gain' if s > 0 else 'loss'})"
                           for t, s in zip(tf["transcription_factor"], tf["hepatic_score"]))
        e = enh[(enh["variant_id"] == r["variant_id"]) & enh["enhancer_level"]]
        mech.append({"gene": r["gene_name"], "variant": r["rsid"], "pip": r["pip"],
                     "gene_change": r["hepatic_score"],
                     "enhancer": e["element"].iloc[0] if len(e) else "",
                     "TF_binding_change": tf_txt})
    level0 = pd.DataFrame(mech)

    g["sign"] = np.sign(g["weighted_score"]).astype(int)
    g["conf_w"] = np.where(g["n_high_conf"] > 0, 1.0, 0.6)
    g["root_value"] = g["sign"] * g["conf_w"]
    g["in_network"] = g["gene_name"].isin(idx)
    g["liver_TPM"] = g["gene_name"].map(tpm)
    roots = g[g["in_network"]].reset_index(drop=True)
    print(f"\nLevel-1 genes: {len(g)}  (in liver network: {len(roots)}, "
          f"with downstream edges: {int((outdeg[[idx[x] for x in roots['gene_name']]] > 0).sum())})")
    if roots.empty:
        sys.exit("No Level-1 gene is in the network.")

    X0 = np.zeros((len(roots), len(nodes)))
    for i, r in roots.iterrows():
        X0[i, idx[r["gene_name"]]] = r["root_value"]
    C, first = propagate(X0, W, a.depth, a.decay)
    net_push = C.sum(axis=0)
    conv_stat = robust_push(C)
    abs_sum = np.abs(C).sum(axis=0)
    n_roots = (np.abs(C) > 1e-9).sum(axis=0)
    agree = np.divide(np.abs(net_push), abs_sum, out=np.zeros_like(net_push), where=abs_sum > 0)

    # ---------------- permutation null (matched on out-degree and liver expression)
    tpm_n = np.array([tpm.get(n, 0) for n in nodes])
    deg_bin = np.digitize(outdeg, [1, 3, 11, 51])
    exp_bin = pd.qcut(pd.Series(tpm_n).rank(method="first"), 5, labels=False).values
    strata = deg_bin * 10 + exp_bin
    pool = {s: np.where(strata == s)[0] for s in np.unique(strata)}
    root_idx = np.array([idx[x] for x in roots["gene_name"]])
    root_val = roots["root_value"].values
    print(f"Permutation test: {a.perm} random gene sets ...")
    null = np.zeros((a.perm, len(nodes)), dtype=np.float32)   # |convergence push| under the null
    breadth_exceed = np.zeros(len(nodes))
    for p in range(a.perm):
        pick = np.array([rng.choice(pool[strata[i]]) for i in root_idx])
        Xr = sparse.csr_matrix((root_val, (np.arange(len(pick)), pick)), shape=(len(pick), len(nodes)))
        Cr, _ = propagate(Xr, W, a.depth, a.decay)
        null[p] = np.abs(robust_push(Cr))
        breadth_exceed += (np.abs(Cr) > 1e-9).sum(axis=0) >= n_roots
        if (p + 1) % 200 == 0:
            print(f"  {p + 1}/{a.perm}")
    # empirical p against a POOLED null: all nodes with a similar number of incoming edges,
    # all permutations (fine resolution, no bell-curve assumption)
    indeg = np.diff(W.tocsc().indptr)
    in_bin = np.digitize(indeg, [2, 4, 8, 16, 32, 64])
    obs = np.abs(conv_stat)
    p_push = np.ones(len(nodes))
    for b in np.unique(in_bin):
        cols = np.where(in_bin == b)[0]
        pooled = np.sort(null[:, cols].ravel())
        n_ge = len(pooled) - np.searchsorted(pooled, obs[cols] - 1e-15, side="left")
        p_push[cols] = (n_ge + 1) / (len(pooled) + 1)
    p_push[obs < 1e-12] = 1.0
    p_breadth = (breadth_exceed + 1) / (a.perm + 1)

    nd = pd.DataFrame({"gene": nodes, "level": np.where(first > 0, first + 1, -1),
                       "net_push": net_push, "convergence_push": conv_stat, "n_level1_genes": n_roots, "agreement": agree,
                       "p_push": p_push, "p_breadth": p_breadth, "liver_TPM_normal": tpm_n})
    for i, r in roots.iterrows():                     # Level-1 genes themselves
        j = idx[r["gene_name"]]
        nd.loc[j, "level"] = 1
    nd = nd[nd["level"] > 0].copy()
    nd["direction"] = np.where(nd["net_push"] > 0, "UP in diabetic", "DOWN in diabetic")
    nd["is_TF"] = nd["gene"].isin(net.loc[net["edge_type"].str.contains("TF"), "source"])
    reached = nd["level"] > 1
    nd.loc[reached, "fdr_push"] = bh(nd.loc[reached, "p_push"])
    nd.loc[reached, "fdr_breadth"] = bh(nd.loc[reached, "p_breadth"])
    nd["convergence"] = (reached & (nd["fdr_push"] < a.fdr) & (nd["n_level1_genes"] >= 2)
                         & (nd["agreement"] >= 0.5)
                         & (np.sign(nd["convergence_push"]) == np.sign(nd["net_push"])))
    nd = nd.sort_values(["convergence", "net_push"], key=lambda c: c.abs() if c.name == "net_push" else c,
                        ascending=False)

    # ---------------- paths from Level-1 genes to each convergence node
    adj = net.groupby("source")["target"].apply(list).to_dict()
    sign = {(s, t): v for s, t, v in net[["source", "target", "sign"]].itertuples(index=False)}
    rows, edges = [], set()
    cand = nd[(nd["level"] > 1) & (nd["n_level1_genes"] >= 2)]
    explore = cand[~cand["convergence"]].sort_values("p_push").head(a.explore)
    nd["exploratory"] = nd["gene"].isin(explore["gene"])
    for _, r in pd.concat([nd[nd["convergence"]], explore]).iterrows():
        j = idx[r["gene"]]
        contrib = C[:, j]
        for i in np.argsort(-np.abs(contrib))[:5]:
            if abs(contrib[i]) < 1e-12:
                continue
            path = bfs_path(adj, roots.loc[i, "gene_name"], r["gene"], a.depth)
            if not path:
                continue
            txt = path[0]
            for u, v in zip(path[:-1], path[1:]):
                txt += f" {'-->' if sign[(u, v)] > 0 else '--|'} {v}"
                edges.add((u, v, sign[(u, v)]))
            rows.append({"convergence_node": r["gene"],
                         "status": "significant" if r["convergence"] else "exploratory",
                         "p_push": r["p_push"], "fdr_push": r["fdr_push"], "from_level1_gene": path[0],
                         "level1_direction": "UP" if roots.loc[i, "sign"] > 0 else "DOWN",
                         "contribution": contrib[i], "path": txt})
    paths = pd.DataFrame(rows)
    ed = pd.DataFrame(sorted(edges), columns=["parent", "child", "sign"]) if edges else \
        pd.DataFrame(columns=["parent", "child", "sign"])
    ed["effect"] = np.where(ed["sign"] > 0, "activates", "inhibits")

    # ---------------- save
    os.makedirs(RES, exist_ok=True)
    nd.to_csv(f"{RES}/tree_nodes.csv", index=False)
    ed.to_csv(f"{RES}/tree_edges.csv", index=False)
    paths.to_csv(f"{RES}/tree_paths.csv", index=False)
    level0.to_csv(f"{RES}/tree_level0_mechanisms.csv", index=False)
    with pd.ExcelWriter(f"{RES}/T2D_tree.xlsx") as xw:
        level0.to_excel(xw, sheet_name="L0_variant_mechanism", index=False)
        g.to_excel(xw, sheet_name="L1_genes", index=False)
        nd[nd["convergence"] | nd["exploratory"]].to_excel(xw, sheet_name="convergence_nodes", index=False)
        paths.to_excel(xw, sheet_name="paths", index=False)
        ed.to_excel(xw, sheet_name="tree_edges", index=False)
        nd.head(5000).to_excel(xw, sheet_name="all_reached_nodes", index=False)

    conv = nd[nd["convergence"]]
    print("\n=== DIABETIC HEPATOCYTE TREE ===")
    print(f"Level 1 genes (AlphaGenome)      : {len(g)}")
    for L in range(2, a.depth + 2):
        print(f"Level {L} genes reached            : {(nd['level'] == L).sum():,}")
    print(f"Convergence nodes (FDR < {a.fdr}, >= 2 Level-1 genes, same direction): {len(conv)} "
          f"(UP {(conv['net_push'] > 0).sum()}, DOWN {(conv['net_push'] < 0).sum()})")
    print(f"\nTop 25 convergence nodes:\n"
          f"{conv.head(25)[['gene', 'level', 'direction', 'net_push', 'convergence_push', 'n_level1_genes', 'agreement', 'fdr_push', 'is_TF']].round(4).to_string(index=False)}")
    ex = nd[nd["exploratory"]]
    print(f"\nExploratory (top {len(ex)} by p, not significant after FDR):\n"
          f"{ex.head(15)[['gene', 'level', 'direction', 'net_push', 'n_level1_genes', 'agreement', 'p_push', 'fdr_push', 'is_TF']].round(5).to_string(index=False)}")
    if len(paths):
        print(f"\nExample paths:\n{paths.head(12)[['path', 'level1_direction']].to_string(index=False)}")
    print(f"\nSaved: {RES}/T2D_tree.xlsx")


if __name__ == "__main__":
    main()
