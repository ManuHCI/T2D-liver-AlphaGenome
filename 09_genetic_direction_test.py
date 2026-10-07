"""
Step 9 - GENETIC DIRECTION TEST: do the tree's predicted directions agree with genetics?

For each liver gene we estimate, from human genetics, whether genetically HIGHER expression of
that gene RAISES or LOWERS type 2 diabetes risk (a Wald-ratio / SMR-style estimate, the
single-variant form of a TWAS):
     genetic direction = sign( beta_T2D(v) / beta_eQTL(v) )
     v = the gene's lead liver eQTL variant (highest PIP, GTEx liver credible sets)
     beta_T2D = effect of the same allele on T2D (FinnGen R12 public summary statistics)

The tree predicts, for each gene, "UP in diabetic" or "DOWN in diabetic".
If the tree is right and the gene lies on the causal path, a gene predicted UP should be one
whose genetically higher expression raises risk (positive Wald ratio), and vice versa.

Test
  concordance = fraction of genes where sign(tree push) == sign(Wald ratio)
  null        = the same tree with Level-1 directions randomly flipped (1,000 times)
  Only genes with a nominal genetic signal (T2D p < 0.05 at their eQTL lead) are counted.
  Genes within 1 Mb of any T2D input variant are EXCLUDED (their signal comes from the same
  GWAS loci that built Level 1 - circular). Level-1 genes are reported separately.

Two commands (on your machine):
  python 09_genetic_direction_test.py --prepare            # downloads FinnGen T2D stats once (~1 GB)
  python 09_genetic_direction_test.py --run --res results_hybrid
Needs: data/eqtl/QTD000266.credible_sets.tsv.gz (step 6), data/gencode.basic.gtf.gz (step 5)
"""
import argparse
import gzip
import importlib.util
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import requests
from scipy import stats

warnings.filterwarnings("ignore")
CS = "data/eqtl/QTD000266.credible_sets.tsv.gz"
GWAS = "data/finngen_R12_T2D.gz"
SMR = "data/smr_liver_t2d.csv"
URLS = ["https://storage.googleapis.com/finngen-public-data-r12/summary_stats/release/finngen_R12_T2D.gz",
        "https://storage.googleapis.com/finngen-public-data-r12/summary_stats/finngen_R12_T2D.gz",
        "https://storage.googleapis.com/finngen-public-data-r11/summary_stats/finngen_R11_T2D.gz",
        "https://storage.googleapis.com/finngen-public-data-r10/summary_stats/finngen_R10_T2D.gz"]


def gene_table():
    rows = []
    with gzip.open("data/gencode.basic.gtf.gz", "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.split("\t", 8)
            if c[2] != "gene":
                continue
            gid = re.search(r'gene_id "([^".]+)', c[8]).group(1)
            nm = re.search(r'gene_name "([^"]+)"', c[8])
            rows.append((gid, nm.group(1) if nm else gid, c[0], int(c[3]), int(c[4])))
    return pd.DataFrame(rows, columns=["gene_id", "gene_name", "chrom", "start", "end"])


# ============================================================ prepare
def download():
    if os.path.exists(GWAS):
        return
    for u in URLS:
        print(f"  trying {u}")
        try:
            with requests.get(u, stream=True, timeout=120) as r:
                if r.status_code != 200:
                    continue
                total = 0
                with open(GWAS + ".part", "wb") as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk)
                        total += len(chunk)
                        if total % (100 << 20) < (1 << 22):
                            print(f"    {total >> 20} MB")
            os.replace(GWAS + ".part", GWAS)
            print(f"  saved {GWAS}")
            return
        except requests.RequestException as e:
            print(f"    failed: {e}")
    sys.exit("Could not download FinnGen T2D summary statistics. Download 'finngen_R12_T2D.gz' from "
             "https://www.finngen.fi/en/access_results and save it as data/finngen_R12_T2D.gz")


def prepare(a):
    if not os.path.exists(CS):
        sys.exit(f"{CS} missing - run 06_liver_eqtl_comparison.py once (it downloads it)")
    cs = pd.read_csv(CS, sep="\t")
    cs["gene_id"] = cs["gene_id"].astype(str).str.split(".").str[0]
    lead = cs.sort_values("pip", ascending=False).drop_duplicates("gene_id")      # lead eQTL per gene
    v = lead["variant"].str.replace("chr", "", regex=False).str.split("_", expand=True)
    lead["c"], lead["pos"], lead["ref"], lead["alt"] = v[0], v[1].astype(int), v[2], v[3]
    lead = lead.rename(columns={"beta": "beta_eqtl"})
    keys = {(c, p) for c, p in zip(lead["c"], lead["pos"])}
    print(f"Liver eQTL lead variants: {len(lead):,} genes")
    download()
    print("Reading FinnGen T2D statistics for those variants ...")
    hits = []
    for ch in pd.read_csv(GWAS, sep="\t", compression="gzip", chunksize=1_000_000,
                          usecols=lambda c: c in ("#chrom", "pos", "ref", "alt", "beta", "sebeta", "pval"),
                          dtype={"#chrom": str}):
        ch["#chrom"] = ch["#chrom"].replace({"23": "X"})
        m = [(c, p) in keys for c, p in zip(ch["#chrom"], ch["pos"])]
        hits.append(ch[m])
    g = pd.concat(hits).rename(columns={"#chrom": "c", "ref": "ref_g", "alt": "alt_g",
                                        "beta": "beta_t2d", "sebeta": "se_t2d", "pval": "p_t2d"})
    m = lead.merge(g, on=["c", "pos"])
    same = (m["ref"] == m["ref_g"]) & (m["alt"] == m["alt_g"])
    swap = (m["ref"] == m["alt_g"]) & (m["alt"] == m["ref_g"])
    m.loc[swap, "beta_t2d"] *= -1
    m = m[same | swap].copy()
    m["wald_ratio"] = m["beta_t2d"] / m["beta_eqtl"]
    m["genetic_direction"] = np.where(m["wald_ratio"] > 0, "higher expression -> MORE T2D",
                                      "higher expression -> LESS T2D")
    out = m[["gene_id", "variant", "pip", "beta_eqtl", "beta_t2d", "se_t2d", "p_t2d", "wald_ratio",
             "genetic_direction"]]
    out.to_csv(SMR, index=False)
    print(f"Genes with a genetic direction estimate: {len(out):,} "
          f"(allele swaps corrected: {int(swap.sum())}); T2D p < 0.05: {(out['p_t2d'] < 0.05).sum():,}")
    print(f"Saved {SMR}\nNext: python 09_genetic_direction_test.py --run --res results_hybrid")


# ============================================================ run
def run(a):
    spec = importlib.util.spec_from_file_location("tree", "04_build_tree.py")
    tree = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tree)
    smr = pd.read_csv(SMR)
    genes = gene_table()
    sym = genes.drop_duplicates("gene_id").set_index("gene_id")["gene_name"]
    smr["gene"] = smr["gene_id"].map(sym)

    # genes near any T2D input variant -> excluded (circular)
    var = pd.read_csv("data/t2d_variants_for_alphagenome.csv")
    near = set()
    for ch, grp in var.groupby("chrom"):
        gg = genes[genes["chrom"] == ch]
        for p in grp["pos"]:
            near |= set(gg.loc[(gg["end"] >= p - a.window) & (gg["start"] <= p + a.window), "gene_name"])

    print("Loading network and Level-1 genes ...")
    net = tree.load_network()
    tpm = tree.load_liver_tpm()
    liver = set(tpm[tpm >= 1].index)
    net = net[net["source"].isin(liver) & net["target"].isin(liver)]
    nodes = sorted(set(net["source"]) | set(net["target"]))
    W, outdeg, idx = tree.build_matrix(net, nodes)
    g1 = pd.read_csv(f"{a.res}/level0_genes.csv")
    g1 = g1[g1["gene_name"].isin(idx)]
    roots = list(zip(g1["gene_name"], np.sign(g1["weighted_score"]) * np.where(g1["n_high_conf"] > 0, 1.0, 0.6)))
    root_set = set(g1["gene_name"])

    def push(rs):
        X0 = np.zeros((len(rs), len(nodes)))
        for i, (gname, v) in enumerate(rs):
            X0[i, idx[gname]] = v
        C, _ = tree.propagate(X0, W, a.depth, a.decay)
        return C.sum(axis=0)

    sig = smr[(smr["p_t2d"] < a.p) & smr["gene"].isin(idx)].drop_duplicates("gene")
    down = sig[~sig["gene"].isin(near) & ~sig["gene"].isin(root_set)]
    lvl1 = smr[smr["gene"].isin(root_set)].drop_duplicates("gene")
    print(f"Genes with genetic direction (T2D p < {a.p}) in the liver network: {len(sig)}; "
          f"after removing genes within {a.window // 1000} kb of T2D variants and Level-1 genes: {len(down)}")

    def concord(p_vec, d, weighted=False):
        pv = np.array([p_vec[idx[gname]] for gname in d["gene"]])
        keep = np.abs(pv) > 1e-12
        if keep.sum() == 0:
            return np.nan, 0
        ok = np.sign(pv[keep]) == np.sign(d["wald_ratio"].values[keep])
        w = np.abs(pv[keep]) if weighted else None
        return float(np.average(ok, weights=w)), int(keep.sum())

    obs = push(roots)
    rng = np.random.default_rng(a.seed)
    rows = []
    for label, d in [("DOWNSTREAM genes (Levels 2+), unweighted", down),
                     ("DOWNSTREAM genes, weighted by |push|", down)]:
        wtd = label.endswith("|push|")
        o, n_g = concord(obs, d, wtd)
        null = [concord(push([(gname, v * rng.choice([-1, 1])) for gname, v in roots]), d, wtd)[0]
                for _ in range(a.perm)]
        null = np.array(null, float)
        rows.append({"test": label, "genes": n_g, "tree_concordance": round(o, 3),
                     "shuffled_mean": round(np.nanmean(null), 3),
                     "shuffled_95%": f"{np.nanpercentile(null, 2.5):.2f}-{np.nanpercentile(null, 97.5):.2f}",
                     "p_tree_vs_shuffled": (np.sum(null >= o) + 1) / (np.sum(~np.isnan(null)) + 1)})
    # Level-1 genes on their own (AlphaGenome directions at the T2D loci; partly circular)
    l1 = lvl1.merge(g1[["gene_name", "weighted_score"]], left_on="gene", right_on="gene_name")
    if len(l1):
        ok = np.sign(l1["weighted_score"]) == np.sign(l1["wald_ratio"])
        rows.append({"test": "LEVEL-1 genes (AlphaGenome direction; same loci - interpret with care)",
                     "genes": len(l1), "tree_concordance": round(ok.mean(), 3), "shuffled_mean": 0.5,
                     "shuffled_95%": "", "p_tree_vs_shuffled":
                         stats.binomtest(int(ok.sum()), len(l1), 0.5, "greater").pvalue})
    out = pd.DataFrame(rows)
    os.makedirs(a.res, exist_ok=True)
    out.to_csv(f"{a.res}/genetic_direction_test.csv", index=False)
    det = down.assign(tree_push=[obs[idx[gname]] for gname in down["gene"]])
    det["tree_direction"] = np.where(det["tree_push"] > 0, "UP in diabetic",
                                     np.where(det["tree_push"] < 0, "DOWN in diabetic", "not reached"))
    det["agree"] = np.sign(det["tree_push"]) == np.sign(det["wald_ratio"])
    det.sort_values("tree_push", key=abs, ascending=False).to_csv(f"{a.res}/genetic_direction_per_gene.csv",
                                                                   index=False)
    pd.set_option("display.width", 200)
    print("\n=== GENETIC DIRECTION TEST ===")
    print(out.to_string(index=False))
    show = det[det["tree_push"] != 0].sort_values("tree_push", key=abs, ascending=False).head(15)
    print("\nStrongest-pushed downstream genes with a genetic estimate:")
    print(show[["gene", "tree_direction", "genetic_direction", "p_t2d", "agree"]].to_string(index=False))
    print("\nReading it: tree_concordance clearly above the shuffled range (small p) = the tree's"
          "\n            up/down predictions match human genetics beyond chance.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--res", default="results_hybrid")
    ap.add_argument("--p", type=float, default=0.05, help="T2D p-value threshold at the eQTL lead")
    ap.add_argument("--window", type=int, default=1_000_000)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    if a.prepare:
        prepare(a)
    elif a.run:
        run(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
