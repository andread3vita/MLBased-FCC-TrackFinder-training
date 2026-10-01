"""Circle-consistency merge on ALL 50 keep-all seeds (ep6 embeddings).

Workers are sharded by seed: each loads its seeds' detector features + cached
embeddings, runs tight-td -> star-merge -> circle-verified merge, and emits
per-particle rows. Results are joined against the ep6 eval cache for pt and
IDEA-reconstructable flags, giving match/strict50/hit-eff/per-pT + a cluster
purity-fake proxy for each variant.
"""
import sys, time, os
import numpy as np
import polars as pl
from multiprocessing import Pool

sys.path.insert(0, "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg")
from src.eval.fcc_cache_parallel import merge_fragments
from src.eval_sweep_v33 import get_clustering_greedy
from src.lowpt_op_sweep import cache_from_dataframe
from src.dataset.parquet_dataset import IDEAParquetDataset

PKG = "/home/marko.cechovic/cgatr-paper/literature/cgatr_fcc_pkg"
CFG = os.environ.get("CIRCLE_CFG", "ft_ep6_keepall")
FWD = f"{PKG}/eval_results/{CFG}/emb_all/forward_hits.parquet"
REF = f"{PKG}/eval_results/{CFG}/fcc_unmerged/cache.parquet"
DATA = "/home/marko.cechovic/cgatr/data_parquet_train/v1_zqq_uds"
OUT = f"{PKG}/eval_results/{CFG}/circle_merge"
TD, MG = 0.05, 0.10
PAIR_LO, PAIR_HI = 0.10, 0.35
THRS = [1.5, 2.0]
VARIANTS = ["dist"] + [f"circle{t}" for t in THRS]


def kasa_fit(x, y, w):
    A = np.stack([x, y, np.ones_like(x)], 1) * w[:, None]
    b = (x**2 + y**2) * w
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = sol[0] / 2, sol[1] / 2
    R2 = sol[2] + cx**2 + cy**2
    return (cx, cy, np.sqrt(R2)) if R2 > 0 else None


def tangency_res(fit, F):
    if fit is None:
        return np.inf
    cx, cy, R = fit
    dc = F[:, 3] > 0.5
    res = np.empty(len(F))
    dv = np.hypot(F[~dc, 0] - cx, F[~dc, 1] - cy)
    res[~dc] = np.abs(dv - R)
    dw = np.hypot(F[dc, 4] - cx, F[dc, 5] - cy)
    res[dc] = np.abs(np.abs(dw - R) - F[dc, 7])
    return np.median(res)


def fit_cluster(F):
    dc = F[:, 3] > 0.5
    x = np.concatenate([F[~dc, 0], F[dc, 4]])
    y = np.concatenate([F[~dc, 1], F[dc, 5]])
    w = np.concatenate([np.full((~dc).sum(), 3.0), np.ones(dc.sum())])
    return kasa_fit(x, y, w) if len(x) >= 4 else None


def labels_variants(betas, X, F):
    lab0 = get_clustering_greedy(betas, X, tbeta=0.1, td=TD)
    lab1 = merge_fragments(lab0, X, betas, MG)
    out = {"dist": lab1}
    vals, cnts = np.unique(lab1[lab1 >= 0], return_counts=True)
    cores = vals[cnts >= 5]
    P = X[cores]
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    ii, jj = np.nonzero((D > PAIR_LO) & (D < PAIR_HI))
    pairs = [(a, b) for a, b in zip(ii, jj) if a < b]
    feats = {c: F[lab1 == c] for c in cores}
    resid = {c: tangency_res(fit_cluster(feats[c]), feats[c]) for c in cores}
    ru_cache = {}
    for a, b in pairs:
        ca, cb = cores[a], cores[b]
        Fu = np.concatenate([feats[ca], feats[cb]])
        ru_cache[(ca, cb)] = tangency_res(fit_cluster(Fu), Fu)
    for thr in THRS:
        parent = {c: c for c in cores}
        def find(c):
            while parent[c] != c:
                parent[c] = parent[parent[c]]; c = parent[c]
            return c
        for (ca, cb), ru in ru_cache.items():
            if ru < thr and ru < 1.3 * max(resid[ca], resid[cb], 0.4):
                ra, rb = find(ca), find(cb)
                if ra != rb:
                    parent[rb] = ra
        lab2 = lab1.copy()
        m = lab2 >= 0
        lut = {c: find(c) for c in cores}
        lab2[m] = np.array([lut.get(l, l) for l in lab2[m]])
        out[f"circle{thr}"] = lab2
    return out


def process_seeds(seeds):
    hits = pl.read_parquet(FWD).filter(pl.col("seed").is_in(list(seeds)))
    cache = cache_from_dataframe(hits, 4)
    by_key = {(e["seed"], e["event_id"]): e for e in cache}
    del hits
    prows, crows = [], []
    for seed in seeds:
        ds = IDEAParquetDataset(DATA, seed_range=(seed, seed + 1), max_hits_per_event=0)
        for idx in range(len(ds)):
            _, _, eid, _ = ds._index[idx]
            e = by_key.get((seed, int(eid)))
            if e is None:
                continue
            d = ds[idx]
            mc_all = d["mc_index"].numpy()
            mask = (~d["is_secondary"].numpy()) & (mc_all != 0)
            if not np.array_equal(mc_all[mask], e["sig_mc"]):
                continue
            F = d["features"].numpy()[mask]
            mc = e["sig_mc"]; nhtm = e["n_hits_total_map"]
            labs = labels_variants(e["sig_beta"], e["sig_coords"], F)
            for var, lab in labs.items():
                for p in np.unique(mc[mc >= 0]):
                    tot = nhtm.get(int(p), int((mc == p).sum()))
                    sel = mc == p
                    ls = lab[sel]; ls = ls[ls >= 0]
                    if len(ls) == 0:
                        prows.append((var, seed, int(eid), int(p), 0, 0.0, 0))
                        continue
                    vals, cnts = np.unique(ls, return_counts=True)
                    best = vals[np.argmax(cnts)]; shared = int(cnts.max())
                    pur = shared / int((lab == best).sum()); he = shared / tot
                    prows.append((var, seed, int(eid), int(p),
                                  int(pur >= 0.75), he, int(pur >= 0.5 and he >= 0.5)))
                vals = np.unique(lab[lab >= 0])
                nfake = 0
                for v in vals:
                    cm = mc[lab == v]
                    _, cc = np.unique(cm[cm >= 0], return_counts=True)
                    if (cc.max() if len(cc) else 0) / len(cm) < 0.75:
                        nfake += 1
                crows.append((var, seed, int(eid), len(vals), nfake))
    return prows, crows


def main():
    t0 = time.time()
    seeds = list(range(1001, 1051))
    shards = [seeds[i::12] for i in range(12)]
    prows, crows = [], []
    with Pool(12) as pool:
        for pr, cr in pool.imap_unordered(process_seeds, shards):
            prows.extend(pr); crows.extend(cr)
            print(f"  shard done ({len(prows):,} particle rows, {time.time()-t0:.0f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    pdf = pl.DataFrame(prows, schema=["variant", "seed", "event_id", "mc_idx",
                                      "matched", "hiteff", "strict"], orient="row")
    cdf = pl.DataFrame(crows, schema=["variant", "seed", "event_id",
                                      "n_clusters", "n_fake"], orient="row")
    pdf.write_parquet(f"{OUT}/particles.parquet")
    cdf.write_parquet(f"{OUT}/clusters.parquet")

    ref = pl.read_parquet(REF, columns=["mc_idx", "event_id", "seed", "pt",
                                        "is_reconstructable_idea"])
    j = pdf.join(ref, on=["mc_idx", "event_id", "seed"], how="inner")
    idea = j.filter(pl.col("is_reconstructable_idea"))
    bins = [(0.1,0.15),(0.15,0.2),(0.2,0.3),(0.3,0.5),(0.5,1.0),(1.0,3.0),(3.0,30.0)]
    print(f"\n=== ep6 keep-all, 50 seeds, IDEA cuts (td={TD}, merge={MG}) ===")
    print(f"{'variant':>10} | {'match':>6} {'strict50':>8} {'hit-eff':>7} {'fake~':>6} | min-pt-bin")
    for var in VARIANTS:
        s = idea.filter(pl.col("variant") == var)
        c = cdf.filter(pl.col("variant") == var)
        fake = c["n_fake"].sum() / max(c["n_clusters"].sum(), 1)
        bvals = []
        for a, bnd in bins:
            sub = s.filter((pl.col("pt") >= a) & (pl.col("pt") < bnd))
            bvals.append(sub["matched"].mean() if sub.height else float("nan"))
        print(f"{var:>10} | {s['matched'].mean():6.3f} {s['strict'].mean():8.3f} "
              f"{s['hiteff'].mean():7.3f} {fake:6.3f} | {np.nanmin(bvals):.3f}")
        print(f"{'':>10}   bins: " + "  ".join(f"{v:.3f}" for v in bvals))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
