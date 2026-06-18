#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cold-start initial-set selection via foundation-model embeddings + clustering.
================================================================================
Faithful implementation of the selection algorithm in:

    "From Cold Start to Active Learning: Embedding-Based Scan Selection for
     Medical Image Segmentation" (arXiv:2601.18532, 2026).

Pipeline (per their method section):
  1. Take frozen FM embeddings of D_train  (precomputed by extract_embeddings.py)
  2. Project to 2D with t-SNE          (`--reduce tsne2d`, the paper's choice)
  3. For each candidate k, KMeans + mean silhouette score; pick k̂ = argmax
  4. KMeans(k̂); each cluster's MEDOID is the first seed   (k̂ seeds)
  5. Distribute remaining budget R = B - k̂ proportionally to cluster size:
         r_c = round(R * |C_c| / Σ|C_u|)        (largest-remainder to sum to R)
  6. Within each cluster: greedy FARTHEST-POINT sampling, seeded at the medoid
  7. -> B selected train indices, for budget B = round(portion% * N)

Output: a JSON in the *existing* labeled_ids schema so it drops straight into
both the one-shot trainer and `run_AL.py --resume_labeled_ids`:

    thesis/chapter_5/coldstart_fm/labeled_ids/{model_id}.json
    {
      "2.5": {"n_cumulative":51,  "selected":[...51], "cumulative":[...51],
              "k_hat":..., "reduce":"tsne2d", "source":"coldstart_fm:<model>"},
      "10.0": {...}, "20.0": {...}
    }
At each portion the set is built INDEPENDENTLY (a cold-start one-shot pick), so
"selected" == "cumulative" (there is no incremental history). To use a portion
as an AL initial pool: run_AL.py --resume_labeled_ids <json> --resume_from <p>.

Run from repo root:
    python3 thesis/chapter_5/coldstart_fm/select_coldstart.py --model dinov2:base
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from thesis.chapter_5.coldstart_fm.extract_embeddings import cache_path  # noqa: E402

OUT_DIR = os.path.join(REPO, "thesis", "chapter_5", "coldstart_fm", "labeled_ids")
DEFAULT_PORTIONS = [2.5, 10.0, 20.0]
SEED = 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _largest_remainder(float_counts, total):
    """Round float per-cluster counts to ints summing exactly to `total`."""
    float_counts = np.asarray(float_counts, dtype=float)
    floor = np.floor(float_counts).astype(int)
    rem = int(total - floor.sum())
    if rem > 0:
        order = np.argsort(-(float_counts - floor))
        for i in range(rem):
            floor[order[i % len(floor)]] += 1
    elif rem < 0:
        order = np.argsort(float_counts - floor)
        for i in range(-rem):
            # never go below zero
            j = order[i % len(floor)]
            if floor[j] > 0:
                floor[j] -= 1
    return floor


def _medoid(points, global_idx):
    """Return the GLOBAL index of the medoid of `points` (min sum of L2 dist).
    Uses sklearn pairwise_distances (C-optimized, m x m memory) — the naive
    [m,m,D] broadcast blows up on the large (~800-pt) clusters."""
    if len(global_idx) == 1:
        return int(global_idx[0])
    from sklearn.metrics import pairwise_distances
    d = pairwise_distances(points, metric="euclidean")
    return int(global_idx[int(np.argmin(d.sum(axis=1)))])


def _farthest_point(points, global_idx, n_pick, seed_local):
    """Greedy farthest-point sampling within one cluster.
    Start from `seed_local` (local index of the medoid), then repeatedly add the
    point maximizing its min-distance to the already-selected set.
    Returns up to n_pick GLOBAL indices EXCLUDING the seed (seed counted already)."""
    m = len(global_idx)
    if n_pick <= 0 or m <= 1:
        return []
    min_d = np.linalg.norm(points - points[seed_local], axis=1)
    min_d[seed_local] = -1.0                       # seed already chosen
    picked = []
    for _ in range(min(n_pick, m - 1)):
        nxt = int(np.argmax(min_d))
        if min_d[nxt] < 0:
            break
        picked.append(int(global_idx[nxt]))
        d = np.linalg.norm(points - points[nxt], axis=1)
        min_d = np.minimum(min_d, d)
        min_d[nxt] = -1.0
    return picked


def _choose_k(feats, budget, seed=SEED, kmax_cap=50, verbose=True):
    """k̂ = argmax mean-silhouette over candidate k in [2, min(budget, kmax_cap)].
    (k must be <= budget since every cluster contributes one medoid seed; the cap
    keeps the silhouette sweep bounded — it favours modest k in practice anyway.)"""
    n = len(feats)
    kmax = int(min(budget, kmax_cap, max(2, n - 1)))
    candidates = list(range(2, kmax + 1))
    if len(candidates) <= 1:
        return max(2, min(budget, n - 1))
    best_k, best_s = candidates[0], -1.0
    scores = {}
    for k in candidates:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        lab = km.fit_predict(feats)
        if len(np.unique(lab)) < 2:
            continue
        # subsample for the silhouette estimate when n is large (O(n^2 D) otherwise)
        ss = None if len(feats) <= 1500 else 1500
        s = silhouette_score(feats, lab, sample_size=ss, random_state=seed)
        scores[k] = float(s)
        if s > best_s:
            best_k, best_s = k, s
    if verbose:
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
        print(f"    silhouette top-5 (k,score): "
              f"{[(k, round(v,4)) for k,v in top]}  -> k_hat={best_k}")
    return best_k


# --------------------------------------------------------------------------- #
# the algorithm
# --------------------------------------------------------------------------- #
def select_for_budget(embeddings, budget, reduce="tsne2d", seed=SEED, verbose=True):
    """Return a sorted list of `budget` train indices (0..N-1)."""
    n = embeddings.shape[0]
    budget = int(min(budget, n))

    # 1-2. dimensionality reduction
    X = embeddings.astype(np.float32)
    if reduce == "tsne2d":
        perp = float(min(30, max(5, (n - 1) / 3)))
        feats = TSNE(n_components=2, random_state=seed, perplexity=perp,
                     init="pca").fit_transform(X)
    elif reduce == "pca50":
        feats = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
    elif reduce == "none":
        feats = X
    else:
        raise ValueError(f"unknown reduce '{reduce}'")
    feats = np.ascontiguousarray(feats, dtype=np.float32)

    # 3. choose k via silhouette
    k_hat = _choose_k(feats, budget, seed=seed, verbose=verbose)

    # 4. KMeans + medoids
    km = KMeans(n_clusters=k_hat, random_state=seed, n_init=10)
    lab = km.fit_predict(feats)
    clusters = {c: np.where(lab == c)[0] for c in np.unique(lab)}
    cluster_ids = sorted(clusters.keys())

    medoids = {}
    for c in cluster_ids:
        members = clusters[c]
        medoids[c] = _medoid(feats[members], members)
    selected = set(medoids.values())

    # 5. proportional allocation of the remaining budget
    R = budget - len(selected)
    sizes = np.array([len(clusters[c]) for c in cluster_ids], dtype=float)
    if R > 0:
        alloc = _largest_remainder(R * sizes / sizes.sum(), R)
    else:
        alloc = np.zeros(len(cluster_ids), dtype=int)

    # 6. farthest-point sampling within each cluster (seeded at the medoid)
    #    cap each cluster's pick at (cluster_size - 1) extra points; redistribute leftover
    for ci, c in enumerate(cluster_ids):
        members = clusters[c]
        seed_local = int(np.where(members == medoids[c])[0][0])
        n_extra = int(min(alloc[ci], len(members) - 1))
        picks = _farthest_point(feats[members], members, n_extra, seed_local)
        selected.update(picks)

    # if rounding/caps left us short of budget, top up with global farthest-point
    if len(selected) < budget:
        remaining = [i for i in range(n) if i not in selected]
        if remaining:
            rem = np.array(remaining)
            sel_arr = np.array(sorted(selected))
            # min distance from each remaining point to the selected set
            d = np.full(len(rem), np.inf, dtype=np.float32)
            for s in sel_arr:
                d = np.minimum(d, np.linalg.norm(feats[rem] - feats[s], axis=1))
            order = np.argsort(-d)
            for j in order:
                if len(selected) >= budget:
                    break
                selected.add(int(rem[j]))
    # if we overshot (shouldn't, but guard), trim deterministically
    sel = sorted(selected)
    if len(sel) > budget:
        sel = sel[:budget]

    if verbose:
        print(f"    budget={budget} k_hat={k_hat} medoids={len(medoids)} "
              f"final_selected={len(sel)}")
    return sel, k_hat


def run(model_id, portions=DEFAULT_PORTIONS, reduce="tsne2d", seed=SEED, overwrite=False):
    cpath = cache_path(model_id)
    if not os.path.isfile(cpath):
        raise FileNotFoundError(
            f"no cached embeddings for {model_id} at {cpath}; "
            f"run extract_embeddings.py --model {model_id} first")
    payload = torch.load(cpath, map_location="cpu", weights_only=False)
    emb = payload["embeddings"].numpy()
    n = emb.shape[0]
    print(f"\n=== select {model_id}: embeddings {emb.shape}, reduce={reduce} ===")

    out = {}
    for p in portions:
        budget = int(round(n * p / 100.0))
        print(f"  portion {p}% -> budget {budget}")
        sel, k_hat = select_for_budget(emb, budget, reduce=reduce, seed=seed)
        # sanity
        assert len(sel) == len(set(sel)) == budget, \
            f"selection size mismatch: {len(sel)} vs {budget}"
        assert all(0 <= i < n for i in sel)
        pk = str(float(p))
        out[pk] = {
            "n_cumulative": len(sel),
            "selected": list(sel),
            "cumulative": list(sel),
            "k_hat": int(k_hat),
            "reduce": reduce,
            "source": f"coldstart_fm:{model_id}",
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, model_id.replace(":", "__") + ".json")
    if reduce != "tsne2d":
        out_path = out_path.replace(".json", f"_{reduce}.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[saved] {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id, e.g. dinov2:base")
    ap.add_argument("--portions", type=float, nargs="+", default=DEFAULT_PORTIONS)
    ap.add_argument("--reduce", choices=["tsne2d", "pca50", "none"], default="tsne2d")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    run(args.model, args.portions, args.reduce, args.seed)


if __name__ == "__main__":
    main()
