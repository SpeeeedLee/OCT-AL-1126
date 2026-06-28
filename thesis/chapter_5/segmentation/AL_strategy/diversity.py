"""
Diversity AL for nuclei segmentation -- CORE-SET (k-Center-Greedy),
adapted to segmentation by using image-level encoder embeddings.

Faithful to Sener & Savarese (ICLR 2018) and to the Ch4 classification fix:
the greedy selection is CONDITIONED on the already-labeled set -- every
unlabeled point's distance is initialised to its nearest LABELED point, then we
iteratively add the farthest point and update min-distances.
"""
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from thesis.chapter_5.segmentation.AL_strategy.features import extract_encoder_embeddings


def coreset(model, opath, unlabel_data_idx, label_idx, num_data_to_label, device):
    """
    k-Center-Greedy on encoder embeddings, conditioned on the labeled set.

    Returns (to_label_files, info_dict).
    """
    print(f"Core-set: embedding {len(unlabel_data_idx)} unlabeled "
          f"+ {len(label_idx)} labeled images...")
    emb_u = extract_encoder_embeddings(model, opath, unlabel_data_idx, device)  # [Nu, D]

    if len(label_idx) > 0:
        emb_l = extract_encoder_embeddings(model, opath, label_idx, device)     # [Nl, D]
        min_dist = pairwise_distances(emb_u, emb_l).min(axis=1)                 # [Nu]
    else:
        # No labeled centers yet -> start from the single farthest-spread point.
        min_dist = np.full(len(unlabel_data_idx), np.inf)

    selected = []
    for _ in range(num_data_to_label):
        i = int(np.argmax(min_dist))
        selected.append(i)
        # update distances with the newly added center
        d_new = np.linalg.norm(emb_u - emb_u[i], axis=1)
        min_dist = np.minimum(min_dist, d_new)
        min_dist[i] = -1.0  # never reselect

    to_label = [unlabel_data_idx[i] for i in selected]
    info = {
        "strategy": "coreset",
        "final_num": num_data_to_label,
        "selected_files": to_label,
    }
    return to_label, info


# ---------------------------------------------------------------------------
# TypiClust (Hacohen et al., ICML 2022) — low-budget / cold-start diversity AL.
# Image-level adaptation: SAME encoder-GAP embeddings as Core-set
# (extract_encoder_embeddings), faithful to the Ch4 classification implementation
# (classification/AL_strategy/diversity_correct.py::typiclust).
# ---------------------------------------------------------------------------
_K_NN = 20
_MIN_CLUSTER_SIZE = 5
_MAX_NUM_CLUSTERS = 500


def _typicality(feats, k):
    """feats: [m, D] -> per-point typicality = 1 / mean distance to its k nearest."""
    m = feats.shape[0]
    if m == 1:
        return np.array([1.0])
    k = max(1, min(k, m - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(feats)   # +1: includes self
    dist, _ = nn.kneighbors(feats)
    mean_d = dist[:, 1:].mean(axis=1)                     # drop col 0 (self)
    return 1.0 / (mean_d + 1e-5)


def typiclust(model, opath, unlabel_data_idx, label_idx, num_data_to_label, device):
    """
    TypiClust: cluster (unlabeled + labeled) embeddings into ~|L|+budget clusters;
    order clusters by (fewest labeled first, then largest), then round-robin pick the
    most TYPICAL (highest 1/avg-KNN-dist) unlabeled point per cluster.

    Returns (to_label_files, info_dict). Uses the SAME features as Core-set.
    """
    print(f"TypiClust: embedding {len(unlabel_data_idx)} unlabeled "
          f"+ {len(label_idx)} labeled images...")
    emb_u = extract_encoder_embeddings(model, opath, unlabel_data_idx, device)   # [Nu, D]
    Nu = emb_u.shape[0]
    if len(label_idx) > 0:
        emb_l = extract_encoder_embeddings(model, opath, label_idx, device)      # [Nl, D]
        feats = np.vstack([emb_u, emb_l])
    else:
        feats = emb_u
    is_unlabeled = np.arange(feats.shape[0]) < Nu        # first Nu rows are unlabeled

    budget = num_data_to_label
    n_clusters = int(min(len(label_idx) + budget, _MAX_NUM_CLUSTERS))
    n_clusters = max(1, min(n_clusters, feats.shape[0]))
    if n_clusters <= 50:
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    else:
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=5000, n_init=3)
    cl = km.fit_predict(feats)

    clusters = {}
    for c in np.unique(cl):
        members = np.where(cl == c)[0]
        clusters[c] = {"members": members, "size": len(members),
                       "n_lab": int(np.sum(~is_unlabeled[members]))}
    # keep clusters with size>=MIN (relax to all if none); order: fewest-labeled, then biggest
    elig = [c for c in clusters if clusters[c]["size"] >= _MIN_CLUSTER_SIZE]
    if not elig:
        elig = list(clusters.keys())
    order = sorted(elig, key=lambda c: (clusters[c]["n_lab"], -clusters[c]["size"]))

    selected, i, guard, seen = [], 0, 0, set()
    max_guard = budget * 50 + len(order) + 10
    while len(selected) < budget and guard < max_guard:
        guard += 1
        c = order[i % len(order)]; i += 1
        cand = [p for p in clusters[c]["members"] if p < Nu and p not in seen]
        if not cand:
            continue
        cand = np.array(cand)
        typ = _typicality(feats[cand], _K_NN)
        pick = int(cand[int(np.argmax(typ))])
        selected.append(pick); seen.add(pick)

    # safety fill (e.g. too few eligible clusters): take remaining unlabeled by typicality
    if len(selected) < budget:
        remaining = [p for p in range(Nu) if p not in seen]
        if remaining:
            rt = _typicality(emb_u[remaining], _K_NN)
            for j in np.argsort(-rt):
                if len(selected) >= budget:
                    break
                selected.append(remaining[int(j)]); seen.add(remaining[int(j)])

    to_label = [unlabel_data_idx[p] for p in selected]
    info = {"strategy": "typiclust", "final_num": len(to_label), "selected_files": to_label}
    return to_label, info
