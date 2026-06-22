"""
Hybrid AL for nuclei segmentation -- CLUSTER-MARGIN
(Citovsky et al., NeurIPS 2021), adapted to segmentation.

Pipeline (parallels the Ch4 classification cluster_margin):
  1. Score every unlabeled image by image-level MARGIN uncertainty
     (mean over pixels of 1-|2p-1|); take the k_m = k_factor * budget most
     uncertain images as candidates.
  2. Embed candidates with the U-Net encoder (GAP bottleneck) and L2-normalise.
  3. Agglomerative clustering (average linkage). The distance threshold eps is
     scale-adaptive: median pairwise distance * EPS_FRAC (the paper leaves eps
     unspecified; this mirrors the Ch4 choice).
  4. Round-robin over clusters in ASCENDING size, drawing one random member each,
     until `budget` images are picked.
"""
import random
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from thesis.chapter_5.segmentation.AL_strategy.uncertainty import compute_margin_uncertainty
from thesis.chapter_5.segmentation.AL_strategy.features import extract_encoder_embeddings

K_FACTOR = 10      # candidate pool = K_FACTOR * budget (most uncertain)
EPS_FRAC = 0.5     # eps = median pairwise distance * EPS_FRAC


def cluster_margin(model, opath, gpath_cell, unlabel_data_idx, num_data_to_label, device,
                   k_factor=K_FACTOR, eps_frac=EPS_FRAC):
    """Returns (to_label_files, info_dict)."""
    # --- Step 1: margin candidate pool ---
    k_m = min(num_data_to_label * k_factor, len(unlabel_data_idx))
    print(f"Cluster-Margin: scoring {len(unlabel_data_idx)} images, "
          f"taking top {k_m} uncertain candidates...")
    scores = compute_margin_uncertainty(model, opath, unlabel_data_idx, device)
    cand_order = np.argsort(scores)[::-1][:k_m]      # most uncertain first
    cand_files = [unlabel_data_idx[i] for i in cand_order]

    # If the pool is already <= budget, just take it.
    if k_m <= num_data_to_label:
        return cand_files[:num_data_to_label], {"strategy": "cluster_margin",
                                                "note": "pool<=budget", "selected_files": cand_files}

    # --- Step 2: embed + L2-normalise ---
    emb = extract_encoder_embeddings(model, opath, cand_files, device)
    emb = normalize(emb)                             # L2

    # --- Step 3: HAC with scale-adaptive eps ---
    D = pairwise_distances(emb)
    iu = np.triu_indices_from(D, k=1)
    eps = float(np.median(D[iu])) * eps_frac
    hac = AgglomerativeClustering(n_clusters=None, distance_threshold=eps,
                                  linkage='average', metric='euclidean')
    labels = hac.fit_predict(emb)
    n_clusters = labels.max() + 1
    print(f"  HAC eps={eps:.4f} -> {n_clusters} clusters")

    # --- Step 4: round-robin over clusters in ascending size ---
    clusters = {c: list(np.where(labels == c)[0]) for c in range(n_clusters)}
    order = sorted(clusters.keys(), key=lambda c: len(clusters[c]))  # ascending size
    for c in clusters:
        random.shuffle(clusters[c])

    selected = []
    while len(selected) < num_data_to_label:
        progressed = False
        for c in order:
            if clusters[c]:
                selected.append(clusters[c].pop())
                progressed = True
                if len(selected) >= num_data_to_label:
                    break
        if not progressed:
            break

    to_label = [cand_files[i] for i in selected]
    info = {
        "strategy": "cluster_margin",
        "k_m": k_m,
        "n_clusters": int(n_clusters),
        "eps": round(eps, 6),
        "final_num": num_data_to_label,
        "selected_files": to_label,
    }
    return to_label, info
