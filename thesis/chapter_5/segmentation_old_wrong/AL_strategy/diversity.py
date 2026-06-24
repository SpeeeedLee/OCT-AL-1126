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
