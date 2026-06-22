"""
Uncertainty AL for nuclei segmentation -- MARGIN, aggregated as the
pixel-wise MEAN over the image (image-level selection, no patches).

Binary margin per pixel:
    margin = |p - (1-p)| = |2p - 1|         (gap between the two class probs)
    uncertainty = 1 - margin = 2 * min(p, 1-p)
Higher uncertainty  <=>  prob closer to 0.5  <=>  model less sure.

Image score = mean of the per-pixel uncertainty over ALL pixels of the image.
This is the standard "aggregate per-pixel uncertainty by averaging" recipe used
to lift classification-style margin sampling to semantic segmentation.
"""
import numpy as np
import torch
from tqdm import tqdm

from thesis.chapter_5.segmentation.utils.data import o_data

WIDTH, HEIGHT = 384, 512


@torch.no_grad()
def compute_margin_uncertainty(model, opath, files, device, width=WIDTH, height=HEIGHT):
    """Return np array [N] of mean-over-pixels margin-uncertainty, aligned to `files`."""
    model.eval()
    model.to(device)
    scores = []
    for img_name in tqdm(files, desc="Margin uncertainty"):
        img = o_data(opath, [img_name], width, height)
        INPUT = torch.from_numpy(img.astype(np.float32)).to(device=device, dtype=torch.float)
        prob = model(INPUT).squeeze()                 # [H, W], sigmoid prob
        uncertainty = 1.0 - torch.abs(2.0 * prob - 1.0)  # = 2*min(p,1-p)
        scores.append(uncertainty.mean().item())
    return np.array(scores)


@torch.no_grad()
def _mean_pixel_score(model, opath, files, device, kind, width=WIDTH, height=HEIGHT):
    """Per-image score = mean over pixels of a per-pixel uncertainty.
      kind='margin'  : 1-|2p-1|            (= 2*min(p,1-p))
      kind='conf'    : 1-max(p,1-p)        (least-confidence; = min(p,1-p))
      kind='entropy' : -[p log p + (1-p) log(1-p)]
    """
    model.eval(); model.to(device)
    eps = 1e-10
    scores = []
    for img_name in tqdm(files, desc=f"{kind} uncertainty"):
        img = o_data(opath, [img_name], width, height)
        INPUT = torch.from_numpy(img.astype(np.float32)).to(device=device, dtype=torch.float)
        p = model(INPUT).squeeze()
        if kind == 'margin':
            u = 1.0 - torch.abs(2.0 * p - 1.0)
        elif kind == 'conf':
            u = 1.0 - torch.maximum(p, 1.0 - p)
        elif kind == 'entropy':
            pc = torch.clamp(p, eps, 1 - eps)
            u = -(pc * torch.log(pc) + (1 - pc) * torch.log(1 - pc))
        else:
            raise ValueError(kind)
        scores.append(u.mean().item())
    return np.array(scores)


def _select_top(scores, files, k):
    order = np.argsort(scores)[::-1]
    chosen = order[:k]
    return [files[i] for i in chosen], {files[i]: float(scores[i]) for i in order}


def margin(model, opath, gpath_cell, unlabel_data_idx, num_data_to_label, device,
           width=WIDTH, height=HEIGHT):
    """Select images with HIGHEST mean margin-uncertainty (1-|2p-1|)."""
    print(f"Computing margin uncertainty for {len(unlabel_data_idx)} unlabeled images...")
    scores = compute_margin_uncertainty(model, opath, unlabel_data_idx, device, width, height)
    return _select_top(scores, unlabel_data_idx, num_data_to_label)


def confidence(model, opath, gpath_cell, unlabel_data_idx, num_data_to_label, device,
               width=WIDTH, height=HEIGHT):
    """Least-confidence: highest mean (1-max(p,1-p)). NOTE: for BINARY masks this is a
    monotone transform of margin, so it selects the SAME images as margin()."""
    print(f"Computing confidence (least-conf) for {len(unlabel_data_idx)} images...")
    scores = _mean_pixel_score(model, opath, unlabel_data_idx, device, 'conf', width, height)
    return _select_top(scores, unlabel_data_idx, num_data_to_label)


def entropy(model, opath, gpath_cell, unlabel_data_idx, num_data_to_label, device,
            width=WIDTH, height=HEIGHT):
    """Highest mean binary entropy over pixels."""
    print(f"Computing entropy for {len(unlabel_data_idx)} images...")
    scores = _mean_pixel_score(model, opath, unlabel_data_idx, device, 'entropy', width, height)
    return _select_top(scores, unlabel_data_idx, num_data_to_label)
