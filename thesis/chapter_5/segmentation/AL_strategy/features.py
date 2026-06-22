"""
Image-level feature embeddings for diversity / hybrid AL on segmentation.

A whole image is the selection unit (no patches). To get one feature vector per
image we take the U-Net ENCODER bottleneck (Conv6, the deepest conv block,
1024-d) and global-average-pool it -- the standard way to adapt Core-set /
Cluster-Margin to dense prediction (encoder features as the image descriptor).
"""
import numpy as np
import torch

from thesis.chapter_5.segmentation.utils.data import o_data

WIDTH, HEIGHT = 384, 512


@torch.no_grad()
def extract_encoder_embeddings(model, opath, files, device, width=WIDTH, height=HEIGHT):
    """
    Return an [N, C] numpy array of global-average-pooled bottleneck features,
    one row per filename in `files` (order preserved).
    """
    model.eval()
    model.to(device)

    feats = []

    def hook_fn(module, inp, out):
        # out: [B, C, h, w] -> GAP -> [B, C]
        g = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        feats.append(g.view(g.size(0), -1).cpu())

    # Conv6 is the true bottleneck of Optim_U_Net (filter_size*32 = 1024-d).
    handle = model.Conv6.register_forward_hook(hook_fn)
    embeddings = []
    try:
        for name in files:
            feats.clear()
            img = o_data(opath, [name], width, height)
            INPUT = torch.from_numpy(img.astype(np.float32)).to(device=device, dtype=torch.float)
            _ = model(INPUT)
            embeddings.append(feats[0].numpy())
    finally:
        handle.remove()

    return np.vstack(embeddings)  # [N, C]
