#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AL ρ=30% 五模型 Grad-CAM 對比 panel（SL / SK 各 top-5）
========================================================
模型（皆 portion30、seed42、aug4 fine-tuned）：
    random = thesis/gradcam/ckpt/simclr_p30_4x.pth   （= cold-start/random 子集）
    margin / coreset / cluster_margin / typiclust = *_p30_seed42_4x.pth

挑圖 criteria（對每張 SL/SK test 影像，取各模型對「真實病灶類別」的 softmax 機率 P）：
    avg3 = mean(P_margin, P_coreset, P_cluster_margin)
    候選須滿足   P_typiclust < avg3        （TypiClust 也輸這三者平均）
    依         (P_random - avg3) 由小到大   （random 輸最多）排序 → 取 top-5
    SL、SK 各自獨立取 top-5。

test split：重現 classification/utils/data.py 的切法（ImageFolder(val) → stratified
            train_test_split, test_size=0.5, random_state=42 的 test 半）。

輸出：
    thesis/gradcam/out/{SL,SK}_top5_gradcam.png   （rows=5 張影像, cols=Input+5 模型, 標題印 P）
    thesis/gradcam/out/al_p30_selection.json       （挑圖過程的完整機率/排序紀錄）

從 repo root 執行：
    python3 thesis/gradcam/al_p30_gradcam_panels.py --device cuda:5
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from thesis.gradcam.gradcam_view import build_model, compute_cam, _TF  # noqa: E402

CKPT_DIR = os.path.join(_ROOT, "thesis", "gradcam", "ckpt")
VAL_DIR  = os.path.join(_ROOT, "ds", "classification", "seven_class", "val")
OUT_DIR  = os.path.join(_ROOT, "thesis", "gradcam", "out")
NUM_CLASSES = 7

# 欄位顯示順序與檔名
MODELS = [
    ("random",         "simclr_p30_4x.pth"),
    ("margin",         "margin_p30_seed42_4x.pth"),
    ("coreset",        "coreset_p30_seed42_4x.pth"),
    ("cluster_margin", "cluster_margin_p30_seed42_4x.pth"),
    ("typiclust",      "typiclust_p30_seed42_4x.pth"),
]
THREE = ["margin", "coreset", "cluster_margin"]   # avg3 用的三者
TARGET_CLASSES = {"SK": "Seborrhoeic keratosis", "SL": "Solar lentigo"}


def reproduce_test_split():
    """回傳 ImageFolder(val) 與其 test 半的 global indices（與 data.py 完全一致）。"""
    ds = datasets.ImageFolder(VAL_DIR)
    targets = np.array(ds.targets)
    indices = np.arange(len(targets))
    _, test_idx = train_test_split(indices, test_size=0.5,
                                   stratify=targets, random_state=42)
    return ds, sorted(test_idx.tolist())


@torch.no_grad()
def all_probs(models, ds, sample_idxs, device):
    """回傳 {global_idx: {model_name: P(true_class)}}（true_class = 該影像的 ImageFolder label）。"""
    out = {}
    for gi in sample_idxs:
        path, true_cls = ds.samples[gi]
        x = _TF(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        rec = {}
        for name, (model, _tl) in models.items():
            p = torch.softmax(model(x), dim=1)[0, true_cls].item()
            rec[name] = p
        out[gi] = rec
    return out


def select_top5(probs, relax=False):
    """對單一類別的影像集套 criteria，回傳排序後的候選 list（含分數）。
       relax=True：跳過 typiclust<avg3 過濾，改用全部 test 影像（候選太少時用）。"""
    scored = []
    for gi, pr in probs.items():
        avg3 = float(np.mean([pr[m] for m in THREE]))
        if not relax and pr["typiclust"] >= avg3:   # 不滿足 TypiClust < avg3
            continue
        scored.append({
            "idx": gi, "avg3": avg3,
            "random": pr["random"], "typiclust": pr["typiclust"],
            "random_minus_avg3": pr["random"] - avg3,
            "typiclust_minus_avg3": pr["typiclust"] - avg3,
            "probs": pr,
        })
    scored.sort(key=lambda d: d["random_minus_avg3"])   # random 輸最多在前
    return scored


def draw_panel(rows, models, ds, true_cls, device, title, out_path, method="gradcam++"):
    """rows: 已選好的影像紀錄 list；每列 = Input + 5 模型的 Grad-CAM overlay。"""
    ncol = 1 + len(MODELS)
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.7 * nrow))
    if nrow == 1:
        axes = axes[None, :]

    for r, rec in enumerate(rows):
        path, _ = ds.samples[rec["idx"]]
        img = Image.open(path).convert("RGB")
        rgb = np.array(img).astype(np.float32) / 255.0
        H, W = rgb.shape[:2]
        x = _TF(img).unsqueeze(0).to(device)

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(os.path.basename(path), fontsize=7)
        if r == 0:
            axes[r, 0].set_title("Input", fontsize=11)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

        for c, (name, _f) in enumerate(MODELS, start=1):
            model, target_layer = models[name]
            cam, _, probs = compute_cam(model, target_layer, x, true_cls, method)
            cam = F.interpolate(cam[None, None], size=(H, W),
                                mode="bilinear", align_corners=False)[0, 0]
            cam = cam - cam.min()
            cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
            heat = cm.jet(cam)[..., :3]
            overlay = np.clip(0.5 * rgb + 0.5 * heat, 0, 1)
            P = probs[true_cls].item()
            axes[r, c].imshow(overlay)
            axes[r, c].axis("off")
            ttl = f"{name}\nP={P:.2f}"
            if r == 0:
                ttl = f"{name}\nP={P:.2f}"
            axes[r, c].set_title(ttl, fontsize=10,
                                 color=("tab:red" if name == "random" else "black"))

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f" [saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--method", default="gradcam++", choices=["gradcam", "gradcam++"])
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--per_fig", type=int, default=5,
                    help="每張圖幾列影像（top-k 會切成 ceil(k/per_fig) 張）")
    ap.add_argument("--relax_classes", nargs="*", default=[],
                    help="這些類別(如 SK)跳過 typiclust<avg3 過濾、用全部 test 影像")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # 載入 5 個模型
    models = {}
    for name, fn in MODELS:
        m, tl, _ = build_model(os.path.join(CKPT_DIR, fn), NUM_CLASSES, device)
        models[name] = (m, tl)
    print(f"loaded {len(models)} models on {device}")

    ds, test_idx = reproduce_test_split()
    cls_to_idx = ds.class_to_idx
    print(f"test set size = {len(test_idx)}")

    summary = {"criteria": "P_typiclust < avg3(margin,coreset,cluster_margin); "
                           "rank by (P_random - avg3) ascending; top-k per class",
               "models": [m for m, _ in MODELS], "topk": args.topk, "classes": {}}

    for tag, cname in TARGET_CLASSES.items():
        cidx = cls_to_idx[cname]
        idxs = [gi for gi in test_idx if ds.samples[gi][1] == cidx]
        print(f"\n=== {tag} ({cname}, label={cidx}) : {len(idxs)} test imgs ===")
        probs = all_probs(models, ds, idxs, device)
        relax = tag in args.relax_classes
        scored = select_top5(probs, relax=relax)
        print(f"  候選({'放寬:全部 test' if relax else '滿足 typiclust<avg3'}) "
              f"= {len(scored)} / {len(idxs)}")
        top = scored[:args.topk]
        for rec in top:
            print(f"   idx={rec['idx']:4d} random={rec['random']:.3f} "
                  f"avg3={rec['avg3']:.3f} typiclust={rec['typiclust']:.3f} "
                  f"r-avg3={rec['random_minus_avg3']:+.3f}")
        summary["classes"][tag] = {
            "class_name": cname, "class_idx": cidx,
            "n_test": len(idxs), "n_candidates": len(scored),
            "selected": [{"idx": r["idx"],
                          "file": os.path.basename(ds.samples[r["idx"]][0]),
                          **{k: r[k] for k in
                             ("avg3", "random", "typiclust",
                              "random_minus_avg3", "typiclust_minus_avg3")},
                          "all_probs": r["probs"]} for r in top],
        }
        # 切成每 per_fig 張一圖（top-20 → 4 張），排版與原本完全一致
        nfig = (len(top) + args.per_fig - 1) // args.per_fig
        for fi in range(nfig):
            chunk = top[fi * args.per_fig:(fi + 1) * args.per_fig]
            suffix = f"_p{fi + 1}" if nfig > 1 else ""
            draw_panel(chunk, models, ds, cidx, device,
                       title=f"{tag} ({cname}) — Grad-CAM++ on true class, ρ=30% AL models"
                             + (f"  ({fi + 1}/{nfig})" if nfig > 1 else ""),
                       out_path=os.path.join(OUT_DIR, f"{tag}_top{args.topk}_gradcam{suffix}.png"),
                       method=args.method)

    sj = os.path.join(OUT_DIR, "al_p30_selection.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(summary, open(sj, "w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {sj}")


if __name__ == "__main__":
    main()
