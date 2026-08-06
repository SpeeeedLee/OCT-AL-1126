#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay the RED ground-truth cell-nuclei contour on a few TRAIN OCT images and
save each as its own PNG at the image's ORIGINAL resolution (no matplotlib frame).

Run (repo root): python3 thesis/chapter_5/segmentation/plot_gt_overlay.py
  --images "train/a.png" "train/b.png" ...   (optional; defaults below)
"""
import os
import argparse
import cv2

HERE = os.path.dirname(__file__)
IMG_DIR = "ds/segmentation_correct/image"
CELL_DIR = "ds/segmentation_correct/cell"
OUT_DIR = os.path.join(HERE, "figs")
DEFAULT = [
    "train/2_20181119_113828_77-face-N.png",
    "train/1_20180928_090738_56-nose-N.png",
    "train/1_20180525_163552_l_t abdomen.png",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=DEFAULT)
    ap.add_argument("--thickness", type=int, default=1, help="red contour line width (px)")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    for i, name in enumerate(a.images, 1):
        img = cv2.imread(os.path.join(IMG_DIR, name), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(os.path.join(CELL_DIR, name), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            print(f"  [skip] missing image/mask for {name}"); continue
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        _, binm = cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (0, 0, 255), a.thickness)  # red (BGR)
        out = os.path.join(OUT_DIR, f"gt_overlay_{i}.png")
        cv2.imwrite(out, rgb)
        print(f"saved -> {out}  ({img.shape[1]}x{img.shape[0]} px, {len(contours)} nuclei) [{name}]")


if __name__ == "__main__":
    main()
