#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.2　混淆矩陣圖 (Confusion matrix)
==================================
讀 thesis/chapter_5/confusion/{strategy}_p{portion}_seed{seed}.json，畫 7×7 混淆矩陣：
  - rows = 真實類別 (True)、cols = 預測類別 (Predicted)，每格 annotate **張數**。
  - 底色 = 該列 row-normalized 比例（recall 視角；避免被多數類張數淹沒），數字仍是 count。
  - 每列尾端額外一欄 = 該類別單獨準確率（recall = 對角/該列總數，%）。
  - title = "<Method>, ρ=XX%, (Acc=XX%)"（Acc = 整體 test set 準確率）。
  - **AL 圖**：相對 Random「進步最顯著的格子」其 annotation 以**粗體紅字**標出
    （對角線=多對幾張；非對角線=少錯幾張）。

預設一次畫 Random + 七種 AL（共八張），portion=30%、seed=42。

從 repo root：
    python3 thesis/chapter_5/plot_5_2_confusion.py
    python3 thesis/chapter_5/plot_5_2_confusion.py --portion 30 --strategy margin
圖存到 thesis/chapter_5/figs/confusion/。
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CONF_DIR = os.path.join(HERE, "confusion")
OUT_DIR = os.path.join(HERE, "figs", "confusion")

ALL_STRATEGIES = ["random", "conf", "margin", "entropy", "coreset",
                  "typiclust", "badge", "cluster_margin"]
LABEL = {
    "random": "Random", "conf": "Confidence", "margin": "Margin", "entropy": "Entropy",
    "coreset": "Core-set", "typiclust": "TypiClust", "badge": "BADGE",
    "cluster_margin": "Cluster-Margin",
}
NAME_DISP = {"Normal": "Healthy", "Seborrhoeic keratosis": "SK", "Solar lentigo": "SL"}
# 混淆矩陣 row/col 顯示順序（依類別張數遞減）
DISPLAY_ORDER = ["Healthy", "Nevus", "SL", "Eczema", "Psoriasis", "SK", "Vitiligo"]

FONT_TITLE, FONT_LABEL, FONT_TICK, FONT_CELL = 24, 19, 15, 14


def _load(strategy, portion, seed):
    path = os.path.join(CONF_DIR, f"{strategy}_p{portion:g}_seed{seed}.json")
    if not os.path.isfile(path):
        return None
    return json.load(open(path))


def draw(data, out, cm_rand=None):
    cm = np.array(data["matrix"], dtype=int)
    n = cm.shape[0]
    # 依指定顯示順序（依類別張數遞減）重排 row/col：Healthy, Nevus, SL, Eczema, Psoriasis, SK, Vitiligo
    disp_all = [NAME_DISP.get(c, c) for c in data["classes"]]
    perm = [disp_all.index(nm) for nm in DISPLAY_ORDER if nm in disp_all]
    cm = cm[np.ix_(perm, perm)]
    disp = [disp_all[p] for p in perm]
    if cm_rand is not None:
        cm_rand = cm_rand[np.ix_(perm, perm)]
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = cm / np.where(row_sum == 0, 1, row_sum)            # row-normalized → 底色
    recall = 100.0 * np.diag(cm) / np.where(row_sum[:, 0] == 0, 1, row_sum[:, 0])

    has_delta = cm_rand is not None and cm_rand.shape == cm.shape
    delta = None
    if has_delta:
        rs_r = cm_rand.sum(axis=1)
        recall_r = 100.0 * np.diag(cm_rand) / np.where(rs_r == 0, 1, rs_r)
        delta = recall - recall_r                            # per-class Δ vs random (pp)

    GREEN = "#1A8C1A"

    def _top3(vals):
        return set(int(k) for k in np.argsort(vals)[::-1][:3])

    # 只在 Δ 欄標 Top-3（取最大的正進步，不取絕對值）；Per-class Acc 不標粗體；Random 圖無 Δ 欄
    top3_delta = _top3(delta) if has_delta else set()

    # AL 圖含 Δ 欄較寬；Random 只有 Per-class Acc 欄 → 較窄（cell 尺寸維持一致）
    fig, ax = plt.subplots(figsize=(11 if has_delta else 9.6, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    # 每格 annotate 張數
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=FONT_CELL,
                    color="white" if norm[i, j] > 0.55 else "black")

    # 黑色矩陣外框：把 7×7 confusion matrix 與右側 per-class 欄區隔開
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-0.5, -0.5), n, n, fill=False, edgecolor="black",
                           linewidth=2.2, zorder=5))

    # 右側欄位（置中）：Per-class Acc(%) [+ Δ over Random(pp)]；各欄 Top-3 綠色粗體
    RX1 = n + 0.55                                           # Per-class Acc 欄中心
    RX2 = n + 2.1                                            # Δ over Random 欄中心（拉開避免標題重疊）
    for i in range(n):
        ax.text(RX1, i, f"{recall[i]:.1f}%", ha="center", va="center",
                fontsize=FONT_CELL, color="#222222")
    ax.text(RX1, -0.85, "Per-class\nRecall", ha="center", va="center",
            fontsize=FONT_TICK - 1, fontweight="bold", color="#222222")
    if has_delta:
        for i in range(n):
            hi = i in top3_delta
            ax.text(RX2, i, f"{delta[i]:+.1f}", ha="center", va="center",
                    fontsize=FONT_CELL, fontweight="bold" if hi else "normal",
                    color=GREEN if hi else "#222222")
        ax.text(RX2, -0.9, r"$\Delta$ over" + "\nRandom", ha="center", va="center",
                fontsize=FONT_TICK, fontweight="bold", color="#222222")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(disp, rotation=40, ha="right", fontsize=FONT_TICK)
    ax.set_yticklabels(disp, fontsize=FONT_TICK)
    ax.set_xlabel("Predicted", fontsize=FONT_LABEL, labelpad=8)
    ax.set_ylabel("True", fontsize=FONT_LABEL, labelpad=8)
    ax.set_xlim(-0.5, (n + 2.7) if has_delta else (n + 1.25))  # Random 不預留 Δ 欄空白
    # title 置中於 7×7 矩陣上方（不含 recall 欄），不用粗體、ρ% 後直接接括號
    ax.text((n - 1) / 2.0, -0.95,
            f"{LABEL.get(data['strategy'], data['strategy'])}, "
            rf"$\rho$={data['portion']:g}% (Acc={100 * data['acc']:.1f}%)",
            ha="center", va="bottom", fontsize=FONT_TITLE - 2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized fraction", fontsize=FONT_TICK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}  (Acc={100 * data['acc']:.1f}%"
          + (f", top3Δ={[disp[i] for i in sorted(top3_delta, key=lambda k: -delta[k])]}"
             if has_delta else "") + ")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", nargs="+", default=ALL_STRATEGIES)
    ap.add_argument("--portion", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    rand = _load("random", args.portion, args.seed)
    cm_rand = np.array(rand["matrix"], dtype=int) if rand else None

    for strat in args.strategy:
        data = _load(strat, args.portion, args.seed)
        if data is None:
            print(f"[warn] no data for {strat} @ ρ={args.portion:g}% — 先跑 run_5_2_confusion.py")
            continue
        out = os.path.join(args.out_dir, f"5_2_cm_{strat}_p{args.portion:g}_seed{args.seed}.png")
        draw(data, out, cm_rand=None if strat == "random" else cm_rand)


if __name__ == "__main__":
    main()
