#!/usr/bin/env bash
# Cold-start FM one-shot fine-tuning 批次驅動（§5.2）。從 repo root 執行。
#
#   DEVICE=cuda:4 MODELS="dinov2:base" ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh
#
# 可覆寫 MODELS / PORTIONS / DEVICE / PARALLEL：
#   MODELS="dinov2:base clip:large retfound:oct" PORTIONS="2.5 10 20" \
#       DEVICE=cuda:5 PARALLEL=3 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh
#
# 協定：θ² SimCLR 起始、aug4、AdamW+LinearLR(1→0)+CE、batch16、epoch20、lr sweep × 3 runs。非 AL。
# ⚠️ 不需 seed：cold-start 選樣是 **deterministic（固定 labeled ID）**，seed 只會擾動訓練雜訊、
#    不改變影像 → 每個 (model,portion) 只需 tune lr、跑 3 runs、取 best-lr report mean±std。
#    結果寫 {model}_bs16.json（無 seed 後綴）。
# 單卡循序跑（一個 model 一個 portion = lr-grid × 3 runs）。分卡：開多個 shell、各設不同
# DEVICE + MODELS 子集。重跑安全：每個 (model,portion,lr) 已滿 3 runs 自動 skip（印 [skip]）。
# ⚠️ 若某 model/portion 的選樣 ID JSON 不存在 → run_coldstart_fm.py raise，本腳本**直接中止**
#    （刻意不 `|| true`；先跑 select_coldstart.py 補上再重跑）。
set -u

# 預設 = 全部 13 個已產出的 model（含我們自己的 simclr + 不同 size 的 dinov2 / clip / resnet）
MODELS="${MODELS:-simclr:resnet18 resnet_imagenet:resnet18 resnet_imagenet:resnet50 \
resnet_imagenet:resnet101 dinov2:small dinov2:base dinov2:large clip:base clip:large \
radimagenet:resnet50 biomedclip:base retfound:oct medimageinsight:base}"
PORTIONS="${PORTIONS:-2.5 10 20}"
DEVICE="${DEVICE:-cuda:0}"
PARALLEL="${PARALLEL:-1}"        # 一個 lr 的多個 run 同時丟同一張 GPU；設 3 = 三 run 並行

cd "$(git rev-parse --show-toplevel)" || exit 1
echo "MODELS=[$MODELS]"
echo "PORTIONS=[$PORTIONS]  DEVICE=$DEVICE  PARALLEL=$PARALLEL  (no seeds — fixed selected set)"

trap 'echo "Interrupted"; exit 130' INT

for m in $MODELS; do
  for p in $PORTIONS; do
    echo ""
    echo "########## $m  ρ=${p}%  ($DEVICE) ##########"
    python3 thesis/chapter_5/coldstart_fm/run_coldstart_fm.py \
      --model "$m" --portion "$p" --device "$DEVICE" --parallel_runs "$PARALLEL" \
      || { echo "[ABORT] $m ρ=${p}% failed (missing ID JSON? run select_coldstart.py)"; exit 1; }
  done
done
echo "ALL DONE."
