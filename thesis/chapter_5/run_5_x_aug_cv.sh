#!/bin/bash
# =============================================================================
# Ch5 — augmentation 的 10-fold cross-validation（one-shot，NO active learning）
#        aug 對照：w/o · HF · VF · HF+VF · HF+VF+HVF（同 §4.2 的五欄）
#
# init = **ImageNet**（與 §4.2 data-aug 實驗一致）。10 個 chunk（8:1:1）、視窗循環右移。
# fold 1 = 原本 split ＝ §4.2 資料 → **用 import_fold1_from_4_2.py 匯入、不用重跑**。
# 本腳本預設只跑 **folds 2..5**（之後可補 6..10）。
# 對每個 aug × fold × (portion 5/50/100) × (seed) × (lr ∈ §4.2 per-portion grid) 跑 RUNS 次。
#   - portion 5/50：5 seeds (10 24 38 42 57)
#   - portion 100 ：只 seed 42（全集、seed 無關），仍跑各 fold
# 結果存到隔離樹：classification/exp_results/chapter5_aug_cv/...（不碰主實驗）
# 彙整看表：python3 thesis/chapter_5/aggregate_aug_cv.py
#
# 用法（從 repo 根執行）：
#   # 0) 先匯入 fold 1（§4.2 ImageNet）：python3 thesis/chapter_5/import_fold1_from_4_2.py
#   DEVICE=cuda:0 ./thesis/chapter_5/run_5_x_aug_cv.sh
#   DEVICE=cuda:1 FOLDS="2 3" PORTIONS="5 50" ./thesis/chapter_5/run_5_x_aug_cv.sh
# 可覆寫：DEVICE FOLDS PORTIONS SEEDS LRS RUNS AUGS
# ⚠️ 只能用 FOLDS / SEEDS 分卡（結果檔名 = fold{F}_seed{S}）；用 PORTIONS/AUGS 分卡會撞同一檔。
# 重跑安全：每 (aug,portion,lr) 滿 3 runs 會被 check_existing_results raise → || true 跳過。
# =============================================================================
set -u

pids=()
cleanup() {
    echo ""; echo "Caught Ctrl+C! killing children..."
    for pid in "${pids[@]}"; do ps -p "$pid" >/dev/null 2>&1 && kill -9 "$pid" 2>/dev/null; done
    wait 2>/dev/null; echo "killed."; exit 1
}
trap cleanup SIGINT SIGTERM

######## configs（可由環境變數覆寫）########
device=${DEVICE:-cuda:0}
folds=(${FOLDS:-2 3 4 5})                     # fold 1 用 importer；這裡跑 folds 2..5（全做改 6..10）
portions=(${PORTIONS:-5 50 100})
seeds_default=(${SEEDS:-10 24 38 42 57})     # portion 5/50 用；portion 100 一律只 seed 42
runs=${RUNS:-3}
augs=(${AUGS:-no_aug hf vf hfvf 4x})          # 五欄：w/o / HF / VF / HF+VF / HF+VF+HVF
task=hard
epoch=20
exp_path=./exp_results/chapter5_aug_cv
###########################################

cd "$(dirname "$0")/../../classification" || exit 1   # run_first_iter_cv.py 從 classification/ 跑

aug_args() {   # 把 aug 簡稱轉成 run_first_iter_cv.py 的 flag
    case "$1" in
        noaug|no_aug) echo "--no_data_aug" ;;                       # w/o aug (1x)
        hf)   echo "--aug_factor 2 --flip_type horizontal" ;;       # HF (2x)
        vf)   echo "--aug_factor 2 --flip_type vertical" ;;         # VF (2x)
        hfvf) echo "--aug_factor 3" ;;                              # HF+VF (3x)
        4x)   echo "--aug_factor 4" ;;                              # HF+VF+HVF (4x)
        *)    echo "ERR" ;;
    esac
}

lr_grid_for() {   # §4.2 各 portion 用的 lr 網格（LRS 環境變數可全域覆寫）
    if [ -n "${LRS:-}" ]; then echo "$LRS"; return; fi
    case "$1" in
        5)   echo "7e-5 1e-4 3e-4 5e-4 7e-4" ;;
        50)  echo "5e-5 7e-5 1e-4 3e-4 5e-4" ;;
        100) echo "1e-5 5e-5 1e-4 5e-4" ;;
        *)   echo "7e-5 1e-4 3e-4 5e-4 7e-4" ;;
    esac
}

echo "=== Ch5 aug CV ===  device=$device folds=(${folds[*]}) portions=(${portions[*]}) augs=(${augs[*]}) runs=$runs"

# 平行策略：一次只跑「一個 lr 的 N runs」（同卡平行），lr 之間排隊跑完才換下一個。
#   → 同時最多 = runs 個 process（預設 3）。省記憶體；24GB 卡也很安全。
# 寫檔安全：run_first_iter_cv.py 用 fcntl 檔案鎖保護 append → 多 process 寫同一 fold/seed 檔不會 race。

for portion in "${portions[@]}"; do
    if [ "$portion" = "100" ]; then seeds=(42); else seeds=("${seeds_default[@]}"); fi
    lrs=($(lr_grid_for "$portion"))
    for fold in "${folds[@]}"; do
        for aug in "${augs[@]}"; do
            aflags=$(aug_args "$aug")
            for seed in "${seeds[@]}"; do
                echo "  portion=$portion fold=$fold aug=$aug seed=$seed → 每個 LR 的 $runs runs 一起跑、LR 之間排隊"
                for lr in "${lrs[@]}"; do
                    for r in $(seq 1 "$runs"); do      # 一個 lr 的 N runs 同卡平行
                        python3 ./run_first_iter_cv.py \
                            --task_type "$task" --fold "$fold" \
                            --portion "$portion" --seed "$seed" \
                            --pretrained_weights imagenet \
                            $aflags --lr "$lr" --epoch "$epoch" \
                            --device "$device" --exp_path "$exp_path" || true &
                        pids+=($!)
                    done
                    wait; pids=()                       # 等這個 lr 的 runs 跑完才換下一個 lr
                done
            done
        done
    done
done

echo "=== all done ==="
