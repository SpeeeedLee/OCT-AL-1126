# Chapter 5 — 主動學習深入分析（規劃與追蹤）

> 本檔記錄第五章的實驗規劃、設計決策與待辦。**目前為規劃階段，尚未寫 code。**
> 與 `thesis/CLAUDE.md`（全論文追蹤）、`thesis/chapter_4/README.md`（Ch4）並列。

## 範圍與前提

- 第五章起 **聚焦三大策略各自的最佳 AL 方法**，不再跑全部七種。依 Ch4
  `al_curve_each_best.py`（最早達 88.2% 者）目前的前三名為：
  - **Uncertainty → Margin**
  - **Diversity → Coreset**
  - **Hybrid → Cluster-Margin**
  - ⚠️ 待 5-seed 資料最終定版後再鎖定（順序可能微調；以最終 al_curve 為準）。
- **既有資產：每個 AL 方法 × 每個 portion 所選的 data index 都已存檔**
  （`classification/exp_results/.../AL_simclr/labeled_ids/{strategy}_seed{seed}_bs16.json`）。
  → 任何「重訓某 portion 的模型」「分析某次選樣」都可直接由 index 重建，成本低。
- 共同設定沿用 Ch4 §4.1（ResNet-18、θ² SimCLR 初始化、aug4、per-seed best-lr → mean over seeds、std ddof=1）。

---

## 5.1　主動學習超參數敏感度（b₀ 與 b）

AL 軌跡由兩個量界定：**b₀ = 初始隨機標註比例**、**b = 每輪查詢間隔**。Ch4 主結果固定
b₀ = 2.5%、b = 2.5%。本節各別變化其一（**不交叉**，以免計算量爆炸）來 justify 此選擇。

### 5.1.1　變化 b₀（固定 b = 2.5%）
- b₀ ∈ {2.5, 5, 10, 20}%。其後一律 b = 2.5% 跑同樣的三種策略到 ρ=60%。
- 目的：justify 2.5% 的初始隨機池已足夠（或找出更好的 b₀）。
- **比較須在相同總 ρ 下對齊**：b₀ 較大的軌跡「較晚才開始 AL」（第一個 AL 點落在較高 ρ）。
  以「達 target 的 ρ」「曲線下面積 / 同 ρ 之 mean acc」比較，而非看起點。

```bash

# ---- b₀ = 10% ----
B0=10 DEVICE=cuda:3 STRATEGIES="margin"  SEEDS="10 24" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=10 DEVICE=cuda:3 STRATEGIES="margin"  SEEDS="38 42" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=10 DEVICE=cuda:3 STRATEGIES="margin"  SEEDS="57" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下

B0=10 DEVICE=cuda:5 STRATEGIES="coreset"  SEEDS="10 24" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=10 DEVICE=cuda:5 STRATEGIES="coreset"  SEEDS="38 42" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=10 DEVICE=cuda:5 STRATEGIES="coreset"  SEEDS="57" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下

B0=10 STRATEGIES="coreset" SEEDS="10" DEVICE=cuda:2 ./thesis/chapter_5/run_5_1_b0_ablation.sh
B0=10 STRATEGIES="coreset" SEEDS="24" DEVICE=cuda:4 ./thesis/chapter_5/run_5_1_b0_ablation.sh
B0=10 STRATEGIES="coreset" SEEDS="38" DEVICE=cuda:5 ./thesis/chapter_5/run_5_1_b0_ablation.sh
B0=10 STRATEGIES="coreset" SEEDS="57" DEVICE=cuda:6 ./thesis/chapter_5/run_5_1_b0_ablation.sh


B0=10 DEVICE=cuda:0 STRATEGIES="cluster_margin" SEEDS="10 24" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=10 DEVICE=cuda:0 STRATEGIES="cluster_margin" SEEDS="38 42" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=10 DEVICE=cuda:9 STRATEGIES="cluster_margin" SEEDS="57" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下


# ---- b₀ = 20% ----
B0=20 DEVICE=cuda:9 STRATEGIES="margin"  SEEDS="10 24" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:8 STRATEGIES="margin"  SEEDS="38 42" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:8 STRATEGIES="margin"  SEEDS="57" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下

B0=20 DEVICE=cuda:7 STRATEGIES="coreset" SEEDS="10" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:7 STRATEGIES="coreset" SEEDS="24" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:1 STRATEGIES="coreset" SEEDS="38" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:6 STRATEGIES="coreset" SEEDS="42" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:6 STRATEGIES="coreset" SEEDS="57" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下


# hybrid 改跑 cluster_margin（b₀=10、20，各 5 seeds）
B0=10 DEVICE=cuda:9 STRATEGIES="cluster_margin" SEEDS="10 24 38 42 57" ./thesis/chapter_5/run_5_1_b0_ablation.sh
B0=20 DEVICE=cuda:9 STRATEGIES="cluster_margin" SEEDS="10 24" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:7 STRATEGIES="cluster_margin" SEEDS="38" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:5 STRATEGIES="cluster_margin" SEEDS="42" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下
B0=20 DEVICE=cuda:5 STRATEGIES="cluster_margin" SEEDS="57" ./thesis/chapter_5/run_5_1_b0_ablation.sh # 已下

python3 thesis/chapter_5/plot_b0_ablation.py
```

### 5.1.2　變化 b（固定 b₀ = 2.5%）
- b ∈ {2.5, 5, 10, 20}%。
- 目的：justify 2.5% 間隔最佳。**此結論先前已大致驗證，信心較高。**
- 直覺：b 越小 → 每次用「最新、較準的模型」重新評估未標註池 → 查詢品質越高；b 越大則
  一次選太多、後段樣本是用較舊模型挑的、且批內冗餘上升 → 預期 b 越小越好（但 retrain 次數越多、計算越貴）。

### lr 注意事項
- **初始 b₀ 步 = 該 seed 的 random 選樣**（同 seed→同子集），故直接沿用 θ² cold-start 在
  該 **(b₀, seed)** 的 best lr，**免重掃**。已驗證 ρ=5/10/20 × 5 seeds 在 `cold_start_simclr` 全查得到
  （per-seed 差很多，如 ρ=5：3e-5～3e-4）。
- 機制：`run_AL.py` 新增 `--coldstart_lr_path`（指向真正的 `./classification/exp_results`），讓初始步
  的 lr 查表用真 cold-start 樹，而結果仍寫到隔離的 `ch5_b0_ablation/b0_<B0>/`。
- 後續 ρ>b₀ 步仍走 sweep + best-val（option A）。

```bash
B=10 DEVICE=cuda:0 STRATEGIES="margin"         SEEDS="10 24" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=5  DEVICE=cuda:0 STRATEGIES="margin"         SEEDS="10 24" ./thesis/chapter_5/run_5_1_b_ablation.sh

B=10 DEVICE=cuda:2 STRATEGIES="margin"         SEEDS="38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=5 DEVICE=cuda:2 STRATEGIES="margin"         SEEDS="38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh


B=10 DEVICE=cuda:0 STRATEGIES="coreset"         SEEDS="10 24" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=5 DEVICE=cuda:0 STRATEGIES="coreset"         SEEDS="10 24" ./thesis/chapter_5/run_5_1_b_ablation.sh

B=10 DEVICE=cuda:2 STRATEGIES="coreset"         SEEDS="38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh 
B=5 DEVICE=cuda:2 STRATEGIES="coreset"         SEEDS="38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh 

B=10 DEVICE=cuda:4 STRATEGIES="cluster_margin"         SEEDS="10 24" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=5 DEVICE=cuda:4 STRATEGIES="cluster_margin"         SEEDS="10 24" ./thesis/chapter_5/run_5_1_b_ablation.sh

B=10 DEVICE=cuda:4 STRATEGIES="cluster_margin"         SEEDS="38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=5 DEVICE=cuda:4 STRATEGIES="cluster_margin"         SEEDS="38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh

# 6/14: 加入b=20%的，這樣才更有望看到performance collapse!
B=20 DEVICE=cuda:7 STRATEGIES="margin"         SEEDS="10 24 38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=20 DEVICE=cuda:6 STRATEGIES="coreset"        SEEDS="10 24 38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh
B=20 DEVICE=cuda:1 STRATEGIES="cluster_margin" SEEDS="10 24 38 42 57" ./thesis/chapter_5/run_5_1_b_ablation.sh


python3 thesis/chapter_5/plot_b_ablation.py
```

---

## 5.2　以 cold-start AL 演算法改良初始選樣（b₀ 不再純隨機）

**動機**：目前 b₀ 是「純隨機」。改用 *cold-start / low-budget AL*（不需任何標註、純用 SSL 表示空間
密度/覆蓋挑第一批）來選 b₀，期望比隨機更好的起點 → 抬升整條軌跡。注意「cold-start AL」此處指
**文獻意義的低預算主動選樣**（非 codebase 的被動 `cold_start_*` baseline）。

### 候選方法（代表性高、引用多；皆能「零標註自選第一批」）
1. **TypiClust**（Hacohen et al., *ICML 2022*）— 對 SSL 特徵分群，挑各群最典型（最稠密）點。
   **本 codebase 已實作**（`diversity_correct.py::typiclust`），可直接拿來選 b₀。低預算公認 SOTA 之一。
   → **5.2 近期就用這個當 b₀ 選樣器。**
2. **ProbCover**（Yehuda et al., *NeurIPS 2022*，"Active Learning Through a Covering Lens"）—
   把選樣視為 *Max Probability Cover*：在 SSL 嵌入空間以半徑 δ 的球覆蓋資料，貪婪挑「能覆蓋最多
   未覆蓋高密度點」者。實作輕（~60 行、吃 frozen 特徵）。**若要再加一個新 cold-start，這是首選；可晚點做。**

### ⏸ 暫緩實作（晚點再說）
- **"Making Your First Choice" / CSVAL**（Chen et al., MIDL 2023）— 忠實版需要在 **SSL 預訓練時逐 epoch
  記錄每張的對比信心 μ̂**（要動 SimCLR 訓練、重跑一次），**太麻煩 → 暫緩**。
  （若之後仍想做，可用「k-NN 密度 proxy」版規避 μ̂，但目前先不做。）
- **USL / USL-T**（Wang et al., ECCV 2022）— training-free 版雖不難，USL-T 需端到端學分群＋防 collapse 較重；
  與上者一併**暫緩**，related work 提及即可。

> 共通：都吃 frozen SSL 特徵、無需標註即可選 b₀；做成「b₀ 選樣器」即可（之後 b 步仍用 5.1 的策略）。
> **近期 5.2 範圍 = TypiClust（已有）為主，必要時加 ProbCover；其餘兩法暫緩。**

### 相關近期文獻：用 *foundation model* 的 embedding 做 cold-start 選樣（2024–2026）
**與本論文差異**：我們是用「**自己 pretrain 的 SimCLR θ²** 特徵」做 cold-start AL；以下這批論文改用**大型醫療/通用 foundation model 的現成 embedding**（多數 frozen、不需自訓）→ embedding → clustering → 挑各群代表當初始標註集。可作 related work，並可考慮拿 DINOv2 等 embedding 當我們 SimCLR cold-start 的對照 baseline。

1. **Foundation Model Makes Clustering A Better Initialization For Cold-Start Active Learning**（arXiv:2402.02561, 2024）
   - 核心＝foundation embedding → clustering → 選代表樣本當 AL 初始集；在醫療影像 classification + segmentation 驗證，勝過 random 等 baseline。
   - **試過的 foundation model**：TorchXRayVision (TXRV)、CXR Foundation (CXRF)、REMEDIS（皮膚/乳攝/病理/眼底/胸X 多域 SSL）、ImageNet-supervised DenseNet-121（baseline）。胸X 專用模型(TXRV/CXRF) > 多域 REMEDIS。
2. **MedCAL-Bench: A Comprehensive Benchmark on Cold-Start Active Learning with Foundation Models for Medical Image Analysis**（arXiv:2508.03441, 2025）
   - 系統性 benchmark：foundation model 當 feature extractor × 多種選樣策略。**DINO 家族在 segmentation 最佳**；不同策略各擅長不同 dataset。
   - **試過的 foundation model（14 個）**：SAM(ViT-B/H)、MedSAM(ViT-B)、SAM2(2b+/2.1b+)；CLIP(RN50x64/ViT-L_14/@336)、MedCLIP(ResNet/ViT)；DINO(ViT-S_16/B_16)、DINOv2(ViT-B_14/G_14)；ImageNet-1k ResNet18（baseline）。
3. **From Cold Start to Active Learning: Embedding-Based Scan Selection for Medical Image Segmentation**（arXiv:2601.18532, 2026）
   - foundation embedding + clustering（自動定群數、群內 proportional sampling）→ 接 uncertainty-based AL；在 X-ray/MRI 三 dataset 勝過 random。
   - **試過的 foundation model**：僅一個 — **ResNet-50 pretrained on RadImageNet**（~135 萬張放射影像）；框架對降維/特徵抽取方法 agnostic，但未實測其他模型。

> 觀察：三篇用的 foundation model **沒有重疊到 OCT/皮膚 OCT 專用模型**（多為胸X/放射/通用 SAM·CLIP·DINO），且都沒用 Microsoft **MedImageInsight**。→ 我們若改用涵蓋 dermoscopy/OCT 的 embedding（見下）做 cold-start，有區隔空間。

### 實驗設計
- baseline：隨機 b₀（Ch4 主結果）。對照：TypiClust / ProbCover / (Making-First-Choice 或 USL) 選 b₀。
- b₀ 大小取 5.1 結論的最佳值；後續 b 步固定用三大策略各自最佳。
- 看「換更聰明的 b₀」是否讓整條軌跡（尤其低 ρ 段）顯著上移。

### ✅ 已實作：foundation-model embedding + clustering 選初始集（`coldstart_fm/`）
忠實實作上述 **2026 那篇（arXiv:2601.18532）** 的選樣演算法，且做成 **model-agnostic**：任何 FM 當
frozen 特徵抽取器，後面接同一套 clustering 選 b₀。詳見 `thesis/chapter_5/coldstart_fm/README.md`。
- **演算法**：FM embedding → **t-SNE 2D** → 對每個 k 跑 KMeans 取 **silhouette** 最高者 k̂ →
  KMeans(k̂)、各群 **medoid** 當種子 → 剩餘預算 **按群大小比例** 分配 → 群內 **greedy farthest-point**。
  （合成 7-blob 資料可正確回收 k̂=7；真實 OCT embedding 上 silhouette 多選 **小 k̂≈2–4**＝這些 embedding
  空間的自然結構是少數大群、非 7 診斷類，故預算主要靠 proportional + FPS 覆蓋鋪開。）
- **三步驟（皆 repo root 執行）**：
  1. `coldstart_fm/extract_embeddings.py --all --device cuda:N` → 抽 D_train(2032) 特徵存 `embeddings/{model}.pt`
     （ImageFolder 順序＝train index，與 `run_AL`/`get_data` 一致；灰階→RGB；有 head 的模型取 penultimate）。
  2. `coldstart_fm/select_coldstart.py --model <id>` → 在 **ρ=2.5/10/20%** 選樣，存 `labeled_ids/{model}.json`
     （**沿用既有 labeled_ids schema**：`{portion:{selected,cumulative,k_hat,...}}`）。
  3. 訓練：① 一鏡到底 one-shot `coldstart_fm/run_coldstart_fm.py` + `.sh`（獨立檔、未動 core；θ²/aug4/
     sweep-lr；結果寫 `coldstart_fm_simclr/{model}_bs16.json`）。**⚠️ 不需 seed**：選樣 deterministic（固定 ID）→
     seed 只擾動訓練雜訊；故 tune lr、跑 3 runs、取 **best-lr → mean±std over its runs**（非 per-seed-over-
     subsets）。`.sh` 已移除 `SEEDS`（env 只剩 `MODELS/PORTIONS/DEVICE/PARALLEL`）；舊的 seed 檔會被 plot
     當額外 reps pool 進去。`--parallel_runs 3` 可把一個 lr 的 3 runs 同卡並行。畫圖 `plot_coldstart_fm.py`
     （一 portion 一張長條圖，best-lr→mean±std；Random 為 θ² cold-start 5 random subsets 參考）。
     ② 當 AL 初始池＝**直接用既有** `run_AL.py --resume_labeled_ids <json> --resume_from 2.5`（**零修改 core**）。
- **已產出（13 個模型 × ρ{2.5,10,20}，全數 integrity-verified）**：**`simclr:resnet18`（＝我們自己的 θ²
  SimCLR backbone，512-d penultimate、codebase 原生 transform 全解析度；in-domain 自訓對照點，與下游
  one-shot trainer 同一個 ckpt `resnet18_simclr_lr0.0002_bs256_ep500.pkl`）**、`resnet_imagenet:{18,50,101}`、
  `dinov2:{small,base,large}`、`clip:{base,large}`、`radimagenet:resnet50`（＝原 paper 抽取器）、
  `biomedclip:base`、**`retfound:oct`**（retinal OCT，HF gated 已取得 access；載 `.pth` 進 timm ViT-L/16、
  missing=0/unexpected=0、無 remote code）、**`medimageinsight:base`**（用既有 `feature_extract/MedImageInsights/`
  本地 package＝Microsoft UniCL；裝 einops/mup/fvcore/tenacity；吃 base64 bytes 故走 `input_mode='path'`；
  **修了 bug**：原 `load_model()` 漏 `model.eval()`→dropout 開著特徵不確定，已強制 eval）。
- **k̂ 觀察**：silhouette 在真實 OCT embedding 多選小 k̂（2–4）；retfound:oct 的 silhouette 最高（k̂=2, 0.51）。
- **未跑**：`retfound:cfp`（color-fundus，比 :oct 更遠離皮膚 OCT；同一 gated access）。
- **域不匹配（誠實寫進論文）**：我們是 **灰階皮膚 OCT**，這些 FM 沒一個完全匹配（RadImageNet=放射、
  BiomedCLIP=生醫文獻、RETFound=*視網膜* OCT）→ 這正是 §5.2 要比的對象（現成 FM embedding vs 自訓 SimCLR θ²）。

---

## 5.3　所選影像之分析（量化 + 質性）

**全部可由已存的 labeled_ids index 重建，不需重跑 AL。** 預設在 **ρ=30%** 比較。

### 【優先做】混合矩陣（confusion matrix）對比
- 是的，7×7 的就叫 **confusion matrix**（列=真實類別、欄=預測類別）。
- 在 ρ=30% 下，用 **random vs 各 AL 方法** 所選 index 重訓模型，畫各自的 7×7 confusion matrix
  （在固定的 test set 上）。
- 目的：看 AL 相對 random **在哪些類別**把對角線（正確率）拉高、把哪些易混淆對的off-diagonal壓低
  → 解釋「AL 靠改善哪幾類來提升整體 acc」。
- 多 seed 取平均的 confusion matrix（或差值矩陣 AL − random）會更有說服力。

### 量化（一）：所選影像的 GT 類別分布 — `plot_5_3_selection_dist.py`（✅ 已實作）
- 各方法所選集合 over 7 個 ground-truth label 的分布。資料讀 `AL_simclr/labeled_ids/`
  （即主線 **b₀=2.5%、b=2.5%、5 seeds**；cumulative=該 portion 累積標記集）。
- **baseline = 整個 train set 的類別比例**（= random 的期望，analytic 無模擬雜訊），畫成灰虛線/灰 bar。
- 樣式全對齊 Ch4 AL 折線圖：色/marker（`plot_al_curve.py` GROUPS）、`figsize=(12,8)`、
  FONT 26/20/15、分組 legend（Uncertainty/Diversity/Hybrid 三欄 + Random 獨立底列）。圖無 title（caption 自寫；trend 例外，title=類別名）。輸出到 `figs/5_3_*`。
- 三種圖（`--plot`）：
  - `dist` — 各方法每類 share(%) + baseline。
  - `diff` — 相對 baseline 的偏差；`--diff pp`（百分點，預設）或 `relative`（相對%）。y 軸 `Over / Under Sampling vs. Random (%)`。
  - `trend` — 橫軸 ρ、縱軸某類 share(%) 隨 portion 變化（`--class`，預設 Normal）+ baseline 虛線；看「該類比重從哪個 ρ 開始偏離」。
- 預設策略：dist/diff = 全七種（檔名 `all`）；trend = margin/coreset/cluster_margin。明確 `--strategy` 則照給的。

```bash
# 分布圖 + 差異圖（全七種，預設 ρ=22.5%）；--portion 換比例
python3 thesis/chapter_5/plot_5_3_selection_dist.py --portion 30 --plot both

# Normal 比重 vs portion（全七種）
python3 thesis/chapter_5/plot_5_3_selection_dist.py --plot trend \
    --strategy conf margin entropy coreset typiclust badge cluster_margin
# 其餘六類同上，逐一換 --class "Eczema" / "Nevus" / "Psoriasis" / "Seborrhoeic keratosis" / "Solar lentigo" / "Vitiligo"
```

- **觀察**：除 **TypiClust** 外（density-based，貼著原分佈、Normal 維持 ~40%），其餘六法都從 ρ=5% 起把多數類
  **Normal 壓到 ~25–30%**（ρ≈12.5–17.5% 觸底）、把少數類拉高；uncertainty 因避高信心、coverage 型 diversity（k-center）因密集區少數點即覆蓋、BADGE/Cluster-Margin 兼具 → 同向。
- **與 AL 表現對照（重要）**：TypiClust 是唯一不壓 Normal 的，**也是 AL 最差**（達 88.2% 需 ρ≈42.5% vs 其餘 25–30%；ρ=60% 僅 90.5% vs ~94–95%）。但它是低預算法、用在中高預算本就吃虧（ρ=10% 時其實與他法持平），故「壓低 Normal=對」是此預算區間的相關現象，深層因是「選 informative 樣本」，類別偏移只是 symptom。
- **TypiClust 原 paper（Hacohen et al. ICML 2022, arXiv:2202.02794）對 class imbalance 的處理**：
  - §4.3.2 用 **TV distance(labeled set 類別分布, ground-truth 類別分布)** 當指標，主張 TypiClust 此距離最低
    →「queries with better class balance」；並稱「labeled set approximately class-balanced，即使選樣不看 label」。
  - §4.3.5 / App G.1.2 有在 **imbalanced CIFAR-10**（Munjal et al. 2020）測試：**低預算贏、高預算輸**。
  - **關鍵解讀（可寫進論文 discussion）**：該指標是「貼近資料真實分布」。CIFAR-10 本身平衡 → 貼近=均衡，看起來是優點；
    但在我們 **imbalanced 的皮膚資料**，同一機制 = **忠實複製多數類佔比（Normal~40%）= 不修正不平衡**，反成劣勢。
    其 paper **未**做「多數 vs 少數類 over/under-sampling」的逐類分析 → 我們這組 per-class share / trend 圖正補上此視角。

### 量化（二）：剩餘未標註集的 uncertainty 分布
- uncertainty 方法（Margin）在 ρ=30% 時，對**剩餘未標註池**的 uncertainty score 分布長相
  （直方圖）→ 觀察「還剩多少高不確定樣本」、分布是否隨 ρ 變平。

### 量化（三）：類別重分布消融 — `run_5_3_redistribution.py` + `plot_5_3_redistribution.py`（✅ 已實作）
**問題**：AL 贏，是因為它把「各類別標註比例」重分布到特定比重？還是「逐影像挑選」本身也關鍵？
**做法（切開兩者）**：沿用 Margin / Core-set / Cluster-Margin 在各 portion、跨 5 seeds 的
**「各類別累積選取張數平均」**當 target 各類張數（= class redistribution），但**實際影像於對應類別中
『隨機』抽取**（每 seed 各自一組隨機抽、largest-remainder 取整使總和 = ρ·2032）。
- **協定 = cold-start one-shot（非 iterative）**：θ²-SimCLR(lr2e-4/bs256/ep500) backbone→7-class fc、
  aug4、AdamW+LinearLR(1→0)+CE、batch16、epoch20。每 seed 多 lr 各 run 3 次→該 seed best-mean lr；
  最終 over 5 seeds(10,24,38,42,57) 的 **mean±std (ddof=1)**，與 4.4 聚合慣例一致。
- **預設只跑 ρ=10/20/30%**（lr 網格沿用 4.3：ρ≤10→`3e-5 5e-5 1e-4 3e-4`、ρ20/30→`5e-5 1e-4 5e-4`）。
- **結果隔離**：`classification/exp_results/classification_hard/redistribution_simclr/{strategy}_seed{seed}_bs16.json`
  （與 `AL_simclr/` 同結構，plot 用同一套 per-seed-best 聚合）。重跑安全（滿 3 runs 自動 skip）。
- **跑**：`DEVICE=cuda:6 ./thesis/chapter_5/run_5_3_redistribution.sh`
  （可 `STRATEGIES=/PORTIONS=/SEEDS=` 覆寫；分卡＝多 shell 各設不同 DEVICE 與 SEEDS 子集）。
- **畫**：`python3 thesis/chapter_5/plot_5_3_redistribution.py`（`--no_random` 不畫 Random 參考線）。
  三張 `figs/5_3_redistribution_{margin,coreset,cluster_margin}.png`，title=`Ablation on <Method>`，
  每張＝原始 AL（實線）vs **Class Redistribution Only**（同色虛線、空心 marker、含 error bar）＋ Random（灰虛線參考）。
- **解讀**：redistribution-only 若 ≈ AL → 類別重分布即足以解釋；若 ≈ Random → 逐影像挑選才是關鍵。

### 5.3.3　代表性影像討論（最具/最不具資訊量之影像）— 設計（待實作）
**核心問題**：AL「最先選」與「到高 portion 都不選」的影像各長什麼樣？只訓練在這兩極端上，效能差多少？
反向理解「`D_train` 中哪些影像對模型最有/最沒幫助(甚至有害)」。

**三臂 fine-tune 比較（皆 ρ=10% = 203 張、seed 42、同 one-shot 協定：θ²/aug4/sweep-lr→best）**
| 臂 | 標註集 = | 來源 |
|---|---|---|
| **Random 10%** | 該 seed 隨機子集 | cold-start 既有（可重建 index）|
| **AL top-10%** | Margin/Core-set/Cluster-Margin 在 ρ=10% 的 **cumulative**（最先選的） | `AL_simclr/labeled_ids` 既有 |
| **Bottom-10%** | 該方法 AL **跑到 ρ=90% 後仍未被選**的 leftover（全 train 池 \ cumulative@90%） | **需延伸跑 AL 到 90%**（見下） |
- 預期：**AL top > Random > Bottom**；若 Bottom 甚至低於 Random，才足以稱這些影像「相對有害」。
- ⚠️ **類別分布 confound（必做）**：Bottom-10% 極可能高度 Healthy-skewed → 它差可能來自「類別極偏」而非「影像沒用」。
  **務必同時印出三臂的類別分布**，解讀時分清楚（呼應 5.3.1 重分布故事）。
- per-method：Bottom 與 AL-top 各方法一組；Random 共用。

**視覺化（每類別取 10 張縮圖呈現三臂）**
- Random / AL-top / Bottom 各自每個類別抽 10 張展示。
- **AL-top 視覺化要扣掉初始隨機 2.5%**（只顯示 AL 真正挑的 `selected`，非含 random 起點的 cumulative）；
  但 **fine-tune 仍用完整 cumulative@10%**（才對得上「標準 AL@10%」語意）。
- 另附 Bottom-10% 的類別分布長條 + 歸納其影像特徵（多為典型多數類、清晰無病灶歧義…，看圖再寫）。

**延伸 AL 到 90%（取得 Bottom-10% leftover；只跑 seed 42、標準 sweep-lr 協定）**
- AL 軌跡本是 stateful（從 `portion_start` 用 `random.seed` 重建）。`run_AL.py` 新增 **`--resume_labeled_ids` + `--resume_from`**：
  直接載入 4.4 既有軌跡的 **cumulative@60%** 當起始已標註集，**只重訓一個 selector（anchor=60%，不選新樣本，因 cum@60=1219=target）**、
  再跑 65→90，**免重算前段 2.5→57.5**（三方法 cum@60 都剛好 1219）。腳本預設已開啟 resume@60。
- 寫到**隔離 exp_path**（`ch5_5_3_3_extend/`），不污染 4.4 主線 `AL_simclr`。
- `run_AL.py` 新增 `--parallel_lr`：把同一 portion 的多個候選 lr **在同一張 GPU 上同時訓練**（ThreadPoolExecutor，
  每 lr 各自 DataLoader），縮短 wall-clock；選取邏輯/結果與序列版一致，預設關閉（4.4 不受影響）。
- 一個方法一個腳本（挑空 GPU、可同時開三張卡跑）：
```bash
DEVICE=cuda:3 ./thesis/chapter_5/run_5_3_3_extend_margin.sh # 已下
DEVICE=cuda:2 ./thesis/chapter_5/run_5_3_3_extend_coreset.sh # 已下
DEVICE=cuda:0 ./thesis/chapter_5/run_5_3_3_extend_cluster_margin.sh # 已下
DEVICE=cuda:2 ./thesis/chapter_5/run_5_3_3_extend_typiclust.sh # 已下
```
- 產出 `ch5_5_3_3_extend/classification_hard/AL_simclr/labeled_ids/{method}_seed42_bs16.json`；
  其 `90.0` 的 `cumulative` 之**補集**即該方法的 Bottom-10%（≈203 張）。
- **省時：分段 b（腳本預設）**。`run_AL.py` 新增 `--portions`（顯式 portion 清單，覆寫等距 arange）：
  前段 **b₀=2.5、b=2.5 到 60**（忠於主線軌跡）、後段 **b=5 到 90**，落在 90% → leftover 剛好 10%。
  portion 數 36→**30**，且砍掉的 6 個都是後段大資料量步（62.5/67.5/…/87.5）。可用 env `PORTIONS=` 覆寫。
  ⚠️ caveat：ρ>60% 改 b=5 表示 AL 在後段「重評估未標註池」的頻率變低、每批選更多 → 後段選樣略粗,
  但對「最後 10% 從未被選」的判定影響很小（這些是全程穩定高 margin/最冗餘者），可接受。
- ⚠️ **成本**：即便分段，到 90% 後段資料量仍大（90%≈1829 張×aug4）→ 偏久；`--parallel_lr`
  讓同 portion 的 lr 並行，但單卡 compute 仍部分序列化，實際加速為「部分」而非滿 ×N（記憶體無虞）。
- 〔暫不做〕想法一（某 model 對未標註池的 uncertainty/margin score 分布 + top/median/bottom 影像）先擱置。

**待寫**：`run_5_3_3_representative.py`（三臂 custom-idx 訓練，沿用 redistribution 那套）+ `plot_5_3_3_*.py`（縮圖格 + 類別分布 + 三臂 acc 長條）。

### 質性：UMAP / t-SNE 視覺化 — `plot_5_3_umap.py` + `plot_5_3_umap_grid.py`（✅ 已實作）
- 用某模型的 **512 維 last-layer 特徵**（global-avgpool 後、fc 前）把 train（或 test）set 投到 2D。
  類別=顏色+marker；AL 選取的點用**星號**(`--highlight star`)或**黑色空心方框**(`--highlight box`，無 `_star` 後綴) 標出。
  特徵 + 2D embedding 存 `umap_cache/{model}_{method}[_test].npz`，跨 strategy/portion 重用。
- **降維**：`--method umap | tsne`（t-SNE 先 PCA→50 再降）。圖依 `figs/{method}/{model}/` 分子資料夾。
- **feature extractor（重要取捨）**：模型放 `thesis/gradcam/ckpt/`，`--ckpt` 指定。
  - **看 AL 選樣幾何 → 用「未經分類 finetune」的表示空間**（frozen SimCLR θ² backbone =
    `SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl`）：類別重疊、有邊界，看得出 diversity 覆蓋 / uncertainty 邊界。
  - finetuned 分類器（如 `simclr_p100_4x.pth`）會把類別壓成**孤立團塊**（trivially separable），且是全資料訓練的、AL 從沒用過 → 當底圖不貼切；只適合看「團塊內選樣率」。
  - ⚠️ 誠實性 caveat：AL 選樣**當下**用的是「該 portion 當下 finetuned 模型」的 backbone（diversity）或 softmax（uncertainty），既非 frozen 也非 p100。frozen backbone 只是**固定中性的共同底圖**（同 TypiClust 等論文做法），論文需一句話交代。
- 樣式（皆為論文字級）：星號/背景點等大、legend 依**類別張數多→少**排序、title 只有策略+ρ（無 seed）、box 黑框。
```bash
# 單圖（預設 margin/coreset/cluster_margin × {30,15}%；--strategy 可加 typiclust 等）
python3 thesis/chapter_5/plot_5_3_umap.py --ckpt SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl \
    --method umap --highlight star --portion 30 15 10 --device cuda:6
# 純特徵空間（無任何選取標註）
python3 thesis/chapter_5/plot_5_3_umap.py --ckpt <ckpt> --base
```

#### 六宮格：特徵空間隨 finetune portion 演變 — `plot_5_3_umap_grid.py`
- 2×3 拼圖，子圖 (a)–(f) = ρ∈{0,15,30,50,70,100}% 的 finetune 模型對 train/test set 的特徵空間。
  - ρ=0% = 完全沒 finetune = **frozen SimCLR θ² backbone**；ρ=100% = `simclr_p100_4x.pth`；
    ρ=15/30/50/70% = `simclr_p{ρ}_4x.pth`（缺的格子自動標 pending，補上 checkpoint 重跑即填）。
  - 每個子圖**各自畫 UMAP-1/UMAP-2**（六張是獨立 UMAP、兩維物理意義不同，**不可共用軸標**）。
```bash
python3 thesis/chapter_5/plot_5_3_umap_grid.py --method umap --split train --device cuda:6
python3 thesis/chapter_5/plot_5_3_umap_grid.py --method umap --split test  --device cuda:6
```
- **產生中間四格 checkpoint（finetune，存到 `thesis/gradcam/ckpt/`）**：
```bash
for p in 15 30 50 70; do python3 thesis/chapter_5/finetune_full_model.py --portion $p --device cuda:6; done
```
  `finetune_full_model.py` 的 lr **自動查 `cold_start_simclr/random{seed}_bs16.json` 該 (portion,seed) 的 best-lr**
  （per-seed，與論文主線一致；查不到才退回 5e-4，或 `--lr` 手動覆寫）。模型 = θ² backbone、aug4、seed42、20 epoch；子集為 random ρ%。
  實測 best-lr：ρ15→5e-5、ρ30→5e-5、ρ50→1e-4、ρ70→1e-4。

#### 【TODO，等 model 都訓練好後再做】
- **用 random ρ=15% 的 finetune 模型（`simclr_p15_4x.pth`）當 feature extractor，重畫各 AL 選取影像的黑框圖**
  （`plot_5_3_umap.py --ckpt thesis/gradcam/ckpt/simclr_p15_4x.pth --highlight box ...`）。
  理由：p100 模型類別已 trivially 分離、frozen 又完全沒監督訊號；ρ=15% 是「低預算但已有監督結構」的折衷底圖，較貼近 AL 實際運作的表示空間。

### 【之後做】combine TypiClust + 其他 AL（hybrid scheduling）
- 想法：**低 ρ 用 TypiClust（typical/density）選樣，高 ρ 切換到 uncertainty / BADGE 等**，
  取兩者在各自預算區間的優勢。
- **這是 TypiClust 原 paper 留下的洞**：它只觀察到「低預算用 typical、高預算用 uncertain」的相變，
  並把「低預算範圍到哪、何時切換」明白列為 future work（"...we leave for future work."），
  **沒有實作也沒提出 switching 機制**。→ 我們實作並定出切換點 = 直接回應其 future work（novelty 之一）。
- 與 §5.2（用 TypiClust 改良 b₀）相關但不同：§5.2 只換「初始批 b₀」；此處是**整條軌跡的策略排程**。
- ⚠️ caveat：本 codebase 的 TypiClust 是 **warm-start、用當前 finetuned 模型的 backbone 特徵**，
  已偏離原版「frozen SSL 特徵 + 零標註自選首批」；做 hybrid 前需決定是否補一版忠實的 frozen-SSL TypiClust。
- **狀態：留到之後做（先完成上面的 confusion matrix / 類別分布 / UMAP 量化分析）。**

---

## 5.x　Augmentation 的 10-fold 交叉驗證（robustness check）

驗證 §4.2 的 aug 結論（w/o · HF · VF · HF+VF · HF+VF+HVF）不是單一 train/val/test 切分的 artifact：
**10-fold CV**、one-shot（**無 AL**）、init = **ImageNet（與 §4.2 一致）**，在 ρ=5/50/100% 比五種 aug。
**fold 1 = §4.2 既有資料（直接匯入、不重跑）**；本腳本跑 **folds 2..5**，之後可補 6..10。

**CV 切法（全部在程式層面、不動 `ds/` 資料夾）**：把 `train/`(2032)+`val/`(509) 共 2541 張 pool 起來
（8:1:1 → **10 個 chunk**）。chunk 定義刻意讓 **fold 1 = 原本的 split**：
`chunk 0..7` = `train/` 依類別 stratify 切 8 份、`chunk 8/9` = 原本的 val/test（對齊 `utils/data.py`）。
fold f（1-indexed）把 10-chunk 視窗循環右移 (f−1) 格：
`train={c_s..c_{s+7}}, val=c_{s+8}, test=c_{s+9}`（s=f−1, mod 10）。
→ fold 1 (s=0) = train/ 全部 + 原 val + 原 test ＝**原本 8:1:1 split**（已驗證：彙整數字與 §4.2 表完全一致）。
labeled 子集在該 fold 的 train pool 內用 seed 抽樣（各 aug 同 seed→同子集，公平比較）。

- **portion 5/50**：5 seeds(10,24,38,42,57)；**portion 100**：只 seed 42（全集、seed 無關），仍跑各 fold。
- **lr 網格（§4.2 per-portion）**：ρ5=`7e-5 1e-4 3e-4 5e-4 7e-4`、ρ50=`5e-5 7e-5 1e-4 3e-4 5e-4`、
  ρ100=`1e-5 5e-5 1e-4 5e-4`；每 (aug,fold,seed,lr) 跑 3 runs。
- **平行**：同一 (portion,fold,aug,seed) 的「所有 LR × 3 runs」一次丟**同一張 GPU** 平行（ρ5/50=15、ρ100=12 個 process）。
  `run_first_iter_cv.py` 用 `fcntl` 檔案鎖保護 JSON append → 多 process 寫同一 fold/seed 檔不會 race。
- **彙整（每 fold 一張表）**：`aggregate_aug_cv.py` 對**每個 fold 各印一張表**（rows=portion、cols=五種 aug），
  每格 = 該 fold 的 **mean±std over seeds**（per-seed best-lr）；**ρ100 單 seed → std 取 best-lr 的 runs**（§4.2 SPECIAL_RHO_100，ddof=1）。
  最後另附一張「ALL folds pooled」總覽。fold 1 之表 = §4.2 原表（已驗證 means 完全一致）。

```bash
# 0) 先把 §4.2(ImageNet) 匯入成 fold 1（一次就好）
python3 thesis/chapter_5/import_fold1_from_4_2.py

# 1) 跑 folds 2..5（從 repo 根；可覆寫 DEVICE/FOLDS/PORTIONS/SEEDS/LRS/RUNS/AUGS）
#    分卡只用 FOLDS/SEEDS（結果檔名=fold{F}_seed{S}）；勿用 PORTIONS/AUGS 分→撞同檔。
DEVICE=cuda:2 FOLDS=2 ./thesis/chapter_5/run_5_x_aug_cv.sh &
DEVICE=cuda:3 FOLDS=3 ./thesis/chapter_5/run_5_x_aug_cv.sh &
DEVICE=cuda:4 FOLDS=4 ./thesis/chapter_5/run_5_x_aug_cv.sh &
DEVICE=cuda:6 FOLDS=5 ./thesis/chapter_5/run_5_x_aug_cv.sh &
# 之後補後 5 folds：FOLDS="6 7 8 9 10"

# 2) 看表（主表 + 每 fold 細節）
python3 thesis/chapter_5/aggregate_aug_cv.py
python3 thesis/chapter_5/aggregate_aug_cv.py --detail
```

- 程式：[`classification/utils/data_cv.py`](../../classification/utils/data_cv.py)（10-fold 切分，fold 1=原本 split）、
  [`classification/run_first_iter_cv.py`](../../classification/run_first_iter_cv.py)（單次訓練 entry，ImageNet init）、
  [`import_fold1_from_4_2.py`](import_fold1_from_4_2.py)（匯入 fold 1）、
  [`aggregate_aug_cv.py`](aggregate_aug_cv.py)（印表）。
- 結果隔離樹：`classification/exp_results/chapter5_aug_cv/classification_hard/aug_cv_imagenet/fold{F}_seed{S}_bs16_ep20.json`。
- 重跑安全：每 (aug,portion,lr) 滿 3 runs 會被 `check_existing_results` 擋（腳本 `|| true` 跳過）。

---

## 5.4　延伸至影像分割任務（之後做）

- 嘗試把分類的 AL 觀察遷移到 2D 影像分割（`segmentation/` 既有 U-Net pipeline）。
- 細節待 5.1–5.3 收斂後再規劃。

---

## 待確認 / 開放問題
1. **b₀ 大小是否顯著影響？**（user 對此較不確定）— 見下方「討論」。先做 5.1.1 的 ablation 來定論。
2. ~~5.2 第三法選 "Making Your First Choice" 還是 USL？~~ → 兩者**暫緩**（MYFC 需動 SSL 預訓練、太麻煩）。
   5.2 近期用 TypiClust（已實作），必要時加 ProbCover。
3. Ch5 鎖定的三方法，待 Ch4 5-seed 最終 al_curve 定版後確認（目前 Margin / Coreset / Cluster-Margin）。
4. confusion matrix 與其餘 5.3 分析，是否都統一在 ρ=30%、5 seeds 平均。


## Class Redistribution Ablation Experiments (5.3)

```bash
# 訓練（預設 margin/coreset/cluster_margin × ρ=10/20/30 × 5 seeds）
DEVICE=cuda:6 ./thesis/chapter_5/run_5_3_redistribution.sh

# 想分卡加速：開多個 shell，各設不同 DEVICE + SEEDS 子集，例如
SEEDS="10" DEVICE=cuda:0 ./thesis/chapter_5/run_5_3_redistribution.sh # 已下
SEEDS="24" DEVICE=cuda:1 ./thesis/chapter_5/run_5_3_redistribution.sh # 已下
SEEDS="38" DEVICE=cuda:2 ./thesis/chapter_5/run_5_3_redistribution.sh # 已下
SEEDS="42" DEVICE=cuda:7 ./thesis/chapter_5/run_5_3_redistribution.sh # 已下
SEEDS="57" DEVICE=cuda:9 ./thesis/chapter_5/run_5_3_redistribution.sh # 已下


# portion = {15,25,35,45,55}
SEEDS="10 24" DEVICE=cuda:1 ./thesis/chapter_5/run_5_3_redistribution_odd.sh # 已下
SEEDS="38 42" DEVICE=cuda:2 ./thesis/chapter_5/run_5_3_redistribution_odd.sh # 已下
SEEDS="57" DEVICE=cuda:7 ./thesis/chapter_5/run_5_3_redistribution_odd.sh # 已下

# portion = {40,50,60}
SEEDS="10" DEVICE=cuda:0 ./thesis/chapter_5/run_5_3_redistribution_high.sh # 已下
SEEDS="24" DEVICE=cuda:0 ./thesis/chapter_5/run_5_3_redistribution_high.sh # 已下
SEEDS="38" DEVICE=cuda:4 ./thesis/chapter_5/run_5_3_redistribution_high.sh # 已下
SEEDS="42" DEVICE=cuda:5 ./thesis/chapter_5/run_5_3_redistribution_high.sh # 已下
SEEDS="57" DEVICE=cuda:6 ./thesis/chapter_5/run_5_3_redistribution_high.sh # 已下

# 跑完畫圖
python3 thesis/chapter_5/plot_5_3_redistribution.py
# 或加 --no_random
```


## Bottom-10% Training

```bash
for lr in 3e-5 5e-5 1e-4 3e-4 5e-4; do
    python3 thesis/chapter_5/run_5_3_3_bottom10.py \
        --strategy margin --seed 42 --lr $lr --device cuda:7
done # 已下

for lr in 3e-5 5e-5 1e-4 3e-4 5e-4; do
    python3 thesis/chapter_5/run_5_3_3_bottom10.py \
        --strategy cluster_margin --seed 42 --lr $lr --device cuda:9
done # 已下

for lr in 3e-5 5e-5 1e-4 3e-4 5e-4; do
    python3 thesis/chapter_5/run_5_3_3_bottom10.py \
        --strategy coreset --seed 42 --lr $lr --device cuda:8
done # 已下

```


## Foundation models to guide initial labeled pool

```bash
PARALLEL=3 MODELS="resnet_imagenet:resnet18 resnet_imagenet:resnet50" PORTIONS="2.5 10 20" DEVICE=cuda:4 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下

PARALLEL=3 MODELS="resnet_imagenet:resnet101 dinov2:small"            PORTIONS="2.5 10 20" DEVICE=cuda:5 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下

PARALLEL=3 MODELS="retfound:oct medimageinsight:base"                PORTIONS="2.5 10 20" DEVICE=cuda:6 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下

PARALLEL=3 MODELS="simclr:resnet18" PORTIONS="2.5 10 20" DEVICE=cuda:9 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下

PARALLEL=3 MODELS="dinov2:base dinov2:large"                          PORTIONS="2.5 10 20" DEVICE=cuda:8 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下

PARALLEL=3 MODELS="clip:base clip:large"                              PORTIONS="2.5 10 20" DEVICE=cuda:7 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下

PARALLEL=3 MODELS="resnet_imagenet:resnet152" PORTIONS="2.5 10 20" DEVICE=cuda:1 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下

PARALLEL=3 MODELS="radimagenet:resnet50 biomedclip:base"             PORTIONS="2.5 10 20" DEVICE=cuda:3 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下

PARALLEL=3 MODELS="simclr:resnet50" PORTIONS="2.5 10 20" DEVICE=cuda:9 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下

PARALLEL=3 MODELS="simclr:resnet50" PORTIONS="2.5 10 20" DEVICE=cuda:9 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh

PARALLEL=3 MODELS="retfound:cfp" PORTIONS="2.5 10 20" DEVICE=cuda:4 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下

PARALLEL=3 MODELS="resnet_imagenet:resnet34" PORTIONS="2.5 10 20" DEVICE=cuda:5 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下

PARALLEL=3 MODELS="simclr:resnet152" PORTIONS="2.5 10 20" DEVICE=cuda:8 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下

PARALLEL=3 MODELS="simclr:resnet101" PORTIONS="2.5 10 20" DEVICE=cuda:9 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下 (這是有點錯掉的，太早的epoch了)

PARALLEL=3 MODELS="simclr:resnet50_best" PORTIONS="2.5 10 20" DEVICE=cuda:6 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下


## 最新補的，theta_simclr^1
PARALLEL=3 MODELS="simclr:resnet18_random" PORTIONS="2.5 10 20" \
    DEVICE=cuda:9 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh

# plot results
python3 thesis/chapter_5/coldstart_fm/plot_coldstart_fm.py          # all 3 portions at once
python3 thesis/chapter_5/coldstart_fm/plot_coldstart_fm.py --portions 10   # just one

### 快速補LR
python3 thesis/chapter_5/coldstart_fm/run_coldstart_fm.py \
    --model resnet_imagenet:resnet152 --portion 20 \
    --lr_grid 1e-5 3e-5 --parallel_runs 3 --device cuda:8 # 已下

python3 thesis/chapter_5/coldstart_fm/run_coldstart_fm.py \
    --model simclr:resnet152_best --portion 2.5 \
    --lr_grid 1e-5 --parallel_runs 3 --device cuda:8 # 已下

## Best val simclr pretraining ckpt

PARALLEL=3 MODELS="simclr:resnet152_best" PORTIONS="2.5 10 20" DEVICE=cuda:7 ./thesis/chapter_5/coldstart_fm/run_coldstart_fm.sh # 已下

```

Plot

```bash
python3 thesis/chapter_5/coldstart_fm/plot_coldstart_fm.py
```


## SimCLR Pretraining on larger resnet

```bash
python3 SSL/simclr/run.py -data ./ds/classification/seven_class/train -a resnet50 --epochs 500 -b 256 --lr 0.0002 --device cuda:9 --micro_bs 128 # 已下 
# --grad_cache off

python3 SSL/simclr/run.py -data ./ds/classification/seven_class/train -a resnet152 --epochs 500 -b 256 --lr 0.0002  --device cuda:1 # micro_bs 盡量大，才能避免來自BN的Numerical error (不太影響) # 已下


python3 SSL/simclr/run.py -data ./ds/classification/seven_class/train -a resnet101 --epochs 500 -b 256 --lr 0.0002 --device cuda:3 --micro_bs 128 # 已下

python3 SSL/simclr/run.py -data ./ds/classification/seven_class/train -a resnet34 --epochs 500 -b 256 --lr 0.0002 --device cuda:1 --grad_cache off # 已下


###### 試試看是否有Overfitting? --> 有!

python3 SSL/simclr/run.py -data ./ds/classification/seven_class/train -a resnet50 --epochs 500 -b 256 --lr 0.0002 --micro_bs 128 --device cuda:6 --val-open # 已下


python3 SSL/simclr/run.py -data ./ds/classification/seven_class/train -a resnet152 --epochs 500 -b 256 --lr 0.0002 --micro_bs 128 --device cuda:2 --val-open
# 已下
```


開始從initial warm start labeled pool做AL

```bash
python3 classification/run_AL.py \
    --task_type hard \
    --AL_strategy margin \
    --pretrained_weights simclr \
    --simclr_path SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl \
    --portion_start 2.5 --portion_end 62.5 --portion_interval 2.5 \
    --resume_labeled_ids thesis/chapter_5/coldstart_fm/labeled_ids/simclr__resnet18.json \
    --resume_from 2.5 \
    --exp_path classification/exp_results/ch5_fm_init_al \
    --seed 42 --device cuda:9
  # 已下

python3 classification/run_AL.py \
    --task_type hard \
    --AL_strategy margin \
    --pretrained_weights simclr \
    --simclr_path SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl \
    --portion_start 2.5 --portion_end 62.5 --portion_interval 2.5 \
    --resume_labeled_ids thesis/chapter_5/coldstart_fm/labeled_ids/simclr__resnet18.json \
    --resume_from 2.5 \
    --exp_path classification/exp_results/ch5_fm_init_al \
    --seed 10 --device cuda:3
    # 已下

python3 classification/run_AL.py \
    --task_type hard \
    --AL_strategy margin \
    --pretrained_weights simclr \
    --simclr_path SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl \
    --portion_start 2.5 --portion_end 62.5 --portion_interval 2.5 \
    --resume_labeled_ids thesis/chapter_5/coldstart_fm/labeled_ids/simclr__resnet18.json \
    --resume_from 2.5 \
    --exp_path classification/exp_results/ch5_fm_init_al \
    --seed 24 --device cuda:4
  # 已下

python3 classification/run_AL.py \
    --task_type hard \
    --AL_strategy coreset \
    --pretrained_weights simclr \
    --simclr_path SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl \
    --portion_start 2.5 --portion_end 62.5 --portion_interval 2.5 \
    --resume_labeled_ids thesis/chapter_5/coldstart_fm/labeled_ids/simclr__resnet18.json \
    --resume_from 2.5 \
    --exp_path classification/exp_results/ch5_fm_init_al \
    --seed 42 --device cuda:8
  # 已下

python3 classification/run_AL.py \
    --task_type hard \
    --AL_strategy coreset \
    --pretrained_weights simclr \
    --simclr_path SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl \
    --portion_start 2.5 --portion_end 62.5 --portion_interval 2.5 \
    --resume_labeled_ids thesis/chapter_5/coldstart_fm/labeled_ids/simclr__resnet18.json \
    --resume_from 2.5 \
    --exp_path classification/exp_results/ch5_fm_init_al \
    --seed 10 --device cuda:2
  # 已下

python3 classification/run_AL.py \
    --task_type hard \
    --AL_strategy coreset \
    --pretrained_weights simclr \
    --simclr_path SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl \
    --portion_start 2.5 --portion_end 62.5 --portion_interval 2.5 \
    --resume_labeled_ids thesis/chapter_5/coldstart_fm/labeled_ids/simclr__resnet18.json \
    --resume_from 2.5 \
    --exp_path classification/exp_results/ch5_fm_init_al \
    --seed 24 --device cuda:0
 # 已下

python3 classification/run_AL.py \
    --task_type hard \
    --AL_strategy cluster_margin \
    --pretrained_weights simclr \
    --simclr_path SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl \
    --portion_start 2.5 --portion_end 62.5 --portion_interval 2.5 \
    --resume_labeled_ids thesis/chapter_5/coldstart_fm/labeled_ids/simclr__resnet18.json \
    --resume_from 2.5 \
    --exp_path classification/exp_results/ch5_fm_init_al \
    --seed 42 --device cuda:9
  # 已下

python3 classification/run_AL.py \
    --task_type hard \
    --AL_strategy cluster_margin \
    --pretrained_weights simclr \
    --simclr_path SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl \
    --portion_start 2.5 --portion_end 62.5 --portion_interval 2.5 \
    --resume_labeled_ids thesis/chapter_5/coldstart_fm/labeled_ids/simclr__resnet18.json \
    --resume_from 2.5 \
    --exp_path classification/exp_results/ch5_fm_init_al \
    --seed 10 --device cuda:9
    # 已下

python3 classification/run_AL.py \
    --task_type hard \
    --AL_strategy cluster_margin \
    --pretrained_weights simclr \
    --simclr_path SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl \
    --portion_start 2.5 --portion_end 62.5 --portion_interval 2.5 \
    --resume_labeled_ids thesis/chapter_5/coldstart_fm/labeled_ids/simclr__resnet18.json \
    --resume_from 2.5 \
    --exp_path classification/exp_results/ch5_fm_init_al \
    --seed 24 --device cuda:8
  # 已下
```

Plot
```bash
python3 thesis/chapter_5/plot_5_x_warmstart_al.py
```

## 獲取AL ckpt 以利GradCam分析

```bash
python3 thesis/gradcam/dump_al_p30_ckpt.py --device cuda:4 # 已下
```


## 做在semantic segmentation, DICE

多數由claude code直接幫我下指令


AE預訓練:
```bash
python3 thesis/chapter_5/segmentation/autoencoder/train_ae.py --epochs 2000 --device cuda:7

# Plot
python3 thesis/chapter_5/segmentation/autoencoder/plot_ae_progress.py
```

剩餘Data Aug:

```bash
bash thesis/chapter_5/segmentation/scripts/run_vf_30_100.sh   cuda:8 3   
# VF,  一次 3 個併發
# 已下

bash thesis/chapter_5/segmentation/scripts/run_vfhv_30_100.sh cuda:6 3   
# VF+HV, 一次 3 個併發
# 已下
```


Plot All Figures
```bash
python3 thesis/chapter_5/segmentation/report_table.py
```


## 6/24 重新做在semantic segmentation, DICE --> 之前全跑錯了!!

### 1. Aug sweep(w/o / HF / 4x,各 portion lr 掃描)
每個丟一張卡(第 3 參數 PAR = 同卡併發數,24GB 卡建議 2–3、49GB 可 5):

```bash
bash thesis/chapter_5/segmentation/scripts/run_aug_sweep.sh 1 cuda:9 3 "2.5 5 10 100 20 40 60 80 30 70 90" # 已重下
bash thesis/chapter_5/segmentation/scripts/run_aug_sweep.sh 2 cuda:6 3 "2.5 5 10 100 20 40 60 80 30 70 90" # 已重下
bash thesis/chapter_5/segmentation/scripts/run_aug_sweep.sh 4 cuda:7 3 "2.5 5 10 100 20 40 60 80 30 70 90" # 已重下
bash thesis/chapter_5/segmentation/scripts/run_aug_sweep.sh vf   cuda:8 3 "2.5 5 10 100 20 40 60 80 30 70 90"   # VF only # 已重下
bash thesis/chapter_5/segmentation/scripts/run_aug_sweep.sh vfhv cuda:0 5 "2.5 5 10 100 20 40 60 80 30 70 90"   # VF+HV  # 已重下

for AUG in 1 2 4 vf vfhv; do
  bash thesis/chapter_5/segmentation/scripts/run_aug_sweep.sh $AUG cuda:9 3 "50"
done
```
每個 aug ≈ 183 jobs:ρ{2.5…90} × 5 seeds × per-portion 3 lrs + ρ=100 單 seed×6 runs。
ρ<100 = 1 run/格(rerun-safe);結果寫 exp_results/aug_curve/aug{1,2,4}/。
HF aug 現在真的有意義了(資料不再預翻)。

### 2. Autoencoder 訓練(train split,你決定 epoch 數)
```bash
python3 thesis/chapter_5/segmentation/autoencoder/train_ae.py --epochs 2000 --device cuda:1
```
--epochs 你自己定;snapshot/ckpt 時間點照舊(1,3,10,30,50,100,200,500,1000,1500,2000)。
dataroot 已預設 ds/segmentation_correct,train 在 490 張上跑、每 epoch 印 loss。
舊的 AE / SimCLR ckpt + training loss 全在 segmentation_old_wrong/(rename 後已自動隔離 = 等同 _old_wrong),新的寫在新 segmentation/autoencoder/ 下。

### 3. 畫圖(都照常,只是在新 segmentation 底下)

```bash
python3 thesis/chapter_5/segmentation/report_table.py                      # aug 曲線 + 表
python3 thesis/chapter_5/segmentation/autoencoder/plot_ae_progress.py      # AE loss/重建圖
python3 thesis/chapter_5/segmentation/plot_dataset_samples.py              # 資料樣本(已指新資料)
```


AE weight init
```bash
bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
     --ae-epoch 2000 --portions 5 --seeds 10 24 38 42 57 --aug 4 --device cuda:6 --max-runs 3


bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
     --ae-epoch 2000 --portions 2.5 5 10 20 30 40 50 60 70 80 90 100 --seeds 10 24 --aug 4 --device cuda:0 --max-runs 5

bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
     --ae-epoch 2000 --portions 2.5 5 10 20 30 40 50 60 70 80 90 100 --seeds 38 42 --aug 4 --device cuda:1 --max-runs 5

bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
     --ae-epoch 2000 --portions 2.5 5 10 20 30 40 50 60 70 80 90 100 --seeds 57 --aug 4 --device cuda:8 --max-runs 3


bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
     --ae-epoch 2000 --portions 2.5 5 10 20 30 40 50 60 70 80 90 100 --seeds 10 24 --aug 1 --device cuda:2 --max-runs 5

bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
     --ae-epoch 2000 --portions 2.5 5 10 20 30 40 50 60 70 80 90 100 --seeds 38 42 --aug 1 --device cuda:3 --max-runs 5

bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
     --ae-epoch 2000 --portions 2.5 5 10 20 30 40 50 60 70 80 90 100 --seeds 57 --aug 1 --device cuda:9 --max-runs 3

bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
     --ae-epoch 2000 --portions 10 --seeds 10 24 38 42 57 --aug 1 --device cuda:9 --max-runs 3

bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
     --ae-epoch 1000 --portions 2.5 5 --seeds 10 24 38 42 57 --aug 1 --device cuda:7 --max-runs 3

bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
     --ae-epoch 2000 --portions 20 --seeds 10 24 --aug 4 \
     --lrs 0.007 0.01 --device cuda:9 --max-runs 3

## 以上皆已下!
```

Warmup
```bash
bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
  --ae-epoch 2000 --portions 2.5 --aug 4 --device cuda:1 --max-runs 4 --warmup 5

# 已下

bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
  --ae-epoch 2000 --portions 2.5 --aug 4 --device cuda:2 --max-runs 4 --warmup 10

# 已下

bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
  --ae-epoch 2000 --portions 2.5 --aug 4 --device cuda:1 --max-runs 4 --warmup 15

# 已下

bash thesis/chapter_5/segmentation/scripts/run_ae_finetune.sh \
  --ae-epoch 2000 --portions 10 --aug 4 --device cuda:2 --max-runs 4 --warmup 10

```

存 model ckpt

```bash
bash thesis/chapter_5/segmentation/scripts/run_3aug_ckpt_infer.sh cuda:0
```


最後! 主動學習!

```bash
DEVICE=cuda:0 STRATEGIES=margin SEEDS="10 24" bash thesis/chapter_5/segmentation/scripts/run_al.sh
DEVICE=cuda:2 STRATEGIES=margin SEEDS="38" bash thesis/chapter_5/segmentation/scripts/run_al.sh
DEVICE=cuda:0 STRATEGIES=margin SEEDS=57 bash thesis/chapter_5/segmentation/scripts/run_al.sh
DEVICE=cuda:2 STRATEGIES=margin SEEDS="42" bash thesis/chapter_5/segmentation/scripts/run_al.sh

DEVICE=cuda:3 STRATEGIES=coreset SEEDS="10" bash thesis/chapter_5/segmentation/scripts/run_al.sh
DEVICE=cuda:3 STRATEGIES=coreset SEEDS="24" bash thesis/chapter_5/segmentation/scripts/run_al.sh
DEVICE=cuda:3 STRATEGIES=coreset SEEDS="38" bash thesis/chapter_5/segmentation/scripts/run_al.sh
DEVICE=cuda:4 STRATEGIES=coreset SEEDS="42" bash thesis/chapter_5/segmentation/scripts/run_al.sh
DEVICE=cuda:8 STRATEGIES=coreset SEEDS=57 bash thesis/chapter_5/segmentation/scripts/run_al.sh


DEVICE=cuda:9 STRATEGIES=cluster_margin SEEDS=10 bash thesis/chapter_5/segmentation/scripts/run_al.sh
DEVICE=cuda:9 STRATEGIES=cluster_margin SEEDS=24 bash thesis/chapter_5/segmentation/scripts/run_al.sh
DEVICE=cuda:7 STRATEGIES=cluster_margin SEEDS=38 bash thesis/chapter_5/segmentation/scripts/run_al.sh
DEVICE=cuda:7 STRATEGIES=cluster_margin SEEDS=42 bash thesis/chapter_5/segmentation/scripts/run_al.sh
DEVICE=cuda:2 STRATEGIES=cluster_margin SEEDS=57 bash thesis/chapter_5/segmentation/scripts/run_al.sh

# 以上皆已下

DEVICE=cuda:1 STRATEGIES=margin SEEDS="24" bash thesis/chapter_5/segmentation/scripts/run_al.sh

# 以上補下了


# --- margin (w/o Aug) ---
DEVICE=cuda:5 STRATEGIES=margin         SEEDS=10 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh # 已下
DEVICE=cuda:8 STRATEGIES=margin         SEEDS=24 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh # 已下
DEVICE=cuda:8 STRATEGIES=margin         SEEDS=38 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh # 已下
DEVICE=cuda:0 STRATEGIES=margin         SEEDS=42 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh # 已下
DEVICE=cuda:0 STRATEGIES=margin         SEEDS=57 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh # 已下
# --- coreset (w/o Aug) ---
DEVICE=cuda:1 STRATEGIES=coreset  SEEDS=10 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh # 已下
DEVICE=cuda:1 STRATEGIES=coreset        SEEDS=24 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh # 已下
DEVICE=cuda:1 STRATEGIES=coreset        SEEDS=38 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh # 已下
DEVICE=cuda:4 STRATEGIES=coreset        SEEDS=42 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh # 已下
DEVICE=cuda:4 STRATEGIES=coreset        SEEDS=57 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh # 已下
# --- cluster_margin (w/o Aug) ---
DEVICE=cuda:1 STRATEGIES=cluster_margin SEEDS=10 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh # 已下
DEVICE=cuda:6 STRATEGIES=cluster_margin SEEDS=24 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh # 已下
DEVICE=cuda:6 STRATEGIES=cluster_margin SEEDS=38 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh # 已下
DEVICE=cuda:N STRATEGIES=cluster_margin SEEDS=42 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh
DEVICE=cuda:N STRATEGIES=cluster_margin SEEDS=57 AUG=1 EXP=./thesis/chapter_5/segmentation/exp_results/al_noaug bash thesis/chapter_5/segmentation/scripts/run_al.sh

```

畫 noaug
```bash
python3 thesis/chapter_5/segmentation/plot_al_noaug.py
```