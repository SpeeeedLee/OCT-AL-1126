# CLAUDE.md — Ch5 Semantic-Segmentation Active Learning (nuclei)

Tracker for the **Chapter-5 extension of active learning to 2D nuclei semantic
segmentation** on skin-OCT. Self-contained under `thesis/chapter_5/segmentation/`.
The upstream `segmentation/` package is **NOT modified** — its pure U-Net
(`model.py`), loss, and dice tool are **copied verbatim** here; everything new
(aug, lr-sweep AL, plots, runners) lives in this folder.

> Companion memory: `ch5-segmentation-al` (auto-recall). Thesis-wide conventions:
> `thesis/CLAUDE.md`. This file is the segmentation-specific source of truth.

---

## 1. Scope / locked decisions (user-confirmed)

- **Task**: nuclei segmentation ONLY (binary; `ds/segmentation/cell/` masks). No layer seg.
- **Model**: pure `Optim_U_Net` — **no deep supervision, no deep feature sharing**
  (both already disabled upstream). Bottleneck = `Conv6` (filter·32 = 1024-d).
- **Selection unit = whole image** (NO patches).
- **Single fold** (`--fold 0`), **NO cross-validation**.
- **Data**: 1224 images → `random.Random(42).shuffle` → 5 folds of 244.
  fold 0 ⇒ **train 736 / val 244 / test 244**.
- **Metric**: train → keep the **lowest-validation-loss epoch** checkpoint →
  evaluate on the held-out **test** split → **per-image mean Dice** (headline).
  NOT the last epoch. Per-epoch train/val loss+dice are also saved.
- **Augmentation**: offline flip, image+mask synced. `--aug_factor`
  1=none / 2=+HF / 3=+HF+VF / 4=+HF+VF+HFV. Default 4 for AL.
- **LR**: tuned per condition (thesis convention = **per-seed best-lr → mean±std**).
  AL uses `--lr_schedule sweep` (per-portion lr grid, lowest-val-loss model =
  selector, every lr's test_dice stored) — mirrors classification option-A.
  `lr_grid_for(portion)`: <15→[5e-4,1e-3,3e-3]; 15–50→[3e-4,1e-3,3e-3]; ≥50→[3e-4,5e-4,1e-3].
- **Seeds**: the 5 classification seeds **10, 24, 38, 42, 57** (user once typo'd 38 as "32").
  ρ=100 is seed-independent → single seed 42 × reps.

---

## 2. AL strategies (classification → segmentation)

Image-level adaptation (lit-standard: aggregate per-pixel scores; encoder features as
the image descriptor):

| Strategy | File | How it lifts to seg |
|---|---|---|
| `margin` | `AL_strategy/uncertainty.py` | per-pixel `1-|2p-1|`, **mean over all pixels** |
| `confidence` | `AL_strategy/uncertainty.py` | `1-max(p,1-p)`, mean over pixels |
| `entropy` | `AL_strategy/uncertainty.py` | binary entropy, mean over pixels |
| `coreset` | `AL_strategy/diversity.py` | k-Center-Greedy on encoder-GAP embeddings, conditioned on labeled set |
| `typiclust` | `AL_strategy/diversity.py` | TypiClust (Hacohen ICML'22): KMeans(\|L\|+budget) on the **same** encoder-GAP embeddings as coreset → order clusters (fewest-labeled, then biggest) → round-robin pick the most TYPICAL (1/avg-KNN-dist, K=20) unlabeled point. Low-budget/cold-start AL. |
| `cluster_margin` | `AL_strategy/hybrid.py` | margin pool → L2-norm embeddings → HAC(avg) → round-robin |
| `random` | in `run_AL.py` | passive baseline (= aug4 cold-start curve) |

⚠️ **Margin ≡ Confidence for BINARY masks** (least-conf is a monotone transform of
margin → identical selection). **Do NOT run both.** Uncertainty group = **Margin +
Entropy** only. (conf/entropy are implemented + wired, but only entropy gets run.)

---

## 3. How to run (repo root, `conda activate oct-env`)

```bash
# cold-start / aug-curve point (passive random subset)
python3 thesis/chapter_5/segmentation/run_first_iter.py --dataroot ./ds/segmentation \
    --portion 30 --seed 42 --aug_factor 4 --lr 1e-3 --epoch 25 --device cuda:0

# active learning (per-portion lr sweep)
python3 thesis/chapter_5/segmentation/run_AL.py --AL_strategy margin \
    --dataroot ./ds/segmentation --portion_start 2.5 --portion_end 60 \
    --portion_interval 2.5 --seed 42 --aug_factor 4 --lr_schedule sweep --device cuda:0

# GPU-pool runner: jobfile lines contain __DEV__; 1 job/device, co-resident OK
bash thesis/chapter_5/segmentation/scripts/run_pool.sh <jobfile> "cuda:0 cuda:1 ..."

# tables + regenerate all 4 figures (best-lr bolded per seed)
python3 thesis/chapter_5/segmentation/report_table.py
```

Result JSON: `exp_results/<tree>/nuclei/{cold_start_random,AL_random}/...json`,
keyed `{portion: {lr: [run_dict]}}`; AL run_dicts also store `selected_idx`
(this round) + `labeled_idx` (cumulative).

### Result trees (which is authoritative)
- `aug_curve/aug{1,2,4}/` — **Task-0 aug ablation** (none/HF/4×), 5-seed, lr-swept;
  the **aug4 line = the Random baseline (Task-2)**.
- `margin_sweep/`, `al_sweep/` — AL strategies, 5-seed, lr-swept (`*_sweep` = the
  proper protocol; plots glob `*_sweep`).
- `lr_v2/`, `base_v2/`, `aug_lr/` — clean (post-race-fix) single-seed sweeps for
  Tasks 3/4/5-baseline and the fair aug×lr check.
- ⚠️ **STALE/CORRUPT, do not use**: `lr_sweep/`, original `main_aug4` baseline (file-race
  victims). `main_aug4/AL_random/margin_seed42` = old fixed-lr single-seed margin
  (still used as plot fallback only).

### Plots (classification-style, y-axis = Dice)
- `plot_aug_curve.py` → `figs/aug_curve.png` (none/HF/4×; colors/markers MATCH
  `classification/exp/data_aug/plot_all.py`: 4×=blue·o `#1F77B4`, HF=brown·D `#8C564B`,
  none=gray·v `#7F7F7F`).
- `plot_al_groups.py` → `figs/{uncertainty,diversity,hybrid}.png` (uncertainty=Margin+
  Entropy / diversity=Core-set+TypiClust / hybrid=BADGE+Cluster-Margin; each + Random
  dashed). Random drawn to ρ=100; AL strategies to 60.
- Style: `set_xlim(0,103)` so ρ=2.5 isn't flush; xticks `[5,10,20…100]`.
- **figsize convention (user, 2026-06-23): all thesis curve figures = `(12, 8)`** for a
  consistent look across the writeup (aug_curve / aug_curve_focus / al_groups / aggregate).
  Only `plot_dataset_samples.py` (OCT image montage) keeps its own aspect.
- `plot_aug_curve_focus.py` → `figs/aug_curve_focus.png`: aug curve + a **magnifier inset**
  (ρ=2.5–20) emphasising 4x-vs-HF. SERIES tuples = `(aug,label,color,marker,ls,zoom,bound)`:
  `zoom`=draw in inset, `bound`=sets inset y-limits (only 4x/HF bound it; w/o-Aug is drawn
  but clipped, box stays fixed). Box+inset border solid black, connector lines dashed.
  Extend by adding a SERIES row (e.g. VF(2x) teal `#17BECF`, or VF+HV).
- **Fallback logic**: plots PREFER new `*_sweep`/`aug_curve` data, FALL BACK per-portion
  to old `main_aug4`/`base_v2` so curves stay complete while the 5-seed run fills.
  Old fallback points are n=1 (no error bars) and get replaced as the batch climbs.

---

## 4. Results so far (seed42 unless noted)

- **100% Nuclei Dice** (lr-tuned): none **0.723** / HF **0.783** / 4× **0.780** (HF≈4×≫none).
  no-aug 0.72 matches 智皓's paper 71.25. ✓
- **Aug ablation (FAIR, per-aug best-lr)**: 10% 4×(0.69)>HF(0.68)>none(0.65);
  100% HF≈4×≈0.78>none. → **4× = safe default** (best low-data, ties HF full-data).
  The earlier "2×>4× @100%" was a **fixed-lr artifact** (4× trains 2× more/epoch →
  wants a lower lr). Best lr **drops with more data+aug** (5%→1e-3, 100%→3e-4).
- **LR sensitivity**: 5% very sensitive (peak 1e-3, ≤5e-4 collapses); 50% nearly flat;
  sensitivity ↓ as data grows.
- **Task-4** (10% random, 5-seed per-seed-best-lr): **0.6896 ± 0.0025**.
- **Margin vs Random**: ≈ Random below ρ≈15% (cold-start), **wins from ~17.5%**,
  peak **+1.3pt at ρ≈27.5–30%**, gap shrinks toward full data. (single-seed prelim;
  5-seed swept version in progress.)
- **Core-set**: tracks slightly above Random at low ρ (prelim).

---

## 5. Hard-won lessons / gotchas (READ before launching batches)

- **FILE RACE (critical)**: many concurrent jobs appending to the *same*
  `random_<seed>.json` (portion/lr are keys but the file is shared) → read-modify-write
  clobbers, **silent data loss**. FIX in `run_first_iter.py`: save re-reads + appends
  under `fcntl.flock` (short critical section AFTER training). `run_AL.py` writes one
  file per (strategy,seed) = single writer, so it's safe **as long as you never run two
  processes for the same (strategy,seed)**. To run extra portions/lrs/reps concurrently,
  either rely on the lock or give each job an isolated `--exp_path`.
- **No duplicate AL trajectories**: a `run_pool` with more jobs than devices keeps the
  extra queued; if you ALSO launch one manually you get a duplicate later when a slot
  frees → two writers on one AL json (race). If you nudge a queued seed manually,
  **kill that pool's dispatcher** (in-flight children survive) so it never dispatches the
  duplicate.
- **`pgrep`/`kill` self-match**: `pgrep -af run_AL` matches THIS shell command too;
  `pgrep|grep|awk|kill` often passes multiple/garbage PIDs → `kill: arguments must be
  process IDs`. Use exact PIDs + `kill -0 <pid>` to test liveness, or read
  `/proc/<pid>/cmdline`. Don't trust `pgrep -c` for counts.
- **Disconnect-safety**: pools launched `nohup … &` get re-parented to **systemd (PID 1)**
  and survive SSH drop (verified). Background *monitor* Bash tasks DO get killed on
  disconnect — that's fine, data keeps accruing; just regenerate on reconnect. (Memory
  rule `feedback-tmux-longruns`: prefer tmux/nohup for long runs.)
- **GPU map**: `--device cuda:N` ≠ physical N (see repo `gpu_map.md`). 49GB cards =
  cuda:0,1,2,3; 24GB = cuda:4,5. Each U-Net job ~8GB → co-reside ≤3 on 49GB, ≤2 on 24GB.
- **Shared `job_NNN.log` names** across concurrent pools interleave — rely on JSON +
  the analyzers, not per-job logs, for truth. Grep ALL logs for `Traceback|out of memory`.
- **`run_first_iter` auto-skips** when `len(data[portion][lr]) >= --max_runs` (raises →
  pool moves on). So reruns are safe and you can make one pool *block* another by
  filling `max_runs` (e.g. ρ=100 pool fills 6 reps at the main pool's lrs → main skips).
- **ρ=100 is seed-independent** → single seed 42 × reps; std from the reps, not seeds.

---

## 6. TODO / status

- [ ] **Aug curve** (Task 0): fill ρ=30–90 for HF/none with the 5-seed lr-swept batch
      (4× already dense via base_v2 fallback; replace fallback as new data lands).
- [ ] **Margin AL** 5-seed swept → finish ρ→60 (in progress, ρ≈15–22).
- [ ] **Core-set** 5-seed swept → finish ρ→60 (all 5 seeds now running).
- [ ] **Entropy** (Uncertainty group) — run 5-seed swept (skip Confidence ≡ margin).
- [ ] **Cluster-Margin** (Hybrid group) — run 5-seed swept (fills `figs/hybrid.png`).
- [ ] Once new data passes a portion, **drop the old base_v2/main_aug4 fallback** so all
      curves are pure 5-seed (the fallback is only a progress-preview crutch).
- [x] **SimCLR pretraining code BUILT** (2026-06-23) at `simclr/` — `model.py`
      (`UNetEncoderSimCLR` = Conv1..Conv6 + GAP + MLP head; encoder keys are a clean
      SUBSET of `Optim_U_Net` → `strict=False` load = 0 unexpected, verified), `data.py`
      (grayscale flat-folder 2-view; same aug recipe as classification = color-free;
      pretrains on 736 train imgs / `--all_images` for 1224), `trainer.py` (InfoNCE +
      GradCache; **InstanceNorm → gradcache is EXACT**, no BN caveat), `run.py` (bs256/
      ep500/Adam/cosine/τ0.07/out_dim32; crop 128 + micro_bs 32 for VRAM). CPU smoke-test
      passed. **NOT yet run** (heavy ~hrs; wait for free GPU + keep ≤80% load).
      Run: `python3 thesis/chapter_5/segmentation/simclr/run.py --device cuda:N`, then
      downstream `--init simclr --simclr_path simclr/ckpt/..._encoder.pkl`.
      ⚠️ from-scratch (no ImageNet for this custom encoder) → **SWEEP lr, don't fix 2e-4**;
      then fine-tune with **frozen-decoder warm-up / differential LR** (Taleb NeurIPS'20).
- [ ] Run SimCLR pretrain (lr sweep) + downstream init comparison (random vs simclr).
- [ ] Multi-seed error bars on the 100% aug tie (currently seed42).

### Timing reference
~26 s/epoch at ρ=30/aug4 solo; **ρ=100/aug4 ≈ 35–40 min solo, ~60–75 min co-resident**
(368 batches/epoch × 25). High-ρ aug runs are the slow tail.
