# Chapter 5 — Active Learning for Nuclei Segmentation

Extends the Chapter-4 classification AL story to **2D semantic segmentation**
(skin-OCT cell-nuclei). Self-contained under `thesis/chapter_5/segmentation/`;
the source `segmentation/` package is **not modified** (pure U-Net `model.py`,
`loss.py`, `tool.py` are copied verbatim).

## Scope / decisions (locked with the user)

- **Task**: nuclei segmentation **only** (binary, `ds/segmentation/cell/` masks).
  No layer segmentation.
- **Model**: pure `Optim_U_Net` — **no deep supervision, no deep feature
  sharing** (both already disabled in the source model).
- **Selection unit = whole image** (no patch / region selection).
- **Single fold** (`--fold 0`), **no 5-fold cross-validation**.
- **Metric**: train, keep the checkpoint at the **lowest validation-loss epoch**,
  evaluate it on the held-out **test split** → per-image mean **Dice** (headline).
  Per-epoch train/val loss & Dice are also recorded.
- **Augmentation**: offline flip, image+mask synced. `--aug_factor`
  `1`=none `2`=+HF `3`=+HF+VF `4`=+HF+VF+HFV (default 4, = Ch4 "aug4").
- **AL protocol**: the `.py` accepts any `--lr`/`--seed`; the run scripts keep
  **lr fixed** across portions and vary **seed ∈ {10,24,38,42,57}**. Fresh U-Net
  each portion; the previous portion's best-val-loss model is the **selector**.

## AL strategies (classification → segmentation)

Image-level adaptation, following the standard literature recipe (aggregate
per-pixel scores; encoder features as the image descriptor):

| Strategy         | File                       | How it lifts to segmentation |
|------------------|----------------------------|------------------------------|
| `margin`         | `AL_strategy/uncertainty.py` | per-pixel margin `1-|2p-1|`, **mean over all pixels** of the image |
| `coreset`        | `AL_strategy/diversity.py`   | k-Center-Greedy on **encoder bottleneck (Conv6) GAP** embeddings, conditioned on the labeled set |
| `cluster_margin` | `AL_strategy/hybrid.py`      | margin candidate pool → L2-norm encoder embeddings → HAC (avg linkage, eps = median-dist·0.5) → round-robin by ascending cluster size |
| `random`         | (in `run_AL.py`)             | passive baseline |

## Run (from repo root, `conda activate oct-env`)

Cold-start passive baseline:
```bash
python3 thesis/chapter_5/segmentation/run_first_iter.py \
    --dataroot ./ds/segmentation --portion 30 --seed 42 \
    --aug_factor 4 --lr 0.001 --epoch 25 --device cuda:0
# or the sweep over 5 seeds × portions:
DEVICE=cuda:0 ./thesis/chapter_5/segmentation/scripts/run_coldstart.sh
```

Active learning:
```bash
python3 thesis/chapter_5/segmentation/run_AL.py --AL_strategy margin \
    --dataroot ./ds/segmentation --portion_start 5 --portion_end 60 \
    --portion_interval 2.5 --seed 42 --aug_factor 4 --lr 0.001 --device cuda:0
# or the full sweep (4 strategies × 5 seeds):
DEVICE=cuda:0 ./thesis/chapter_5/segmentation/scripts/run_al.sh
```

Aggregate + plot:
```bash
python3 thesis/chapter_5/segmentation/aggregate.py --plot
```

## Results JSON

- Cold-start: `exp_results/nuclei/cold_start_<init>/random_<seed>_bs<bs>.json`
  → `{ portion: { lr: [ {test_dice, best_val_epoch, best_val_loss, best_val_dice,
  aug_factor, seed, n_labeled, per_epoch:{train_loss,val_loss,train_dice,val_dice}} ] } }`
- AL: `exp_results/nuclei/AL_<init>/<strategy>_seed<seed>_bs<bs>.json`
  → same run dict **plus** `selected_idx` (picked this round) and `labeled_idx`
  (cumulative labeled set) so every AL selection is fully reproducible.

`<init>` is `random` now; `simclr` is reserved (see below).

## SimCLR init (later)

`train_unet`/both runners accept `--init simclr --simclr_path <ckpt>`; the ckpt
is loaded into the U-Net with `strict=False` (encoder warm-start, decoder
random). The SSL pretraining script itself (encoder-only SimCLR on the OCT
images) is **not built yet** — deferred per the plan. Realistic note: there is no
off-the-shelf ImageNet checkpoint for this custom InstanceNorm U-Net encoder, so
"init" is either random or self-supervised.
