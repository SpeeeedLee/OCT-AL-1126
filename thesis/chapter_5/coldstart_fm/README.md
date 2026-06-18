# Cold-start AL via Foundation-Model embeddings + clustering (Ch.5 §5.2)

Implements the selection algorithm of **"From Cold Start to Active Learning:
Embedding-Based Scan Selection for Medical Image Segmentation"**
(arXiv:2601.18532, 2026), made *model-agnostic*: any foundation model (FM) can
be plugged in as the frozen feature extractor, then the same clustering picks
the initial labeled set. We use it to choose the initial labeled images for our
Skin-OCT classification AL (replacing the random `b₀`).

## Pipeline (3 stages, all from repo root)

```
                extract_embeddings.py            select_coldstart.py
  D_train (2032) ───────────────────►  {model}.pt ──────────────────►  labeled_ids/{model}.json
   ImageFolder      frozen FM forward    [N,D] cache   tSNE→silhouette-k→KMeans      {2.5,10,20%}
   (gray→RGB)                                          →medoid+proportional+FPS
```

1. **`extract_embeddings.py`** — runs one FM over all 2032 train images in
   ImageFolder order (row `i` == train index `i`, identical to `run_AL`/`get_data`),
   caches `{embeddings[N,D], indices, labels, classes}` to `embeddings/{model}.pt`.
   ```bash
   python3 thesis/chapter_5/coldstart_fm/extract_embeddings.py --all --device cuda:5
   # or one model:  --model dinov2:base
   ```

2. **`select_coldstart.py`** — the 2026 algorithm:
   embeddings → **t-SNE 2D** → for each k, KMeans + mean **silhouette** → k̂=argmax
   → KMeans(k̂) → each cluster **medoid** is a seed → remaining budget split
   **proportional to cluster size** → **greedy farthest-point** within each cluster.
   Writes `labeled_ids/{model}.json` at portions **2.5 / 10 / 20 %** in the exact
   `labeled_ids` schema used by the AL runner.
   ```bash
   python3 thesis/chapter_5/coldstart_fm/select_coldstart.py --model dinov2:base
   # --reduce {tsne2d(default,faithful) | pca50 | none}   --portions 2.5 10 20
   ```

3. **Train** on the selected set (two interchangeable consumers):
   - **One-shot** (`run_coldstart_fm.py`, standalone — no core file touched), same
     θ²-SimCLR/aug4 protocol as `run_5_3_redistribution.py`. **No seed needed**: the
     selected set is deterministic (fixed IDs), so we just tune lr, run `--runs` (=3)
     times per lr, pick the best lr, and report **mean±std over its runs** (the thesis
     "fixed-subset" convention — *not* per-seed-over-random-subsets). Batch driver:
     `run_coldstart_fm.sh` (env `MODELS= PORTIONS= DEVICE= PARALLEL=`, no `SEEDS`).
     ```bash
     python3 thesis/chapter_5/coldstart_fm/run_coldstart_fm.py \
         --model dinov2:base --portion 10 --device cuda:4 [--parallel_runs 3]
     # -> classification/exp_results/classification_hard/coldstart_fm_simclr/{model}_bs16.json
     # (legacy: --seed N writes {model}_seedN_bs16.json; the plot pools those as extra reps)
     ```
   - **AL initial pool** — reuse the EXISTING `run_AL.py` resume flags (no core change):
     ```bash
     python3 classification/run_AL.py --task_type hard --AL_strategy margin \
         --pretrained_weights simclr --simclr_path SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl \
         --resume_labeled_ids thesis/chapter_5/coldstart_fm/labeled_ids/dinov2__base.json \
         --resume_from 2.5 --portion_start 5 --portion_end 62.5 --portion_interval 2.5 \
         --seed 42 --device cuda:0
     ```
     The cold-start set at ρ=2.5% becomes the initial labeled pool; AL then queries
     5→60% with the chosen strategy. (At the 2.5% anchor the runner has no per-seed
     cold-start lr to look up — the set isn't the random seed-subset — so it falls
     back to a normal lr sweep; harmless.)

## Models (`<family>:<size>`)

| family | sizes | emb dim | deps | domain note |
|---|---|---|---|---|
| `simclr` | resnet18 | 512 | local θ² ckpt | **our own** in-domain SimCLR backbone (the §5.2 reference) |
| `resnet_imagenet` | resnet18/50/101 | 512/2048/2048 | torchvision | ImageNet (natural) |
| `dinov2` | small/base/large | 384/768/1024 | transformers | natural (SSL) |
| `clip` | base/large | 512/768 | transformers | natural (image tower) |
| `radimagenet` | resnet50 | 2048 | torchvision + HF weights | **the paper's extractor**; CT/MRI/US |
| `biomedclip` | base | 512 | open_clip | 15M PMC biomedical pairs |
| `retfound` | oct/cfp | 1024 | timm + **gated** HF weights | retinal OCT/CFP (MAE ViT-L) |
| `medimageinsight` | base | 1024 | local `feature_extract/` pkg + einops/mup/fvcore/tenacity | Microsoft UniCL, multi-domain medical |

**Available model IDs (copy-paste) + status** (`python3 extractors.py` prints the registry):
```
# ✅ DONE — embeddings + labeled_ids/{2.5,10,20%} already produced (13 models)
simclr:resnet18            # OUR OWN θ² SimCLR backbone (in-domain ref point)
resnet_imagenet:resnet18   resnet_imagenet:resnet50   resnet_imagenet:resnet101
dinov2:small               dinov2:base                dinov2:large
clip:base                  clip:large
radimagenet:resnet50       biomedclip:base
retfound:oct               medimageinsight:base
# ⏸ available but not run (further from skin OCT than retfound:oct)
retfound:cfp               # retinal color-fundus (needs same HF gated access as :oct)
```
To add a new family: subclass `FoundationExtractor` in `extractors.py` (define
`transform` + `_forward` -> [B,D]) and register it in `_FAMILY_OF`. Nothing else changes.

- **Embedding choice**: for models with a classification head we take the
  **penultimate** representation (the layer feeding the head), never the logits.
- **Grayscale**: OCT images are mode-L; ImageFolder's default loader replicates to
  3-channel RGB, and each FM uses its own canonical preprocessing.
- **Domain caveat**: our data is *grayscale skin OCT*. None of these FMs match it
  exactly — RadImageNet is radiology, BiomedCLIP is biomedical literature, RETFound
  is *retinal* OCT (modality-match, anatomy-mismatch). That mismatch is itself the
  comparison §5.2 studies (off-the-shelf FM embedding vs our own SimCLR θ²).

### Gated / extra-setup models (notes)
- **`medimageinsight:base`** — uses the **local** package in `feature_extract/MedImageInsights/`
  (Microsoft UniCL; vision weights `2024.09.27/vision_model/medimageinsigt-v1.0.0.pt`). Needs
  `pip install einops mup fvcore tenacity`. Consumes raw image **bytes** (base64) and does its own
  preprocessing → `extract_embeddings.py` feeds it file paths (`input_mode='path'`). **Important fix:**
  the bundled `load_model()` forgets `model.eval()`, leaving dropout ON (non-deterministic features);
  our `MedImageInsightExtractor` forces `.eval()`. Device honoured via `clf.device`.
- **`retfound:oct`** — weights are **gated** on HF
  (`YukunZhou/RETFound_mae_natureOCT`). Request access on the model page, then
  `huggingface-cli login` (or `export HF_TOKEN=...`). The extractor loads the
  official `.pth` into a plain `timm` ViT-L/16 — **no remote code**.
- **MedImageInsight** — not yet implemented: the only HF mirror
  (`lion-ai/MedImageInsights`) ships a custom package that must be imported
  (executes external code). Pending explicit go-ahead.

## Plotting the fine-tune results
`plot_coldstart_fm.py` — one **bar chart per portion** (title `ρ = XX%`), one bar per model
(feature extractor), grouped by family with shared hue (size shade differs; accuracy printed
on top, italic+underlined size token inside the bar), **black std error bars**, grey **Random**
bar + reference line. Aggregation: FM bars = best-lr → **mean±std (ddof=1) over that lr's pooled
runs** (fixed set; any legacy per-seed files are pooled as extra reps). Random = θ² cold-start
per-seed-best over 5 random subsets (its std = subset variance).
Also prints a structured per-seed table to the terminal. Matches Ch4/5 font + style.
```bash
python3 thesis/chapter_5/coldstart_fm/plot_coldstart_fm.py            # ρ=2.5/10/20 at once
python3 thesis/chapter_5/coldstart_fm/plot_coldstart_fm.py --portions 10
# -> thesis/chapter_5/figs/5_2_coldstart_fm_rho{p}.png
```
Models still training are skipped (logged) so it's safe to run mid-sweep.

## Files
- `extractors.py` — base `FoundationExtractor` + one subclass per family; registry.
- `weights.py` — resolves/downloads non-auto weights (RadImageNet).
- `extract_embeddings.py` — cache embeddings.
- `select_coldstart.py` — the clustering selection (faithful to the paper).
- `run_coldstart_fm.py` — standalone one-shot trainer on a selected set.
- `embeddings/*.pt`, `labeled_ids/*.json` — outputs.

## Notes on the real-data behaviour
The silhouette step picks **small k̂** on our OCT embeddings (k̂≈2–4 depending on
FM), i.e. the data's natural structure in these embedding spaces is a few broad
groups rather than the 7 diagnostic classes — most of the budget is therefore
spread by proportional farthest-point coverage, not by per-class clustering.
(On synthetic 7-blob data the same code recovers k̂=7, confirming correctness.)
k̂ is reported per model in each JSON (`k_hat`).
