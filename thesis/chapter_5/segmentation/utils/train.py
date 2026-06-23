"""
Central U-Net training for Chapter 5 nuclei-segmentation experiments.

Single entry point `train_unet(...)` used by BOTH the cold-start baseline
(run_first_iter.py) and the active-learning loop (run_AL.py), so that the
training protocol is identical everywhere.

Protocol (decided with the user):
  - Pure U-Net (Optim_U_Net, no deep supervision / no deep feature sharing).
  - Loss = BinaryDiceLoss, Adam, StepLR.
  - Offline flip augmentation on the TRAINING set only (image+mask synced).
  - Model selection = the epoch with the LOWEST validation loss; that
    checkpoint is restored and evaluated on the held-out TEST split.
  - Headline metric = per-image mean Dice on the test split.
  - Single fold (no cross-validation).
  - Per-epoch train/val loss and Dice are recorded and returned.

A `simclr_path` argument is wired in (load_state_dict strict=False into the
U-Net encoder) for the later SimCLR-init experiment; with simclr_path=None the
model is randomly initialised.
"""
import copy
import time
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm

from thesis.chapter_5.segmentation.utils.data import (
    data_loader, o_data, g_data_cell_binary, expand_with_aug,
)
from thesis.chapter_5.segmentation.utils.model import Optim_U_Net
import thesis.chapter_5.segmentation.utils.loss as L
from thesis.chapter_5.segmentation.utils.tool import compute_dice_binary

HEIGHT = 512
WIDTH = 384


def build_model(input_nc, device, simclr_path=None):
    """Create a pure U-Net. Optionally warm-start the encoder from a SimCLR ckpt."""
    model = Optim_U_Net(img_ch=input_nc, output_ch=1, USE_DS=False, USE_DFS=False)
    if simclr_path:
        state = torch.load(simclr_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"✓ SimCLR encoder loaded from {simclr_path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return model.to(device)


@torch.no_grad()
def _evaluate(model, files, opath, gpath_cell, device, loss_func):
    """Per-image mean Dice + mean loss over `files` (no augmentation)."""
    model.eval()
    losses, dices = [], []
    loader = Data.DataLoader(dataset=list(files), batch_size=8, shuffle=False, num_workers=4)
    for batch in loader:
        img = o_data(opath, batch, WIDTH, HEIGHT)
        gt = g_data_cell_binary(gpath_cell, batch, WIDTH, HEIGHT)
        INPUT = torch.from_numpy(img.astype(np.float32)).to(device=device, dtype=torch.float)
        target = torch.from_numpy(gt.astype(np.float32)).to(device=device, dtype=torch.float)
        out = model(INPUT)
        losses.append(loss_func(out, target).item())
        pred = (out > 0.5).float().cpu().numpy()
        for k in range(len(batch)):
            dices.append(compute_dice_binary(pred[k:k + 1], gt[k:k + 1]))
    return float(np.mean(losses)), float(np.mean(dices))


def train_unet(label_idx, opt, device):
    """
    Train a U-Net on the given labeled filenames and evaluate on the test split.

    Args (opt attributes):
        dataroot, fold, input_nc, lr, step, epoch, batch_size, aug_factor
        simclr_path (optional, default None)
    Args:
        label_idx: list of training filenames (bare names; aug applied here)
        device: torch device

    Returns a dict:
        {
          "test_dice": float,           # per-image mean Dice on test split (headline)
          "best_val_loss": float,
          "best_val_epoch": int,        # 1-indexed
          "best_val_dice": float,       # val Dice at the best-val-loss epoch
          "per_epoch": {train_loss, val_loss, train_dice, val_dice: [...]},
          "n_labeled": int,
          "aug_factor": int,
          "model": <trained model restored to best-val-loss weights>,
        }
    """
    gpath_cell = opt.dataroot + "/cell/"
    opath = opt.dataroot + "/image/"
    aug_factor = int(getattr(opt, "aug_factor", 1))
    simclr_path = getattr(opt, "simclr_path", None)

    # Fixed val/test split (single fold)
    _, valid_data_LD, test_data_LD = data_loader(opath, opt.fold)
    valid_files, test_files = [], []
    for b in valid_data_LD:
        valid_files.extend(b)
    for b in test_data_LD:
        test_files.extend(b)

    # Build augmented training items
    train_items = expand_with_aug(list(label_idx), aug_factor)
    train_LD = Data.DataLoader(dataset=train_items, batch_size=opt.batch_size,
                               shuffle=True, num_workers=4)

    print('\n' + '=' * 80)
    print(f'TRAIN U-Net | labeled={len(label_idx)} imgs x aug{aug_factor} '
          f'= {len(train_items)} items | val={len(valid_files)} test={len(test_files)}')
    print('=' * 80)

    model = build_model(opt.input_nc, device, simclr_path=simclr_path)
    loss_func = L.BinaryDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.1)

    # --- frozen-decoder warm-up (for SimCLR encoder init) ---
    # First `warmup` epochs: encoder FROZEN (decoder-only trains; lr decays normally).
    # After warmup: encoder UNFROZEN, encoder+decoder share the SAME lr to the end
    # (NO differential LR). Total epochs unchanged (= opt.epoch) for fairness.
    warmup = int(getattr(opt, "warmup", 0))
    enc_mods = [model.Conv1, model.Conv2, model.Conv3,
                model.Conv4, model.Conv5, model.Conv6]

    def _set_encoder_trainable(flag):
        for m in enc_mods:
            for p in m.parameters():
                p.requires_grad = flag

    if warmup > 0:
        _set_encoder_trainable(False)
        print(f"✓ Warm-up: encoder FROZEN for first {warmup} epochs (decoder-only), "
              f"then unfreeze (same lr, no differential LR)")

    rec = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_dice = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for EPOCH in range(opt.epoch):
        if warmup > 0 and EPOCH == warmup:
            _set_encoder_trainable(True)
            print(f"✓ Warm-up done (epoch {EPOCH}): encoder UNFROZEN — "
                  f"encoder+decoder now train together at the same lr")
        start = time.time()
        model.train()
        tr_loss, tr_dice = [], []
        prog = tqdm(train_LD, desc=f'E{EPOCH+1}/{opt.epoch}', ncols=100,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for batch in prog:
            img = o_data(opath, batch, WIDTH, HEIGHT)
            gt = g_data_cell_binary(gpath_cell, batch, WIDTH, HEIGHT)
            INPUT = torch.from_numpy(img.astype(np.float32)).to(device=device, dtype=torch.float)
            target = torch.from_numpy(gt.astype(np.float32)).to(device=device, dtype=torch.float)

            out = model(INPUT)
            loss = loss_func(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss.append(loss.item())
            pred = (out > 0.5).float()
            tr_dice.append(compute_dice_binary(pred.cpu().detach().numpy(), gt))
            prog.set_postfix({'Loss': f'{loss.item():.4f}'})

        scheduler.step()

        # Validation (per-image, no aug)
        v_loss, v_dice = _evaluate(model, valid_files, opath, gpath_cell, device, loss_func)
        t_loss, t_dice = float(np.mean(tr_loss)), float(np.mean(tr_dice))
        rec["train_loss"].append(round(t_loss, 6))
        rec["val_loss"].append(round(v_loss, 6))
        rec["train_dice"].append(round(t_dice, 6))
        rec["val_dice"].append(round(v_dice, 6))

        # Model selection: lowest validation loss
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch = EPOCH + 1
            best_val_dice = v_dice
            best_state = copy.deepcopy(model.state_dict())

        print(f'  E{EPOCH+1}: train L {t_loss:.4f} D {t_dice:.4f} | '
              f'val L {v_loss:.4f} D {v_dice:.4f} | {time.time()-start:.1f}s'
              f'{"  <- best" if best_epoch == EPOCH + 1 else ""}')

    # Restore best-val-loss checkpoint and evaluate on TEST split
    model.load_state_dict(best_state)
    _, test_dice = _evaluate(model, test_files, opath, gpath_cell, device, loss_func)

    print('\n' + '-' * 80)
    print(f'BEST val-loss epoch {best_epoch}/{opt.epoch} (val_loss={best_val_loss:.4f}, '
          f'val_dice={best_val_dice:.4f}) -> TEST Dice = {test_dice:.4f}')
    print('-' * 80)

    return {
        "test_dice": round(test_dice, 4),
        "best_val_loss": round(best_val_loss, 6),
        "best_val_epoch": best_epoch,
        "best_val_dice": round(best_val_dice, 4),
        "per_epoch": rec,
        "n_labeled": len(label_idx),
        "aug_factor": aug_factor,
        "model": model,
    }
