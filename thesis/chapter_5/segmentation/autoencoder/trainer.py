"""
Autoencoder pretraining loop, following Fu et al. (TMI 2024) settings:
AdamW (lr 1e-5, weight_decay 0.01), 5-epoch linear warm-up (2e-6 -> 1e-5),
then halve lr when the reconstruction loss plateaus (patience 3). MSE loss.

Saves the full AE and the ENCODER-ONLY state dict (Conv1..Conv6) — the latter is
what downstream `run_first_iter.py --init ae --simclr_path <...>_encoder.pkl`
loads into Optim_U_Net (strict=False). Intermediate encoder checkpoints are
dumped every `save_every` epochs so a short run can be fine-tuned early.
"""
import os
import time
import torch
import torch.nn as nn


class AETrainer:
    def __init__(self, model, args):
        self.a = args
        self.dev = args.device
        self.model = model.to(self.dev)
        self.mse = nn.MSELoss()
        self.base_lr = args.lr
        self.warmup = args.warmup
        self.warmup_start = 2e-6
        self.opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.5, patience=3)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        self.tag = f"unet_ae_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"

    def _set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def _save_encoder(self, suffix=""):
        path = os.path.join(self.a.ckpt_dir, f"{self.tag}_encoder{suffix}.pkl")
        torch.save(self.model.encoder_state_dict(), path)
        return path

    def train(self, loader):
        n = len(loader)
        for ep in range(1, self.a.epochs + 1):
            # linear warm-up of the lr over the first `warmup` epochs
            if self.warmup and ep <= self.warmup:
                frac = ep / self.warmup
                self._set_lr(self.warmup_start + frac * (self.base_lr - self.warmup_start))

            self.model.train()
            t0, tot = time.time(), 0.0
            for x in loader:
                x = x.to(self.dev, non_blocking=True)
                recon = self.model(x)
                loss = self.mse(recon, x)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                tot += loss.item()
            avg = tot / max(n, 1)

            if not (self.warmup and ep <= self.warmup):
                self.sched.step(avg)   # plateau-halving only after warm-up
            lr_now = self.opt.param_groups[0]["lr"]
            print(f"  AE E{ep}/{self.a.epochs}: recon_mse={avg:.6f} "
                  f"lr={lr_now:.2e} {time.time()-t0:.1f}s", flush=True)

            if self.a.save_every and ep % self.a.save_every == 0 and ep != self.a.epochs:
                p = self._save_encoder(f"_ep{ep}")
                print(f"    ↳ encoder ckpt @ep{ep} -> {p}", flush=True)

        full = os.path.join(self.a.ckpt_dir, f"{self.tag}.pkl")
        torch.save(self.model.state_dict(), full)
        enc = self._save_encoder()
        print(f"\n✓ AE done. full -> {full}\n            encoder -> {enc}", flush=True)
        return enc
