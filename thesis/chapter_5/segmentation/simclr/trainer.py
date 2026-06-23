"""
SimCLR trainer for the U-Net encoder — InfoNCE loss + GradCache (micro-batched
true full-batch InfoNCE) so batch-size 256 fits under VRAM. Faithful to the
classification trainer (`SSL/simclr/simclr.py`).

BONUS: `Optim_U_Net` / `conv_block` use **InstanceNorm** (per-sample), so the one
GradCache caveat (BatchNorm computed per micro-batch) does NOT apply here — the
micro-batched gradient is EXACT vs a single full-batch forward.
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def accuracy(output, target, topk=(1,)):
    maxk = max(topk); bs = target.size(0)
    _, pred = output.topk(maxk, 1, True, True); pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / bs)
            for k in topk]


class SimCLRTrainer(object):
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model.to(args.device)
        self.opt = optimizer
        self.sched = scheduler
        self.args = args
        self.crit = torch.nn.CrossEntropyLoss().to(args.device)

    def info_nce(self, features, bs):
        labels = torch.cat([torch.arange(bs) for _ in range(self.args.n_views)], 0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.args.device)
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim = sim[~mask].view(sim.shape[0], -1)
        pos = sim[labels.bool()].view(labels.shape[0], -1)
        neg = sim[~labels.bool()].view(sim.shape[0], -1)
        logits = torch.cat([pos, neg], 1) / self.args.temperature
        lab = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        return logits, lab

    def _step_full(self, images):
        """Plain full-batch SimCLR step: one forward over all 2*bs views, InfoNCE,
        one backward. Unambiguously the true bs InfoNCE (no GradCache). Use when
        VRAM allows (preferred)."""
        bs = images.size(0) // self.args.n_views
        self.opt.zero_grad()
        feats = self.model(images)                  # [2*bs, out_dim], single forward
        logits, labels = self.info_nce(feats, bs)
        loss = self.crit(logits, labels)
        loss.backward()
        self.opt.step()
        return loss, logits, labels

    def _step_gradcache(self, images, micro_bs):
        """3-pass GradCache step → true full-batch InfoNCE under low VRAM."""
        N = images.size(0)
        bs = N // self.args.n_views
        self.opt.zero_grad()
        # (1) cache embeddings (no grad)
        with torch.no_grad():
            reps = torch.cat([self.model(images[s:s + micro_bs])
                              for s in range(0, N, micro_bs)], 0)
        reps = reps.detach().requires_grad_(True)
        # (2) full-batch InfoNCE → grad w.r.t. cached embeddings
        logits, labels = self.info_nce(reps, bs)
        loss = self.crit(logits, labels)
        loss.backward()
        rep_grads = reps.grad.detach()
        # (3) re-forward each micro-batch WITH grad, backprop cached grads
        for s in range(0, N, micro_bs):
            f = self.model(images[s:s + micro_bs])
            torch.autograd.backward(f, grad_tensors=rep_grads[s:s + f.size(0)])
        self.opt.step()
        return loss, logits, labels

    def train(self, loader, micro_bs):
        a = self.args
        ck_dir = os.path.join(os.path.dirname(__file__), "ckpt")
        js_dir = os.path.join(os.path.dirname(__file__), "json")
        os.makedirs(ck_dir, exist_ok=True); os.makedirs(js_dir, exist_ok=True)
        base = f"unet_simclr_lr{a.lr}_bs{a.batch_size}_ep{a.epochs}"
        ck_path = os.path.join(ck_dir, base + ".pkl")
        enc_path = os.path.join(ck_dir, base + "_encoder.pkl")   # Conv1..Conv6 only
        js_path = os.path.join(js_dir, base + ".json")
        hist = {"arch": "unet_encoder", "lr": a.lr, "batch_size": a.batch_size,
                "epochs": a.epochs, "crop_size": a.crop_size, "micro_bs": micro_bs,
                "n_train": a.n_train, "history": []}

        mode = "full-batch (no gradcache)" if getattr(a, "full_batch", True) else f"gradcache micro_bs={micro_bs}"
        print(f"[U-Net SimCLR] bs={a.batch_size} {mode} crop={a.crop_size} "
              f"lr={a.lr} ep={a.epochs} on {a.n_train} train imgs")
        for ep in range(a.epochs):
            self.model.train()
            ep_loss = ep_t1 = ep_t5 = 0.0; nb = 0
            for images, _ in tqdm(loader, ncols=100, desc=f"ep{ep+1}/{a.epochs}"):
                images = torch.cat(images, 0).to(a.device)   # [n_views*bs, 1, H, W]
                if getattr(a, "full_batch", True):
                    loss, logits, labels = self._step_full(images)
                else:
                    loss, logits, labels = self._step_gradcache(images, micro_bs)
                t1, t5 = accuracy(logits, labels, topk=(1, 5))
                ep_loss += loss.item(); ep_t1 += t1[0].item(); ep_t5 += t5[0].item(); nb += 1
            lr_now = self.sched.get_last_lr()[0]
            self.sched.step()
            rec = {"epoch": ep, "loss": ep_loss / nb, "top1": ep_t1 / nb,
                   "top5": ep_t5 / nb, "lr": lr_now}
            hist["history"].append(rec)
            print(f"  ep{ep}: loss {rec['loss']:.4f}  top1 {rec['top1']:.1f}%  lr {lr_now:.6f}")
            with open(js_path, "w") as f:
                json.dump(hist, f, indent=2)
            # periodic + final checkpoints
            if (ep + 1) % 50 == 0 or ep == a.epochs - 1:
                torch.save(self.model.state_dict(), ck_path)
                torch.save(self.model.encoder_state_dict(), enc_path)
        print(f"✓ saved {ck_path}\n✓ saved encoder-only {enc_path}")
        return ck_path, enc_path
