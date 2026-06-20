import logging
import os
import sys
import json
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        # self.writer = SummaryWriter()
        # self.writer = SummaryWriter(log_dir='./SSL/simclr/tb_logs')
        # logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        # time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        # log_dir = os.path.join("./SSL/simclr/tb_logs", time_str)
        # self.writer = SummaryWriter(log_dir=log_dir)
        # logging.basicConfig(filename=os.path.join(log_dir, 'training.log'),
        #                     level=logging.DEBUG)
        # 建立包含更多資訊的log目錄名稱
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = f"{self.args.arch}_lr{self.args.lr}_bs{self.args.batch_size}_ep{self.args.epochs}_{time_str}"
        log_dir = os.path.join("./SSL/simclr/tb_logs", exp_name)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, 'training.log'),
            level=logging.DEBUG
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    # ------------------------------------------------------------------ #
    # Held-out contrastive validation (overfitting monitor). Does NOT touch
    # the model/optimizer; RNG is snapshotted+restored by the caller so the
    # training stream is byte-identical with or without validation.
    # ------------------------------------------------------------------ #
    def _info_nce_dyn(self, features):
        """Same InfoNCE as info_nce_loss but batch size inferred from features
        (val-half = 254 imgs ≠ args.batch_size)."""
        bs = features.shape[0] // self.args.n_views
        labels = torch.cat([torch.arange(bs) for _ in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(self.args.device)
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim = sim[~mask].view(sim.shape[0], -1)
        positives = sim[labels.bool()].view(labels.shape[0], -1)
        negatives = sim[~labels.bool()].view(sim.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1) / self.args.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        return logits, labels

    @torch.no_grad()
    def evaluate_contrastive(self, val_loader):
        """Mean InfoNCE loss / top1 / top5 over the val set (eval mode). Returns dict."""
        was_training = self.model.training
        self.model.eval()
        tot_loss = tot1 = tot5 = 0.0
        n = 0
        for images, _ in val_loader:
            images = torch.cat(images, dim=0).to(self.args.device)
            features = self.model(images)
            logits, labels = self._info_nce_dyn(features)
            loss = self.criterion(logits, labels)
            t1, t5 = accuracy(logits, labels, topk=(1, 5))
            tot_loss += loss.item(); tot1 += t1[0].item(); tot5 += t5[0].item(); n += 1
        if was_training:
            self.model.train()
        return {"val_loss": tot_loss / n, "val_top1": tot1 / n, "val_top5": tot5 / n}

    def _run_validation(self, val_loader, epoch):
        """Validate every 10 epochs (and the final epoch). Deterministic val views
        (seed VAL_SEED) for a clean curve; RNG snapshot+restore → training unaffected."""
        import random
        import numpy as np
        if val_loader is None or not (epoch % 10 == 0 or epoch == self.args.epochs - 1):
            return None
        VAL_SEED = 12345
        rng = (torch.get_rng_state(), np.random.get_state(), random.getstate(),
               torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.manual_seed(VAL_SEED); np.random.seed(VAL_SEED); random.seed(VAL_SEED)
        try:
            m = self.evaluate_contrastive(val_loader)
        finally:
            torch.set_rng_state(rng[0]); np.random.set_state(rng[1]); random.setstate(rng[2])
            if rng[3] is not None:
                torch.cuda.set_rng_state_all(rng[3])
        return m

    def train(self, train_loader, val_loader=None):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        save_config_file(self.writer.log_dir, self.args)

        # 建立資料夾
        model_dir = "./SSL/simclr/ckpt"
        json_dir  = "./SSL/simclr/json"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(json_dir,  exist_ok=True)

        # 共用檔名 (不含副檔名)
        # _wval suffix when validation is on → never overwrite the original (no-val) ckpts
        vtag = "_wval" if val_loader is not None else ""
        base_name = f"{self.args.arch}_simclr_lr{self.args.lr}_bs{self.args.batch_size}_ep{self.args.epochs}{vtag}"
        model_path = os.path.join(model_dir, f"{base_name}.pkl")
        best_path  = os.path.join(model_dir, f"{base_name}_best.pkl")   # lowest val-loss ckpt
        json_path  = os.path.join(json_dir,  f"{base_name}.json")

        # JSON 結構初始化
        training_history = {
            "arch":       self.args.arch,
            "lr":         self.args.lr,
            "batch_size": self.args.batch_size,
            "epochs":     self.args.epochs,
            "history":    []   # 每個 epoch append 一筆
        }
        best_val, best_epoch = float("inf"), -1

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            epoch_loss = 0.0
            epoch_top1 = 0.0
            epoch_top5 = 0.0
            batch_count = 0

            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0).to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                epoch_loss += loss.item()
                epoch_top1 += top1[0].item()
                epoch_top5 += top5[0].item()
                batch_count += 1
                n_iter += 1

            avg_loss    = epoch_loss / batch_count
            avg_top1    = epoch_top1 / batch_count
            avg_top5    = epoch_top5 / batch_count
            current_lr  = self.scheduler.get_lr()[0]

            # TensorBoard
            self.writer.add_scalar('epoch_loss',      avg_loss,   global_step=epoch_counter)
            self.writer.add_scalar('epoch_acc/top1',  avg_top1,   global_step=epoch_counter)
            self.writer.add_scalar('epoch_acc/top5',  avg_top5,   global_step=epoch_counter)
            self.writer.add_scalar('learning_rate',   current_lr, global_step=epoch_counter)

            # Terminal
            print(f"Epoch: {epoch_counter}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Top1 Accuracy: {avg_top1:.2f}%")
            print(f"  Top5 Accuracy: {avg_top5:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            print("-" * 50)

            # JSON 動態更新 ← 每個 epoch 都寫入一次
            rec = {"epoch": epoch_counter, "loss": avg_loss, "top1": avg_top1,
                   "top5": avg_top5, "lr": current_lr}
            vm = self._run_validation(val_loader, epoch_counter)   # every 10 ep (+ last)
            if vm:
                rec.update(vm)
                self.writer.add_scalar('val/epoch_loss',     vm["val_loss"], global_step=epoch_counter)
                self.writer.add_scalar('val/epoch_acc/top1', vm["val_top1"], global_step=epoch_counter)
                self.writer.add_scalar('val/epoch_acc/top5', vm["val_top5"], global_step=epoch_counter)
                print(f"  [val] Loss: {vm['val_loss']:.4f}  Top1: {vm['val_top1']:.2f}%  Top5: {vm['val_top5']:.2f}%")
                if vm["val_loss"] < best_val:        # save lowest-val-loss ckpt
                    best_val, best_epoch = vm["val_loss"], epoch_counter
                    training_history["best_val_loss"], training_history["best_val_epoch"] = best_val, best_epoch
                    torch.save(self.model.state_dict(), best_path)
                    print(f"  [val] *** new best val_loss → saved {os.path.basename(best_path)} (ep{best_epoch})")
            training_history["history"].append(rec)
            with open(json_path, "w") as f:
                json.dump(training_history, f, indent=4)

            self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {avg_loss}\tTop1 accuracy: {avg_top1}")

        # 存 model
        torch.save(self.model.state_dict(), model_path)
        logging.info("Training has finished.")

    # ------------------------------------------------------------------ #
    # GradCache variant — ONLY for arch larger than resnet18 (see run.py).
    # Reproduces the EXACT full-batch (bs) InfoNCE loss/gradient while only
    # holding one micro-batch's activation graph at a time → lets resnet50/101/152
    # keep the same bs/ep/lr (and fp32, matching resnet18) under limited VRAM.
    # Ref: Gao et al. 2021, "Scaling Deep Contrastive Learning Batch Size under
    # Memory Limited Setup". 3 passes per step:
    #   (1) no-grad micro-batch forwards → cache embeddings
    #   (2) full-batch InfoNCE on cached embeddings → grad w.r.t. each embedding
    #   (3) re-forward each micro-batch WITH grad, backprop the cached grad →
    #       accumulate model grads; then optimizer.step()
    # NOTE: fp32 (no autocast/scaler) to match resnet18. Caveat: BatchNorm is
    # computed per micro-batch (not over the full bs), the one unavoidable
    # deviation from a single full-batch forward — negatives are still the full bs.
    # ------------------------------------------------------------------ #
    def train_gradcache(self, train_loader, micro_bs, val_loader=None):
        save_config_file(self.writer.log_dir, self.args)

        model_dir = "./SSL/simclr/ckpt"
        json_dir  = "./SSL/simclr/json"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(json_dir,  exist_ok=True)

        # _wval suffix when validation is on → never overwrite the original (no-val) ckpts
        vtag = "_wval" if val_loader is not None else ""
        base_name = f"{self.args.arch}_simclr_lr{self.args.lr}_bs{self.args.batch_size}_ep{self.args.epochs}{vtag}"
        model_path = os.path.join(model_dir, f"{base_name}.pkl")
        best_path  = os.path.join(model_dir, f"{base_name}_best.pkl")   # lowest val-loss ckpt
        json_path  = os.path.join(json_dir,  f"{base_name}.json")

        training_history = {
            "arch":       self.args.arch,
            "lr":         self.args.lr,
            "batch_size": self.args.batch_size,
            "epochs":     self.args.epochs,
            "grad_cache": True,
            "micro_bs":   micro_bs,
            "history":    []
        }
        best_val, best_epoch = float("inf"), -1

        n_iter = 0
        logging.info(f"Start SimCLR (GradCache, micro_bs={micro_bs}) for {self.args.epochs} epochs.")
        print(f"[GradCache] arch={self.args.arch} bs={self.args.batch_size} "
              f"micro_bs={micro_bs} (fp32) — true full-batch InfoNCE, low VRAM")

        for epoch_counter in range(self.args.epochs):
            epoch_loss = epoch_top1 = epoch_top5 = 0.0
            batch_count = 0

            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0).to(self.args.device)   # [2*bs, C, H, W]
                N = images.size(0)
                self.optimizer.zero_grad()

                # (1) cache embeddings, no grad
                with torch.no_grad():
                    reps = torch.cat([self.model(images[s:s + micro_bs])
                                      for s in range(0, N, micro_bs)], dim=0)
                reps = reps.detach().requires_grad_(True)

                # (2) full-batch InfoNCE → grad w.r.t. cached embeddings (model untouched)
                logits, labels = self.info_nce_loss(reps)
                loss = self.criterion(logits, labels)
                loss.backward()
                rep_grads = reps.grad.detach()

                # (3) re-forward each micro-batch WITH grad, backprop cached grads → accumulate
                for s in range(0, N, micro_bs):
                    f = self.model(images[s:s + micro_bs])
                    torch.autograd.backward(f, grad_tensors=rep_grads[s:s + f.size(0)])
                self.optimizer.step()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                epoch_loss += loss.item()
                epoch_top1 += top1[0].item()
                epoch_top5 += top5[0].item()
                batch_count += 1
                n_iter += 1

            avg_loss = epoch_loss / batch_count
            avg_top1 = epoch_top1 / batch_count
            avg_top5 = epoch_top5 / batch_count
            current_lr = self.scheduler.get_lr()[0]

            self.writer.add_scalar('epoch_loss',     avg_loss,   global_step=epoch_counter)
            self.writer.add_scalar('epoch_acc/top1', avg_top1,   global_step=epoch_counter)
            self.writer.add_scalar('epoch_acc/top5', avg_top5,   global_step=epoch_counter)
            self.writer.add_scalar('learning_rate',  current_lr, global_step=epoch_counter)

            print(f"Epoch: {epoch_counter}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Top1 Accuracy: {avg_top1:.2f}%")
            print(f"  Top5 Accuracy: {avg_top5:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            print("-" * 50)

            rec = {"epoch": epoch_counter, "loss": avg_loss, "top1": avg_top1,
                   "top5": avg_top5, "lr": current_lr}
            vm = self._run_validation(val_loader, epoch_counter)
            if vm:
                rec.update(vm)
                self.writer.add_scalar('val/epoch_loss',     vm["val_loss"], global_step=epoch_counter)
                self.writer.add_scalar('val/epoch_acc/top1', vm["val_top1"], global_step=epoch_counter)
                self.writer.add_scalar('val/epoch_acc/top5', vm["val_top5"], global_step=epoch_counter)
                print(f"  [val] Loss: {vm['val_loss']:.4f}  Top1: {vm['val_top1']:.2f}%  Top5: {vm['val_top5']:.2f}%")
                if vm["val_loss"] < best_val:        # save lowest-val-loss ckpt
                    best_val, best_epoch = vm["val_loss"], epoch_counter
                    training_history["best_val_loss"], training_history["best_val_epoch"] = best_val, best_epoch
                    torch.save(self.model.state_dict(), best_path)
                    print(f"  [val] *** new best val_loss → saved {os.path.basename(best_path)} (ep{best_epoch})")
            training_history["history"].append(rec)
            with open(json_path, "w") as f:
                json.dump(training_history, f, indent=4)

            self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {avg_loss}\tTop1 accuracy: {avg_top1}")

        torch.save(self.model.state_dict(), model_path)
        logging.info("Training (GradCache) has finished.")


    # def train(self, train_loader):

    #     scaler = GradScaler(enabled=self.args.fp16_precision)

    #     # save config file
    #     save_config_file(self.writer.log_dir, self.args)

    #     n_iter = 0
    #     logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
    #     logging.info(f"Training with gpu: {self.args.disable_cuda}.")

    #     for epoch_counter in range(self.args.epochs):
    #         epoch_loss = 0.0
    #         epoch_top1 = 0.0
    #         epoch_top5 = 0.0
    #         batch_count = 0
            
    #         for images, _ in tqdm(train_loader):
    #             images = torch.cat(images, dim=0)
    #             images = images.to(self.args.device)

    #             with autocast(enabled=self.args.fp16_precision):
    #                 features = self.model(images)
    #                 logits, labels = self.info_nce_loss(features)
    #                 loss = self.criterion(logits, labels)
                
    #             self.optimizer.zero_grad()
    #             scaler.scale(loss).backward()
    #             scaler.step(self.optimizer)
    #             scaler.update()

    #             # 累積每個 epoch 的統計資訊
    #             top1, top5 = accuracy(logits, labels, topk=(1, 5))
    #             epoch_loss += loss.item()
    #             epoch_top1 += top1[0].item()
    #             epoch_top5 += top5[0].item()
    #             batch_count += 1

    #             n_iter += 1
            
    #         # 計算平均值
    #         avg_loss = epoch_loss / batch_count
    #         avg_top1 = epoch_top1 / batch_count
    #         avg_top5 = epoch_top5 / batch_count
    #         current_lr = self.scheduler.get_lr()[0]
            
    #         # 記錄到 tensorboard 並同時 print 到 terminal
    #         self.writer.add_scalar('epoch_loss', avg_loss, global_step=epoch_counter)
    #         self.writer.add_scalar('epoch_acc/top1', avg_top1, global_step=epoch_counter)
    #         self.writer.add_scalar('epoch_acc/top5', avg_top5, global_step=epoch_counter)
    #         self.writer.add_scalar('learning_rate', current_lr, global_step=epoch_counter)
            
    #         # Print 到 terminal
    #         print(f"Epoch: {epoch_counter}")
    #         print(f"  Loss: {avg_loss:.4f}")
    #         print(f"  Top1 Accuracy: {avg_top1:.2f}%")
    #         print(f"  Top5 Accuracy: {avg_top5:.2f}%")
    #         print(f"  Learning Rate: {current_lr:.6f}")
    #         print("-" * 50)
            
    #         # # warmup for the first 10 epochs
    #         # if epoch_counter >= 10:
            
    #         ## no warmup, directly consine decay
    #         self.scheduler.step()
        
    #         logging.debug(f"Epoch: {epoch_counter}\tLoss: {avg_loss}\tTop1 accuracy: {avg_top1}")
            
    #     torch.save(self.model.state_dict() ,f"./SSL/simclr/{self.args.arch}_simclr_lr{self.args.lr}_bs{self.args.batch_size}_ep{self.args.epochs}.pkl")
    #     logging.info("Training has finished.")

    # def train(self, train_loader):

    #     scaler = GradScaler(enabled=self.args.fp16_precision)

    #     # save config file
    #     save_config_file(self.writer.log_dir, self.args)

    #     n_iter = 0
    #     logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
    #     logging.info(f"Training with gpu: {self.args.disable_cuda}.")

    #     for epoch_counter in range(self.args.epochs):
    #         for images, _ in tqdm(train_loader):
    #             images = torch.cat(images, dim=0)

    #             images = images.to(self.args.device)

    #             with autocast(enabled=self.args.fp16_precision):
    #                 features = self.model(images)
    #                 logits, labels = self.info_nce_loss(features)
    #                 loss = self.criterion(logits, labels)
    #             self.optimizer.zero_grad()

    #             scaler.scale(loss).backward()

    #             scaler.step(self.optimizer)
    #             scaler.update()
    #             if n_iter % self.args.log_every_n_steps == 0:
    #                 top1, top5 = accuracy(logits, labels, topk=(1, 5))
    #                 self.writer.add_scalar('loss', loss, global_step=n_iter)
    #                 self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
    #                 self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
    #                 self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

    #             n_iter += 1
    #         print("epoch: ",epoch_counter, " loss: ", loss.item())
    #         # warmup for the first 10 epochs
    #         if epoch_counter >= 10:
    #             self.scheduler.step()
    #         logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
    #     torch.save(self.model.state_dict() ,f"./SSL/simclr/{self.args.arch}_simclr_lr{self.args.lr}_ep{self.args.epochs}.pkl")
    #     logging.info("Training has finished.")


    # def train_with_sam(self, train_loader):
    #     """
    #     SimCLR training function with Sharpness Aware Minimization (SAM) support
    #     """
    #     scaler = GradScaler(enabled=self.args.fp16_precision)

    #     # save config file
    #     save_config_file(self.writer.log_dir, self.args)

    #     n_iter = 0
    #     logging.info(f"Start SimCLR training with SAM for {self.args.epochs} epochs.")
    #     logging.info(f"Training with gpu: {self.args.disable_cuda}.")

    #     for epoch_counter in range(self.args.epochs):
    #         for images, _ in tqdm(train_loader):
    #             images = torch.cat(images, dim=0)
    #             images = images.to(self.args.device)

    #             # Zero gradients
    #             self.optimizer.zero_grad()

    #             # First forward pass and backward pass for SAM
    #             with autocast(enabled=self.args.fp16_precision):
    #                 features = self.model(images)
    #                 logits, labels = self.info_nce_loss(features)
    #                 loss = self.criterion(logits, labels)

    #             scaler.scale(loss).backward()

    #             # Define closure function for SAM
    #             def closure():
    #                 self.optimizer.zero_grad()
    #                 with autocast(enabled=self.args.fp16_precision):
    #                     features = self.model(images)
    #                     logits, labels = self.info_nce_loss(features)
    #                     loss = self.criterion(logits, labels)
    #                 scaler.scale(loss).backward()
    #                 return loss

    #             # SAM optimization step
    #             scaler.step(self.optimizer, closure)
    #             scaler.update()

    #             # Logging
    #             if n_iter % self.args.log_every_n_steps == 0:
    #                 top1, top5 = accuracy(logits, labels, topk=(1, 5))
    #                 self.writer.add_scalar('loss', loss, global_step=n_iter)
    #                 self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
    #                 self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
    #                 self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

    #             n_iter += 1

    #         print("epoch: ", epoch_counter, " loss: ", loss.item())
            
    #         # warmup for the first 10 epochs
    #         if epoch_counter >= 10:
    #             self.scheduler.step()
            
    #         logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

    #     torch.save(self.model.state_dict(), f"./SSL/simclr/{self.args.arch}_simclr_sam_lr{self.args.lr}_ep{self.args.epochs}.pkl")
    #     logging.info("Training with SAM has finished.")