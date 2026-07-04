"""
# CONVENTION: primary — Propagating-exception convention.

Training loop for the CUHK-X Small Model Track pipeline.

Features (all CMI-proven):
- AdamW optimizer + CosineAnnealingWarmRestarts
- EMA weight averaging
- Mixed precision (AMP)
- Mixup augmentation
- Modality dropout for missing modality robustness
- StratifiedGroupKFold cross-validation
- Early stopping + best checkpoint saving
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.amp import GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from data.dataset import HARDataset, discover_clips, build_clip_list
from data.collate import collate_fn
from data.augmentations import mixup_samples
from models import HARModel
from training.utils import ModelEMA, EarlyStopping, AverageMeter, accuracy, create_folds

# Upper-bound constants for loop guards
MAX_EPOCHS = 500
MAX_FOLDS = 10


class Trainer:
    """Training orchestrator for the HAR model.

    Args:
        config: Config object.
        fold: fold index for cross-validation.
        train_indices: indices for training split.
        val_indices: indices for validation split.
        clip_list: full list of (user, trial, action_id).
        clips: modality file dict.
        labels: label dict.
    """

    MAX_STEPS_PER_EPOCH = 100000  # upper-bound guard

    def __init__(self, config, fold, train_indices, val_indices, clip_list, clips, labels):
        self.config = config
        self.fold = fold
        self.device = torch.device(config.device)

        # Build datasets
        train_clips = [clip_list[i] for i in train_indices]
        val_clips = [clip_list[i] for i in val_indices]

        self.train_dataset = HARDataset(clips, labels, train_clips, config, is_train=True)
        self.val_dataset = HARDataset(clips, labels, val_clips, config, is_train=False)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, collate_fn=collate_fn, pin_memory=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, collate_fn=collate_fn, pin_memory=True
        )

        # Model
        self.model = HARModel(config).to(self.device)
        self.ema = ModelEMA(self.model, decay=config.ema_decay)
        self.scaler = GradScaler('cuda', enabled=config.mixed_precision)

        # Loss with optional class weighting
        class_weights = None
        if config.flags.use_class_weights:
            class_weights = self._compute_class_weights(train_clips, config.num_classes)
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=config.label_smoothing
        )
        # Auxiliary loss (coarse category — no class weighting needed)
        self.aux_criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Scheduler: linear warmup, then monotonic cosine decay over the
        # rest of the run. CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
        # was tried first, but on this dataset's short per-epoch step
        # count (~34 steps/epoch at 546 train clips / batch 16), each
        # restart snapped LR from ~1e-6 back to 1e-3 and visibly knocked
        # train loss back up (confirmed live: epoch 9->10 LR 2.5e-5->1e-3,
        # train loss 3.03->3.08) right as early stopping (patience=20) was
        # watching val_acc — a real run early-stopped at epoch 25 without
        # ever recovering past its pre-restart best. A single monotonic
        # decay removes that whiplash entirely. warmup_epochs (previously
        # declared in Config but never actually wired to anything) ramps
        # LR up from 10% instead of starting a freshly-initialized model
        # (including the cross_modal_attention block's random
        # MultiheadAttention weights, when enabled) at full LR on step 1.
        # Capped at 25% of the run so a short calibration run (e.g.
        # CUHKX_EPOCHS=10) doesn't spend half its budget warming up.
        warmup_epochs = min(config.warmup_epochs, max(1, config.epochs // 4))
        if warmup_epochs > 0 and config.epochs > warmup_epochs:
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        self.optimizer, start_factor=0.1, end_factor=1.0,
                        total_iters=warmup_epochs),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=config.epochs - warmup_epochs,
                        eta_min=config.lr_min),
                ],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=config.lr_min
            )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stop_patience, mode="max"
        )

        # Output paths
        self.fold_dir = config.output_dir / f"fold_{fold}"
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.fold_dir / "best_model.pth"
        self.last_checkpoint_path = self.fold_dir / "last_checkpoint.pth"

        # Metrics
        self.best_acc = 0.0

        # Resume from a previous interrupted run for this fold, if present.
        # Must run after model/optimizer/scheduler/scaler/ema/early_stopping
        # are all constructed above, since it overwrites their state.
        self.start_epoch = self._try_resume()

    @staticmethod
    def _compute_class_weights(clip_list, num_classes):
        """Compute inverse-sqrt class weights to handle 27.5:1 imbalance.

        Dampened inverse frequency: weight = 1 / sqrt(class_count).
        This prevents extreme up-weighting of the rarest classes (12 clips)
        while still giving them meaningful boost.

        Args:
            clip_list: list of (user_id, user, trial, action_id) tuples.
            num_classes: total number of classes the model predicts (config.num_classes).
                Must NOT be inferred from clip_list — a given train fold can
                happen to omit the globally rarest class entirely, which would
                silently shrink the weight vector below the classifier's width.

        Returns:
            (num_classes,) float tensor of class weights.
        """
        import numpy as np
        from collections import Counter

        counts = Counter(item[3] for item in clip_list)
        weights = np.ones(num_classes, dtype=np.float32)

        for cls_id in range(num_classes):
            cnt = counts.get(cls_id, 1)
            weights[cls_id] = 1.0 / np.sqrt(cnt)

        # Normalize so mean weight = 1
        weights = weights / weights.mean()
        return torch.from_numpy(weights).float()

    def _try_resume(self):
        """Load full training state from a previous interrupted run of this fold.

        Ephemeral environments (Colab, spot instances) can kill a run mid-fold
        long before it finishes. Re-running the same script re-instantiates
        this Trainer, which then picks up from the last epoch that finished,
        instead of restarting the fold from scratch.

        Returns:
            int: epoch index to resume training from (0 if no checkpoint,
            or if a checkpoint exists but doesn't match the current model
            architecture — see the architecture-mismatch handling below).
        """
        if not self.last_checkpoint_path.exists():
            return 0

        ckpt = torch.load(self.last_checkpoint_path, map_location=self.device,
                          weights_only=False)

        # output_dir (and therefore last_checkpoint_path) is keyed only by
        # config label (e.g. "synthesized"), not by which FeatureFlags or
        # scheduler produced it. Changing an architecture flag
        # (segment_pooling, cross_modal_attention, ...) or the LR
        # scheduler class between runs that reuse the same output_dir
        # leaves a checkpoint on disk that no longer matches this run's
        # model/optimizer/scheduler — any of the loads below can raise.
        # A snapshot of the freshly-constructed model's own state is
        # taken first and restored on failure, so a partial load (e.g.
        # model succeeds, then scheduler fails because its class
        # changed) can't leave the model holding old checkpoint weights
        # while everything else silently resets to fresh — that would be
        # an inconsistent, worse-than-either state. Falling back to a
        # fully fresh start is self-healing either way (the stale
        # checkpoint gets overwritten at the end of epoch 0).
        import copy
        fresh_model_state = copy.deepcopy(self.model.state_dict())
        try:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            self.ema.shadow = {k: v.to(self.device) for k, v in ckpt["ema_shadow"].items()}
        except (RuntimeError, KeyError, ValueError) as e:
            self.model.load_state_dict(fresh_model_state)
            print(f"  WARNING: checkpoint at {self.last_checkpoint_path} doesn't "
                  f"match the current run's model/optimizer/scheduler configuration "
                  f"(likely a FeatureFlags or hyperparameter change since it was "
                  f"saved) — starting fold {self.fold + 1} from scratch instead of "
                  f"resuming.\n    {e}")
            return 0

        self.early_stopping.best_score = ckpt["early_stopping_best_score"]
        self.early_stopping.counter = ckpt["early_stopping_counter"]
        self.early_stopping.early_stop = ckpt["early_stopping_early_stop"]
        self.best_acc = ckpt["best_acc"]

        resume_epoch = ckpt["epoch"] + 1
        print(f"  Resuming fold {self.fold + 1} from epoch {resume_epoch + 1} "
              f"(best acc so far: {self.best_acc:.2f}%)")
        return resume_epoch

    def _save_resume_checkpoint(self, epoch):
        """Save full training state so this fold can resume after an
        interruption without losing already-completed epochs.

        Distinct from _save_checkpoint's best-only model weights (which
        inference.py loads for eval) — this includes optimizer/scheduler/
        scaler/EMA/early-stopping state and is overwritten every epoch.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "ema_shadow": self.ema.shadow,
            "early_stopping_best_score": self.early_stopping.best_score,
            "early_stopping_counter": self.early_stopping.counter,
            "early_stopping_early_stop": self.early_stopping.early_stop,
            "best_acc": self.best_acc,
        }
        torch.save(checkpoint, self.last_checkpoint_path)

    def train_epoch(self):
        """Run one training epoch.

        Returns:
            (avg_loss, avg_acc) tuple.
        """
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = self._to_device(batch)

            # Modality dropout: randomly zero out modalities during training
            if self.config.modality_dropout_p > 0:
                batch = self._apply_modality_dropout(batch)

            # Mixup overwrites batch["label"] with a soft one-hot mix, so keep
            # the hard label around for the accuracy metric below.
            hard_label = batch["label"]
            use_mixup = self.config.mixup_alpha > 0 and np.random.random() < 0.5
            if use_mixup:
                batch = self._apply_mixup(batch)

            with autocast(enabled=self.config.mixed_precision):
                output = self.model(batch)
                logits = output["logits"]

                # Primary loss (handles mixup soft labels vs hard labels)
                if use_mixup and batch["label"].dim() > 1:
                    log_probs = F.log_softmax(logits, dim=-1)
                    loss = -(batch["label"] * log_probs).sum(dim=-1).mean()
                else:
                    loss = self.criterion(logits, batch["label"])

                # Auxiliary category loss
                if ("aux_logits" in output and output["aux_logits"] is not None
                        and "category_label" in batch):
                    aux_loss = self.aux_criterion(
                        output["aux_logits"], batch["category_label"])
                    loss = loss + self.config.flags.aux_loss_weight * aux_loss

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update()

            # Metrics
            acc = accuracy(logits.detach(), hard_label, topk=(1,))[0]
            batch_size = hard_label.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)

        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def validate(self):
        """Run validation.

        Returns:
            (avg_loss, avg_acc) tuple.
        """
        self.model.eval()

        # Apply EMA weights for evaluation (CMI standard practice)
        self.ema.save_original()
        self.ema.apply_shadow()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for batch in self.val_loader:
            batch = self._to_device(batch)

            output = self.model(batch)
            logits = output["logits"]
            loss = self.criterion(logits, batch["label"])

            acc = accuracy(logits, batch["label"], topk=(1,))[0]
            batch_size = batch["label"].size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)

        # Restore original weights
        self.ema.restore_original()

        return loss_meter.avg, acc_meter.avg

    def run(self):
        """Run full training for this fold.

        Returns:
            best validation accuracy achieved.
        """
        print(f"\n{'='*60}")
        print(f"  Fold {self.fold + 1}/{self.config.n_folds}")
        print(f"  Train: {len(self.train_dataset)} clips, Val: {len(self.val_dataset)} clips")
        print(f"{'='*60}")
        self.model.log_parameter_breakdown()

        # Resumed from a checkpoint that already finished this fold (either
        # ran out its full epoch budget, or early-stopped) — nothing to do.
        if self.early_stopping.early_stop or self.start_epoch >= self.config.epochs:
            print(f"  Fold {self.fold + 1} already complete. Best Val Acc: {self.best_acc:.2f}%\n")
            return self.best_acc

        for epoch in range(self.start_epoch, self.config.epochs):
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Scheduler step
            self.scheduler.step()

            # Validate
            val_loss, val_acc = self.validate()

            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                f"LR: {lr:.2e} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpoint best
            is_best = self.early_stopping(val_acc)
            if is_best:
                self.best_acc = val_acc
                self._save_checkpoint(epoch, val_acc, is_best=True)
                print(f"  -> New best! Saved to {self.best_model_path}")

            # Full resume state, saved every epoch regardless of is_best, so
            # an interruption never costs more than one epoch of progress.
            self._save_resume_checkpoint(epoch)

            if self.early_stopping.early_stop:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break

        print(f"\n  Fold {self.fold + 1} complete. Best Val Acc: {self.best_acc:.2f}%\n")
        return self.best_acc

    def _to_device(self, batch):
        """Move batch tensors to device."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            elif isinstance(v, dict):
                out[k] = {sk: sv.to(self.device) if isinstance(sv, torch.Tensor)
                          else sv for sk, sv in v.items()}
            else:
                out[k] = v
        return out

    def _apply_modality_dropout(self, batch):
        """Randomly zero-out modalities during training for robustness."""
        p = self.config.modality_dropout_p
        flags = batch["flags"]

        # All modalities: dynamically reshape mask to match tensor ndim
        for key, flag_key in [
            ("imu", "has_imu"), ("radar", "has_radar"), ("skeleton", "has_skeleton"),
            ("depth_color", "has_depth"), ("ir", "has_ir"), ("thermal", "has_thermal"),
        ]:
            if batch[key].numel() == 0 or flags[flag_key].sum() == 0:
                continue
            B = batch[key].shape[0]
            # Create mask with correct number of trailing singleton dims
            mask = (torch.rand(B, device=batch[key].device) > p).float()
            mask = mask.reshape(B, *([1] * (batch[key].ndim - 1)))
            batch[key] = batch[key] * mask
            flags[flag_key] = flags[flag_key] * mask.reshape(B)

        return batch

    def _apply_mixup(self, batch):
        """Apply Mixup augmentation to the batch."""
        alpha = self.config.mixup_alpha
        batch_size = batch["label"].size(0)

        # Generate mixup indices
        lam = np.random.beta(alpha, alpha)
        if lam < 0.5:
            lam = 1.0 - lam
        indices = torch.randperm(batch_size, device=self.device)

        # Mix labels
        y1 = F.one_hot(batch["label"], self.config.num_classes).float()
        y2 = F.one_hot(batch["label"][indices], self.config.num_classes).float()
        batch["label"] = lam * y1 + (1 - lam) * y2

        # Mix time-series modalities
        lam_t = lam
        for key in ["imu", "radar", "skeleton"]:
            if batch[key].numel() > 0:
                batch[key] = lam_t * batch[key] + (1 - lam_t) * batch[key][indices]

        # Mix frame modalities
        for key in ["depth_color", "ir", "thermal"]:
            if batch[key].numel() > 0:
                batch[key] = lam_t * batch[key] + (1 - lam_t) * batch[key][indices]

        return batch

    def _save_checkpoint(self, epoch, acc, is_best=False):
        """Save model checkpoint.

        Saves the EMA shadow weights, not the raw training weights.
        `acc` (the metric that made this "best") was measured in
        validate() against the EMA shadow, not the raw weights — by the
        time validate() returns, it has already called
        ema.restore_original(), so self.model's live parameters are back
        to the noisier raw training weights. Saving those here would
        silently decouple the checkpoint actually used for inference
        (see inference.py, which loads model_state_dict directly) from
        the metric used to select it, defeating the entire point of
        using EMA (CMI standard practice for stable inference).

        Note: state_dict() returns tensors that alias the live
        parameters' storage, not copies — so torch.save() must happen
        while EMA weights are still applied. Restoring raw weights
        before saving (an earlier version of this fix did exactly that)
        would silently overwrite the checkpoint's tensors back to raw
        via restore_original()'s in-place copy_, before the bytes ever
        reach disk, making the "fix" a no-op.
        """
        self.ema.save_original()
        self.ema.apply_shadow()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_acc": self.best_acc,
            "config": self.config,
        }
        torch.save(checkpoint, self.best_model_path)
        self.ema.restore_original()


def run_cross_validation(config=None):
    """Run full cross-validation training.

    Args:
        config: Config object. Uses default if None.

    Returns:
        list of per-fold validation accuracies.
    """
    if config is None:
        config = Config()

    print(f"\n{'#'*60}")
    print(f"  CUHK-X Small Model Track — Cross-Validation Training")
    print(f"  Device: {config.device}")
    print(f"  Folds: {config.n_folds} | Epochs: {config.epochs} | Batch: {config.batch_size}")
    print(f"{'#'*60}")

    # Discover training data
    print("\n[1/3] Discovering training clips...")
    clips, labels = discover_clips(config.train_data)
    clip_list = build_clip_list(clips, labels)
    print(f"  Found {len(clip_list)} clips across {len(set(l[0] for l in clip_list))} users")

    # Create folds
    print("\n[2/3] Creating cross-subject train/val splits...")
    print(f"  Train users: {config.train_users}")
    print(f"  Test users (excluded): {config.test_users}")
    folds = create_folds(clip_list, config, n_folds=config.n_folds, seed=config.seed)
    print(f"  Created {len(folds)} folds from {len(set(item[0] for item in clip_list if item[0] in config.train_users))} train users")

    # Run folds
    print("\n[3/3] Training...")
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        trainer = Trainer(config, fold, train_idx, val_idx, clip_list, clips, labels)
        acc = trainer.run()
        fold_accuracies.append(acc)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Cross-Validation Complete")
    print(f"  Fold Accuracies: {[f'{a:.2f}%' for a in fold_accuracies]}")
    print(f"  Mean: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%")
    print(f"{'='*60}\n")

    return fold_accuracies


if __name__ == "__main__":
    run_cross_validation()
