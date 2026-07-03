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

        # Scheduler: CosineAnnealingWarmRestarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=config.lr_min
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stop_patience, mode="max"
        )

        # Output paths
        self.fold_dir = config.output_dir / f"fold_{fold}"
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.fold_dir / "best_model.pth"

        # Metrics
        self.best_acc = 0.0

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
        print(f"  Params: {self.model.get_parameter_count():.1f}M")
        print(f"{'='*60}\n")

        for epoch in range(self.config.epochs):
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
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_acc": self.best_acc,
            "config": self.config,
        }
        torch.save(checkpoint, self.best_model_path)


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
