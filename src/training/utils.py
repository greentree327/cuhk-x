"""
# CONVENTION: primary — Propagating-exception convention.

Training utilities:
- EMA (Exponential Moving Average) wrapper
- StratifiedGroupKFold setup
- Early stopping
- Metric tracking
"""
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold


class ModelEMA:
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of model parameters, updated with EMA each step.
    Used universally by CMI top solutions for stable inference.

    Args:
        model: nn.Module.
        decay: EMA decay rate (default 0.999).
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights after an optimizer step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average

    def apply_shadow(self):
        """Copy shadow weights into the model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model weights (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])
        # Actually this is the same as apply_shadow — the pattern is:
        # save original → apply shadow → eval → restore original
        pass

    def save_original(self):
        """Save current model weights before applying EMA shadow."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()

    def restore_original(self):
        """Restore original model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])


class EarlyStopping:
    """Early stopping tracker.

    Args:
        patience: number of epochs to wait before stopping.
        mode: 'max' for accuracy, 'min' for loss.
        min_delta: minimum improvement threshold.
    """

    def __init__(self, patience=20, mode="max", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        """Check if training should stop.

        Args:
            score: current validation metric.

        Returns:
            True if this is a new best.
        """
        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == "max":
            improvement = score - self.best_score
        else:
            improvement = self.best_score - score

        if improvement > self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class AverageMeter:
    """Running average tracker."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy.

    Args:
        output: (batch, num_classes) logits.
        target: (batch,) integer labels.
        topk: tuple of k values.

    Returns:
        list of accuracy percentages.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def create_folds(clip_list, config, n_folds=5, seed=42):
    """Create StratifiedGroupKFold splits for validation WITHIN train users.

    Note: The competition specifies a hardcoded cross-subject split:
    Train: users 1-9, 16-24 | Test: users 10-11, 25-26.
    This function creates internal validation folds from the TRAIN users only.
    The test users are NEVER used during training.

    Args:
        clip_list: list of (user_id, user, trial, action_id) tuples.
        config: Config object with train_users/test_users.
        n_folds: number of internal validation folds.
        seed: random seed.

    Returns:
        list of (train_indices, val_indices) per fold (using train users only).
    """
    # Filter to only training users
    train_users_set = set(config.train_users)
    train_indices_all = [
        i for i, item in enumerate(clip_list)
        if item[0] is not None and item[0] in train_users_set
    ]

    if not train_indices_all:
        raise ValueError(
            "No training clips found for configured train_users: "
            f"{config.train_users}"
        )

    # Extract labels and user IDs for StratifiedGroupKFold
    train_clips_subset = [clip_list[i] for i in train_indices_all]
    users = np.array([item[0] for item in train_clips_subset])
    labels = np.array([item[3] for item in train_clips_subset])
    indices = np.arange(len(train_clips_subset))

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds_mapped = []
    for train_sub_idx, val_sub_idx in sgkf.split(indices, labels, groups=users):
        # Map back to original clip_list indices
        train_idx_orig = [train_indices_all[i] for i in train_sub_idx]
        val_idx_orig = [train_indices_all[i] for i in val_sub_idx]
        folds_mapped.append((train_idx_orig, val_idx_orig))

    return folds_mapped
