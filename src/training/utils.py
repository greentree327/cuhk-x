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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


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


def _filter_train_users(clip_list, config):
    """Filter clip_list down to train-users-only, with parallel arrays for
    stratified/grouped splitting. Shared by create_folds() and
    _create_single_split() since both need the same setup.

    Args:
        clip_list: list of (user_id, user, trial, action_id) tuples.
        config: Config object with train_users.

    Returns:
        (train_indices_all, users, labels, indices):
            train_indices_all: original clip_list positions for train users.
            users: array of user_id per train clip (grouping key).
            labels: array of action_id per train clip (stratification key).
            indices: np.arange(len(train_indices_all)), for sklearn's split().
    """
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

    train_clips_subset = [clip_list[i] for i in train_indices_all]
    users = np.array([item[0] for item in train_clips_subset])
    labels = np.array([item[3] for item in train_clips_subset])
    indices = np.arange(len(train_clips_subset))
    return train_indices_all, users, labels, indices


def _create_single_split(clip_list, config, val_size=0.2, seed=42):
    """Single stratified-by-user train/val split (not k-fold).

    For quick iteration/debugging — confirming the pipeline runs, or
    getting a real per-epoch timing number — where a full k-fold sweep
    isn't needed. Trains exactly one model instead of n_folds models.

    Not stratified by class label (GroupShuffleSplit only groups by user,
    unlike StratifiedGroupKFold), so with this dataset's 27.5:1 class
    imbalance the rarest classes may be thin or absent from the validation
    split — acceptable for a sanity check, not a substitute for the real
    k-fold run.

    Args:
        clip_list: list of (user_id, user, trial, action_id) tuples.
        config: Config object with train_users.
        val_size: fraction of train users held out for validation. 0.2
            matches the ~20% validation proportion a 5-fold run gives per
            fold, for a roughly comparable split size.
        seed: random seed.

    Returns:
        list containing exactly one (train_indices, val_indices) tuple,
        matching create_folds()'s return shape.
    """
    train_indices_all, users, labels, indices = _filter_train_users(clip_list, config)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_sub_idx, val_sub_idx = next(gss.split(indices, groups=users))

    train_idx_orig = [train_indices_all[i] for i in train_sub_idx]
    val_idx_orig = [train_indices_all[i] for i in val_sub_idx]
    return [(train_idx_orig, val_idx_orig)]


def create_folds(clip_list, config, n_folds=5, seed=42):
    """Create StratifiedGroupKFold splits for validation WITHIN train users.

    Note: The competition specifies a hardcoded cross-subject split:
    Train: users 1-9, 16-24 | Test: users 10-11, 25-26.
    This function creates internal validation folds from the TRAIN users only.
    The test users are NEVER used during training.

    n_folds == 1 is handled as a single train/val split (see
    _create_single_split) rather than k-fold CV, which is mathematically
    undefined for k=1 (StratifiedGroupKFold raises ValueError for
    n_splits=1 — there's no held-out fold left over).

    Args:
        clip_list: list of (user_id, user, trial, action_id) tuples.
        config: Config object with train_users/test_users.
        n_folds: number of internal validation folds. 1 = single split.
        seed: random seed.

    Returns:
        list of (train_indices, val_indices) per fold (using train users only).
    """
    if n_folds == 1:
        return _create_single_split(clip_list, config, seed=seed)

    train_indices_all, users, labels, indices = _filter_train_users(clip_list, config)

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds_mapped = []
    for train_sub_idx, val_sub_idx in sgkf.split(indices, labels, groups=users):
        # Map back to original clip_list indices
        train_idx_orig = [train_indices_all[i] for i in train_sub_idx]
        val_idx_orig = [train_indices_all[i] for i in val_sub_idx]
        folds_mapped.append((train_idx_orig, val_idx_orig))

    return folds_mapped
