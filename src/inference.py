"""
# CONVENTION: primary — Propagating-exception convention.

Inference pipeline for the CUHK-X Small Model Track.

- Loads trained model checkpoints from cross-validation folds.
- Runs test-time augmentation (TTA): multiple temporal crops → average.
- Handles missing modalities via learned tokens.
- Generates submission.csv in competition format.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config
from data.dataset import HARDataset, discover_clips, build_clip_list
from data.collate import collate_fn
from models import HARModel


class InferencePipeline:
    """Generates predictions for test clips.

    Args:
        config: Config object.
        checkpoint_paths: list of paths to model checkpoints (one per fold).
    """

    MAX_TTA_CROPS = 20  # upper-bound for TTA iterations

    def __init__(self, config, checkpoint_paths):
        self.config = config
        self.checkpoint_paths = checkpoint_paths
        self.device = torch.device(config.device)
        self.models = []

        # Load models
        for ckpt_path in checkpoint_paths:
            model = HARModel(config).to(self.device)
            # weights_only=True (the default since PyTorch 2.6) rejects the
            # checkpoint's "config" entry — a Config/FeatureFlags dataclass,
            # not a plain tensor. Safe to disable here: these checkpoints
            # are only ever produced by our own Trainer._save_checkpoint(),
            # never loaded from an untrusted external source.
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.models.append(model)

        print(f"Loaded {len(self.models)} models for ensemble")

    @torch.no_grad()
    def predict(self, test_loader):
        """Run ensemble prediction.

        Args:
            test_loader: DataLoader for test set.

        Returns:
            predictions: (N,) array of predicted class indices.
        """
        all_logits = []

        for batch in test_loader:
            batch = self._to_device(batch)
            batch_logits = []

            for model in self.models:
                output = model(batch)
                logits = output["logits"]

                # TTA: multiple forward passes with different settings
                if self.config.tta and self.config.tta_crops > 1:
                    tta_logits = [logits]
                    # Simple TTA: add small noise perturbations
                    for _ in range(self.config.tta_crops - 1):
                        noisy_batch = self._add_tta_noise(batch)
                        tta_out = model(noisy_batch)
                        tta_logits.append(tta_out["logits"])
                    logits = torch.stack(tta_logits).mean(dim=0)

                batch_logits.append(logits)

            # Average across models
            avg_logits = torch.stack(batch_logits).mean(dim=0)
            all_logits.append(avg_logits.cpu())

        # Concatenate all batches
        all_logits = torch.cat(all_logits, dim=0)
        predictions = all_logits.argmax(dim=-1).numpy()
        return predictions

    def _to_device(self, batch):
        """Move batch to device."""
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

    def _add_tta_noise(self, batch):
        """Add small Gaussian noise for TTA perturbation."""
        noisy = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                noisy[k] = v + torch.randn_like(v) * 0.005
            elif isinstance(v, dict):
                noisy[k] = {}
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor) and sv.is_floating_point():
                        noisy[k][sk] = sv + torch.randn_like(sv) * 0.005
                    else:
                        noisy[k][sk] = sv
            else:
                noisy[k] = v
        return noisy


def discover_test_clips(test_root):
    """Build clip inventory for test data.

    Returns dict mapping (test_id, test_id) → {modality: file_list}
    to match HARDataset's key format (user, trial).
    """
    from collections import defaultdict
    clips = defaultdict(lambda: defaultdict(list))

    for clip_dir in sorted(test_root.iterdir()):
        if not clip_dir.is_dir() or not clip_dir.name.startswith("SM_"):
            continue
        test_id = clip_dir.name

        for mod_dir in sorted(clip_dir.iterdir()):
            if not mod_dir.is_dir():
                continue
            modality = mod_dir.name
            files = sorted([str(f) for f in mod_dir.rglob("*") if f.is_file()])
            if files:
                clips[(test_id, test_id)][modality] = files

    return dict(clips)


def generate_submission(config=None, checkpoint_dir=None):
    """Generate submission.csv from trained models.

    Args:
        config: Config object.
        checkpoint_dir: directory containing fold_*/best_model.pth files.

    Returns:
        Path to generated submission.csv.
    """
    if config is None:
        config = Config()

    if checkpoint_dir is None:
        checkpoint_dir = config.output_dir

    checkpoint_dir = Path(checkpoint_dir)

    # Find checkpoint files
    checkpoint_paths = sorted(checkpoint_dir.glob("fold_*/best_model.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoint files found in {checkpoint_dir}/fold_*/best_model.pth"
        )
    print(f"Found {len(checkpoint_paths)} checkpoints")

    # Load test CSV
    test_df = pd.read_csv(config.test_csv)
    print(f"Test samples: {len(test_df)}")

    # Discover test data (flat structure: test_id/modality/files)
    print("Discovering test clips...")
    test_clips = discover_test_clips(config.test_data)

    # Build test clip list
    test_clip_list = []
    for _, row in test_df.iterrows():
        test_id = Path(row["path"]).name  # SM_test_0001
        test_clip_list.append((None, test_id, test_id, -1))  # user_id, user, trial, label

    # Build dataset and loader
    test_dataset = HARDataset(test_clips, {}, test_clip_list, config, is_train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # Run inference
    pipeline = InferencePipeline(config, checkpoint_paths)
    predictions = pipeline.predict(test_loader)

    # Generate submission
    submission = pd.DataFrame({
        "path": test_df["path"],
        "prediction": predictions,
    })

    submission_path = Path(checkpoint_dir) / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Prediction distribution:\n{submission['prediction'].value_counts().sort_index()}")

    return submission_path


if __name__ == "__main__":
    generate_submission()
