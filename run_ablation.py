"""
# CONVENTION: primary — Propagating-exception convention

Runs 3 ablation configs (minimal / baseline / synthesized), each with 5-fold CV,
and generates 3 submission CSVs. Writes a timestamped log file to output/run.log.

Usage:
    python run_ablation.py

Monitor:
    - Terminal: epoch-level output
    - Log file:   output/run.log (written in real time)
      On PowerShell:  Get-Content output\run.log -Wait
    - Checkpoints:   output/<config>/fold_*/best_model.pth
    - Submissions:   output/<config>/submission.csv
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.data_bootstrap import ensure_dataset_available


def _env_str(name, default):
    """os.environ.get(name, default), but also falls back to default when
    the variable is set to an empty string. IPython's `%env VAR=` sets VAR
    to "" rather than actually unsetting it, so treating "" as "not set"
    here is what makes `%env CUHKX_FOO=` behave like a real reset."""
    val = os.environ.get(name, "")
    return val if val else default


def _env_int(name, default):
    """Same empty-string-means-unset handling as _env_str, parsed as int."""
    val = _env_str(name, "")
    return int(val) if val else default


def _env_bool(name, default):
    """Same empty-string-means-unset handling as _env_str, parsed as a
    0/1 flag (e.g. CUHKX_CROSS_MODAL_ATTENTION=1)."""
    val = _env_str(name, "")
    return bool(int(val)) if val else default


# Overridable so checkpoints/logs can be redirected to persistent storage
# (e.g. a Google Drive mount in Colab) without moving the code itself there.
# Defaults to "output" (unchanged local behavior) when unset.
OUTPUT_ROOT = Path(_env_str("CUHKX_OUTPUT_ROOT", "output"))
LOG_PATH = OUTPUT_ROOT / "run.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(msg, end="\n"):
    """Print AND append to log file (real-time visible)."""
    print(msg, end=end, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + end)


log(f"{'='*70}")
log(f"  CUHK-X Ablation Run — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"{'='*70}")


def run_config(label, cfg):
    """Train full CV + generate submission."""
    import torch, numpy as np
    from src.data.dataset import discover_clips, build_clip_list
    from src.training.trainer import Trainer
    from src.training.utils import create_folds
    from src.inference import generate_submission

    log(f"\n{'#'*70}")
    log(f"  CONFIG: {label}")
    log(f"  synth={cfg.flags.use_synthesized_features} mask={cfg.flags.use_skeleton_attention_mask} "
        f"crop={cfg.flags.use_spatial_crop} erase={cfg.flags.use_random_erase}")
    log(f"  aux={cfg.flags.use_aux_category_loss} cls_wt={cfg.flags.use_class_weights}")
    log(f"  seg_pool={cfg.flags.use_segment_pooling} cross_attn={cfg.flags.use_cross_modal_attention} "
        f"hflip={cfg.flags.use_handedness_flip}")
    log(f"  imu_dim={cfg.imu_input_dim} skel_dim={cfg.skel_input_dim} "
        f"epochs={cfg.epochs} folds={cfg.n_folds} batch={cfg.batch_size}")
    log(f"{'#'*70}")

    # Discover
    log("\n[1/4] Discovering data...")
    clips, labels = discover_clips(cfg.train_data)
    clip_list = build_clip_list(clips, labels)
    log(f"  {len(clip_list)} clips, {len(set(c[0] for c in clip_list if c[0]))} users")

    # Folds
    log("\n[2/4] CV splits...")
    folds = create_folds(clip_list, cfg, n_folds=cfg.n_folds, seed=cfg.seed)
    log(f"  {len(folds)} folds")

    # Train
    log(f"\n[3/4] Training ({cfg.n_folds} folds × ~{cfg.epochs} epochs)...")
    fold_accs = []
    t0 = time.time()
    for fold, (tr, vl) in enumerate(folds):
        log(f"\n--- Fold {fold+1}/{cfg.n_folds} ---")
        t = Trainer(cfg, fold, tr, vl, clip_list, clips, labels)
        acc = t.run()
        fold_accs.append(acc)
        log(f"Fold {fold+1} best: {acc:.2f}%")
        el = time.time() - t0
        eta = el / (fold + 1) * (cfg.n_folds - fold - 1)
        log(f"  Elapsed: {el/60:.0f}m  ETA: {eta/60:.0f}m  "
            f"GPU mem: {torch.cuda.memory_reserved()/1024**3:.1f}GB" if torch.cuda.is_available() else "")

    mu, sd = np.mean(fold_accs), np.std(fold_accs)
    log(f"\n  CV: {[f'{a:.1f}%' for a in fold_accs]}")
    log(f"  Mean: {mu:.2f}% ± {sd:.2f}%")

    # Submission
    log(f"\n[4/4] Generating submission...")
    ckpts = sorted(cfg.output_dir.glob("fold_*/best_model.pth"))
    if not ckpts:
        log(f"  WARNING: no checkpoints, skipping")
        return mu
    sub = generate_submission(cfg, cfg.output_dir)
    log(f"  Saved: {sub}")
    return mu


def main():
    # Fetch the dataset from Hugging Face if it isn't already extracted in
    # place (no-op locally where it already exists; required in a fresh
    # Colab clone — see download_data.py for the one-time HF_TOKEN setup).
    log("\n[0/4] Checking dataset availability...")
    ensure_dataset_available(Config())

    # Configs
    # batch_size=64 OOMs on an 8GB GPU once all 6 modality encoders run together
    # (measured peak ~10.4GB reserved at batch_size=32, already over physical
    # VRAM and only surviving via WDDM's memory oversubscription — it would
    # eventually OOM over a full 60-epoch run as fragmentation grows).
    # batch_size=16 leaves a safe margin (~5.3GB peak reserved).
    #
    # epochs/n_folds are overridable via env vars so a quick timing/sanity
    # run (e.g. CUHKX_EPOCHS=10 CUHKX_N_FOLDS=1) doesn't require editing
    # this file and remembering to revert it before the real run. Treats an
    # empty string the same as unset — IPython's `%env VAR=` sets VAR to ""
    # rather than actually removing it, and os.environ.get()'s default only
    # applies when the key is absent, not when it's present-but-empty.
    base = dict(
        epochs=_env_int("CUHKX_EPOCHS", 60),
        n_folds=_env_int("CUHKX_N_FOLDS", 5),
        batch_size=16, num_workers=2, mixed_precision=True,
    )

    c1 = Config()
    c1.output_dir = OUTPUT_ROOT / "minimal"
    for a in ['use_synthesized_features','use_skeleton_attention_mask',
              'use_spatial_crop','use_random_erase','use_aux_category_loss',
              'use_class_weights']:
        setattr(c1.flags, a, False)
    for k, v in base.items(): setattr(c1, k, v)

    c2 = Config()
    c2.output_dir = OUTPUT_ROOT / "baseline"
    for k, v in base.items(): setattr(c2, k, v)

    c3 = Config()
    c3.flags.use_synthesized_features = True
    c3.output_dir = OUTPUT_ROOT / "synthesized"
    for k, v in base.items(): setattr(c3, k, v)

    # Architecture/augmentation flags default to whatever each config above
    # already set (usually the FeatureFlags dataclass default), but are
    # overridable per-run via env vars — same empty-string-means-unset
    # convention as epochs/n_folds — so a Colab run can flip them on
    # without editing this file (e.g. CUHKX_CROSS_MODAL_ATTENTION=1).
    for cfg in (c1, c2, c3):
        cfg.flags.use_segment_pooling = _env_bool(
            "CUHKX_SEGMENT_POOLING", cfg.flags.use_segment_pooling)
        cfg.flags.use_cross_modal_attention = _env_bool(
            "CUHKX_CROSS_MODAL_ATTENTION", cfg.flags.use_cross_modal_attention)
        cfg.flags.use_handedness_flip = _env_bool(
            "CUHKX_HANDEDNESS_FLIP", cfg.flags.use_handedness_flip)

    # Which configs to run is also overridable, so a calibration run can
    # target just one (e.g. CUHKX_CONFIGS=minimal) instead of all three.
    all_configs = [("minimal", c1), ("baseline", c2), ("synthesized", c3)]
    requested = _env_str("CUHKX_CONFIGS", "")
    if requested:
        wanted = {name.strip() for name in requested.split(",")}
        all_configs = [(label, cfg) for label, cfg in all_configs if label in wanted]

    results = {}
    t_total = time.time()
    for label, cfg in all_configs:
        try:
            results[label] = run_config(label, cfg)
        except Exception as e:
            log(f"\n  ERROR in {label}: {e}")
            import traceback; log(traceback.format_exc())
            results[label] = None

    el = time.time() - t_total
    log(f"\n{'='*70}")
    log(f"  SUMMARY — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — {el/60:.0f} min total")
    log(f"{'='*70}")
    for l, a in results.items():
        sub = OUTPUT_ROOT / l / "submission.csv"
        log(f"  {l:15s}  {'FAILED' if a is None else f'{a:.2f}%':>8s}  {sub if sub.exists() else 'N/A'}")
    log(f"{'='*70}")
    log(f"\nFull log: {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()
