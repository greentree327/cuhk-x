"""
# CONVENTION: primary — Propagating-exception convention.

Fetches the CUHK-X Small Model Track dataset from Hugging Face and lays it
out exactly where config.py's hardcoded paths expect it, so a fresh
`git clone` + one env var is enough to run the pipeline end to end (e.g. in
a Colab session where the ~50GB dataset can't live in the git repo itself).

Requires: `pip install huggingface_hub` and an HF_TOKEN environment variable
(the dataset repo is gated — accept its access conditions on the HF page
first, then create a read token at https://huggingface.co/settings/tokens).

This module is deliberately defensive rather than hardcoding exact filenames:
the HF repo is gated, so its precise file listing (split-zip volume names,
internal folder nesting) could not be verified end-to-end before writing
this. Every step is auto-discovered at runtime and logged, so a mismatch
surfaces as a clear diagnostic instead of a silent wrong path. If a fresh
Colab run hits an error here, the printed diagnostics (repo file listing,
extracted top-level folders) are what to paste back for a fix.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ID = "Kevin-Pal/CUHK-X_Small_Model_Track"
EXPECTED_MODALITIES = {"Depth_Color", "IR", "Thermal", "IMU", "Radar", "Skeleton"}
MAX_WALK_DEPTH = 6  # upper-bound guard for the directory auto-detection scan


def is_dataset_ready(config):
    """Check whether both train and test data are already extracted in place.

    Args:
        config: Config object with train_data / test_data paths.

    Returns:
        True if both directories exist and contain at least one modality
        folder / clip folder respectively.
    """
    train_ok = (
        config.train_data.is_dir()
        and any(p.name in EXPECTED_MODALITIES for p in config.train_data.iterdir())
    )
    test_ok = (
        config.test_data.is_dir()
        and any(p.name.startswith("SM_") for p in config.test_data.iterdir())
    )
    return train_ok and test_ok


def ensure_dataset_available(config, raw_dir=None):
    """Download + extract the dataset from Hugging Face if not already present.

    Idempotent: if is_dataset_ready(config) is already True, does nothing.

    Args:
        config: Config object (uses config.train_data / config.test_data /
            config.data_root to know where things must end up).
        raw_dir: staging directory for the raw HF download. Defaults to
            a `_hf_raw` folder next to the dataset root.

    Raises:
        RuntimeError: if HF_TOKEN is not set, or if auto-detection can't
            find the expected folder structure after extraction.
    """
    if is_dataset_ready(config):
        print("[data_bootstrap] Dataset already present, skipping download.")
        return

    if raw_dir is None:
        raw_dir = config.data_root.parent / "_hf_raw"
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. This dataset is gated: "
            f"visit https://huggingface.co/datasets/{REPO_ID}, click "
            "'Agree and access repository', then create a read token at "
            "https://huggingface.co/settings/tokens and set it as HF_TOKEN "
            "before running this again."
        )

    _download_repo(raw_dir, token)

    zip_groups = _find_zip_groups(raw_dir)
    print(f"[data_bootstrap] Found archive groups: "
          f"{ {k: [p.name for p in v] for k, v in zip_groups.items()} }")

    # Process smallest archive group(s) first so their disk usage is freed
    # before committing to the largest group's reassemble+extract peak
    # (zip + its extracted output coexisting is the single biggest moment
    # of disk pressure — better to hit it with nothing else outstanding).
    ordered_groups = sorted(
        zip_groups.items(),
        key=lambda kv: sum(p.stat().st_size for p in kv[1]),
    )

    for base_name, parts in ordered_groups:
        full_zip = _reassemble_if_split(raw_dir, base_name, parts)
        extract_scratch = raw_dir / f"_extract_{base_name}"
        extract_scratch.mkdir(exist_ok=True)
        print(f"[data_bootstrap] Extracting {full_zip.name} -> {extract_scratch}")
        _extract_zip(full_zip, extract_scratch)

        # Free the archive immediately after extraction — a ~40GB zip sitting
        # next to its own ~40GB extracted output is the other half of the
        # disk-bloat problem (the first half being the split parts, already
        # freed in _reassemble_if_split).
        full_zip.unlink()
        print(f"[data_bootstrap] Freed {full_zip.name} after extraction")

        _place_extracted(extract_scratch, config)

        # Whatever's left is either empty wrapper dirs (the real content was
        # moved out by _place_extracted) or something detection missed —
        # either way it's not needed once is_dataset_ready() is re-checked
        # below, so don't let it linger and eat disk across retries.
        shutil.rmtree(extract_scratch, ignore_errors=True)

    if not is_dataset_ready(config):
        raise RuntimeError(
            "Extraction finished but the expected folder structure was not "
            "found. train_data=" + str(config.train_data) +
            " test_data=" + str(config.test_data) +
            ". Check the [data_bootstrap] log lines above for what was "
            "actually downloaded/extracted and adjust _place_extracted()."
        )

    # Nothing under raw_dir is needed anymore — the actual data now lives at
    # config.train_data / config.test_data. Reclaim the rest (leftover HF
    # metadata files, .gitattributes, etc.) rather than leaving it to rot.
    shutil.rmtree(raw_dir, ignore_errors=True)
    print("[data_bootstrap] Dataset ready.")


def _download_repo(raw_dir, token):
    """Pull every file in the HF dataset repo into raw_dir."""
    from huggingface_hub import snapshot_download

    print(f"[data_bootstrap] Downloading {REPO_ID} from Hugging Face "
          f"into {raw_dir} ...")
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", token=token,
        local_dir=str(raw_dir),
    )
    listing = sorted(p.name for p in raw_dir.rglob("*") if p.is_file())
    print(f"[data_bootstrap] Downloaded {len(listing)} files: {listing}")


def _ensure_zip_cli():
    """Make sure the `zip` CLI is available (needed to reassemble split
    archives exactly the way the dataset's own instructions specify:
    `zip -s 0 base.zip --out full.zip`)."""
    if shutil.which("zip"):
        return
    print("[data_bootstrap] 'zip' CLI not found, attempting `apt-get install "
          "zip` (Colab/Debian only)...")
    try:
        subprocess.run(["apt-get", "install", "-y", "-qq", "zip"], check=True)
    except Exception as e:
        raise RuntimeError(
            "`zip` CLI is required to reassemble the split archive and could "
            f"not be installed automatically ({e}). Install it manually "
            "(e.g. `apt-get install zip` in Colab) and re-run."
        )


def _find_zip_groups(raw_dir):
    """Group downloaded files by archive base name.

    Handles both a plain single .zip file and Info-ZIP style split archives
    (base.zip + base.z01, base.z02, ...). Grouping is done dynamically from
    whatever was actually downloaded, rather than assuming fixed filenames,
    since the exact split count wasn't verifiable ahead of time (gated repo).

    Returns:
        dict of base_name -> sorted list of Path objects for that archive's
        parts (split parts first in volume order, main .zip last).
    """
    all_zip_like = [
        p for p in raw_dir.rglob("*")
        if p.is_file() and re.search(r"\.zip$|\.z\d\d$", p.name, re.IGNORECASE)
    ]
    groups = {}
    for p in all_zip_like:
        base = re.sub(r"\.zip$|\.z\d\d$", "", p.name, flags=re.IGNORECASE)
        groups.setdefault(base, []).append(p)

    for base, parts in groups.items():
        # Split volumes (.z01, .z02, ...) sort before the main .zip, which
        # must be concatenated last (it holds the central directory record).
        parts.sort(key=lambda p: (p.suffix.lower() == ".zip", p.name))
    return groups


def _reassemble_if_split(raw_dir, base_name, parts):
    """Reassemble a split archive into one file, or return it unchanged.

    Uses the `zip -s 0` CLI (matches the dataset's documented instructions)
    rather than a hand-rolled binary concatenation, since a subtle mistake
    in re-implementing the split-zip format would corrupt the archive.
    """
    if len(parts) == 1:
        return parts[0]

    _ensure_zip_cli()
    main_zip = next(p for p in parts if p.suffix.lower() == ".zip")
    full_zip = raw_dir / f"{base_name}_full.zip"
    if full_zip.exists():
        print(f"[data_bootstrap] {full_zip.name} already assembled, skipping.")
        return full_zip

    print(f"[data_bootstrap] Reassembling {len(parts)} volumes for "
          f"{base_name} via `zip -s 0` ...")
    subprocess.run(
        ["zip", "-s", "0", str(main_zip), "--out", str(full_zip)],
        check=True, cwd=str(raw_dir),
    )

    # Free the split volumes immediately — with a ~40GB archive, keeping
    # both the parts and the reassembled copy around at once is exactly
    # what fills a Colab disk (parts + full_zip + extracted output would
    # otherwise all coexist).
    for part in parts:
        part.unlink()
    print(f"[data_bootstrap] Freed {len(parts)} split volumes after reassembly")
    return full_zip


def _extract_zip(zip_path, dest_dir):
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _place_extracted(extract_scratch, config):
    """Find the modality-folder root (or SM_* clip root) inside whatever was
    just extracted, and move it to the exact path config.py expects.

    Searches breadth-first up to MAX_WALK_DEPTH so it works whether the zip's
    internal layout is `data/<modality>/...` or has extra wrapper folders.
    """
    train_root = _find_dir_with_children(
        extract_scratch, lambda names: any(n in EXPECTED_MODALITIES for n in names)
    )
    if train_root is not None and not config.train_data.exists():
        print(f"[data_bootstrap] Found training data root at {train_root}, "
              f"moving to {config.train_data}")
        config.train_data.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(train_root), str(config.train_data))

    test_root = _find_dir_with_children(
        extract_scratch, lambda names: sum(n.startswith("SM_") for n in names) > 3
    )
    if test_root is not None and not config.test_data.exists():
        print(f"[data_bootstrap] Found test data root at {test_root}, "
              f"moving to {config.test_data}")
        config.test_data.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(test_root), str(config.test_data))

    if train_root is None and test_root is None:
        top_level = [p.name for p in extract_scratch.iterdir()]
        print(f"[data_bootstrap] WARNING: could not identify train or test "
              f"root under {extract_scratch}. Top-level entries: {top_level}")


def _find_dir_with_children(root, predicate, depth=0):
    """BFS for the first directory (at or under root) whose child names
    satisfy predicate(names)."""
    if depth > MAX_WALK_DEPTH:
        return None
    try:
        children = list(root.iterdir())
    except (PermissionError, FileNotFoundError):
        return None

    child_names = [c.name for c in children if c.is_dir()]
    if predicate(child_names):
        return root

    for child in children:
        if child.is_dir():
            found = _find_dir_with_children(child, predicate, depth + 1)
            if found is not None:
                return found
    return None


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import Config
    ensure_dataset_available(Config())
