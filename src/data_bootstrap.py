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

The training archive ships as a split zip (base.zip + base.z01, z02, ...).
A genuine Info-ZIP split archive's ZIP64 locator record has "disk number"
fields baked into its bytes declaring it multi-volume — Python's zipfile
hard-rejects any archive with that flag set (BadZipFile: "zipfiles that
span multiple disks are not supported"), regardless of whether the bytes
are read from one physical file or several. There is no way to satisfy
zipfile without actually clearing that metadata, which is what the
documented `zip -s 0 --out` reassembly does.

An earlier version of this module tried a cheaper path first (unzip
reading directly across the split volumes with no merge step), but this
was confirmed against the real dataset to desync partway through and
silently produce garbage for every file after that point ("bad zipfile
offset" errors starting mid-archive) rather than a clean failure — not
safe to attempt even as a best-effort try. Reassembly via `zip -s 0` is
always used for multi-part archives; it does need ~2x the archive's size
in free disk space for that one step, which is unavoidable for a correct
multi-volume merge.
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
    # before committing to the largest group's extraction peak (archive
    # parts + extracted output coexisting is the single biggest moment of
    # disk pressure — better to hit it with nothing else outstanding).
    ordered_groups = sorted(
        zip_groups.items(),
        key=lambda kv: sum(p.stat().st_size for p in kv[1]),
    )

    for base_name, parts in ordered_groups:
        extract_scratch = raw_dir / f"_extract_{base_name}"
        extract_scratch.mkdir(exist_ok=True)
        print(f"[data_bootstrap] Extracting {base_name} "
              f"({len(parts)} part(s)) -> {extract_scratch}")

        if len(parts) == 1:
            _extract_zip(parts[0], extract_scratch)
            parts[0].unlink()
        else:
            _extract_multi_part(parts, base_name, raw_dir, extract_scratch)
        print(f"[data_bootstrap] Freed archive file(s) for {base_name}")

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
        # must come last in the concatenated stream (it holds the central
        # directory record, which a zip reader looks for at the very end).
        parts.sort(key=lambda p: (p.suffix.lower() == ".zip", p.name))
    return groups


def _extract_multi_part(parts, base_name, raw_dir, dest_dir):
    """Extract a split archive (base.zip + base.z01, z02, ...) into dest_dir.

    Confirmed against the real dataset: unzip reading directly across split
    volumes (no separate merge step) desyncs partway through and silently
    produces garbage for every file after that point ("bad zipfile offset"
    errors starting mid-archive) — not a safe thing to attempt even as a
    best-effort first try, since a corrupted-but-"successful"-looking
    extraction is worse than a clean failure. Always physically reassembles
    via the documented `zip -s 0` method, which does need ~2x the archive's
    size in free disk space for this one step — unavoidable for a correct
    multi-volume merge.
    """
    main_zip = next(p for p in parts if p.suffix.lower() == ".zip")

    print(f"[data_bootstrap] {base_name}: reassembling via `zip -s 0` "
          f"(needs ~2x this archive's size in free disk space for this "
          f"one step)...")
    _ensure_zip_cli()
    full_zip = raw_dir / f"{base_name}_full.zip"
    subprocess.run(
        ["zip", "-s", "0", str(main_zip), "--out", str(full_zip)],
        check=True, cwd=str(raw_dir),
    )
    for part in parts:
        part.unlink()
    _extract_zip(full_zip, dest_dir)
    full_zip.unlink()


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


def _extract_zip(zip_path, dest_dir):
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def _place_extracted(extract_scratch, config):
    """Find the modality-folder root (or SM_* clip root) inside whatever was
    just extracted, and move it to the exact path config.py expects.

    Searches depth-first up to MAX_WALK_DEPTH so it works whether the zip's
    internal layout is `data/<modality>/...` or has extra wrapper folders.

    Each individual test clip folder (SM_test_XXXX) *also* has modality-named
    children (its own IR/Depth_Color/... subfolders) — structurally
    indistinguishable from the aggregate train root by child names alone.
    The train predicate below requires >=2 modality matches (a lone clip
    folder inside a train-shaped tree is implausible) AND excludes any
    candidate directory whose own name starts with "SM_" (which a genuine
    train root, sitting above all actions/users/trials, never does).
    """
    train_root = _find_dir_with_children(
        extract_scratch,
        lambda d, names: (
            not d.name.startswith("SM_")
            and sum(n in EXPECTED_MODALITIES for n in names) >= 2
        ),
    )
    if train_root is not None and not config.train_data.exists():
        print(f"[data_bootstrap] Found training data root at {train_root}, "
              f"moving to {config.train_data}")
        config.train_data.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(train_root), str(config.train_data))

    test_root = _find_dir_with_children(
        extract_scratch, lambda d, names: sum(n.startswith("SM_") for n in names) > 3
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
    """Depth-first search for the first directory (at or under root) whose
    own path and child names satisfy predicate(dir_path, child_names)."""
    if depth > MAX_WALK_DEPTH:
        return None
    try:
        children = list(root.iterdir())
    except (PermissionError, FileNotFoundError):
        return None

    child_names = [c.name for c in children if c.is_dir()]
    if predicate(root, child_names):
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
