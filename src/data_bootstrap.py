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
Rather than physically reassembling that into one merged file — which
needs the original parts AND the merged copy on disk at once, roughly 2x
the archive's size — _ChainedZipParts presents the parts as one continuous
seekable stream, so zipfile can read straight across the split boundaries
without ever materializing a merged copy.
"""
import io
import os
import re
import shutil
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
        archive = _open_archive(parts)
        extract_scratch = raw_dir / f"_extract_{base_name}"
        extract_scratch.mkdir(exist_ok=True)
        print(f"[data_bootstrap] Extracting {base_name} "
              f"({len(parts)} part(s)) -> {extract_scratch}")
        _extract_zip(archive, extract_scratch)
        if hasattr(archive, "close"):
            archive.close()

        # Free the archive parts immediately after extraction — this is the
        # only copy of this data other than the extracted output, so there's
        # no reason to keep it once extraction succeeds.
        for part in parts:
            part.unlink()
        print(f"[data_bootstrap] Freed {len(parts)} archive file(s) for {base_name}")

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


def _open_archive(parts):
    """Return something zipfile.ZipFile can open directly: the lone Path if
    there's only one part, or a _ChainedZipParts stream if the archive is
    split across multiple volumes."""
    if len(parts) == 1:
        return parts[0]
    return _ChainedZipParts(parts)


class _ChainedZipParts(io.RawIOBase):
    """Read-only, seekable, file-like view over multiple files concatenated
    in order — the read side of Info-ZIP's split-archive format, without
    physically merging the parts into a new file on disk.

    Subclasses io.RawIOBase (rather than duck-typing read/seek/tell alone)
    because zipfile's internals also probe .seekable()/.readable() — a
    plain object without those raises AttributeError deep inside zipfile.
    RawIOBase supplies sensible defaults for the rest of the file protocol.
    """

    def __init__(self, parts):
        super().__init__()
        self._parts = parts
        self._sizes = [p.stat().st_size for p in parts]
        self._offsets = []
        total = 0
        for size in self._sizes:
            self._offsets.append(total)
            total += size
        self._total_size = total
        self._pos = 0
        self._fh = None
        self._fh_idx = None

    def _part_index_for(self, pos):
        for i in range(len(self._parts) - 1, -1, -1):
            if pos >= self._offsets[i]:
                return i
        return 0

    def _ensure_open(self, idx):
        if self._fh_idx != idx:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(self._parts[idx], "rb")
            self._fh_idx = idx

    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return False

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._total_size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        return self._pos

    def tell(self):
        return self._pos

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._total_size - self._pos

        chunks = []
        remaining = size
        pos = self._pos
        while remaining > 0 and pos < self._total_size:
            idx = self._part_index_for(pos)
            self._ensure_open(idx)
            part_offset = pos - self._offsets[idx]
            self._fh.seek(part_offset)
            to_read = min(remaining, self._sizes[idx] - part_offset)
            chunk = self._fh.read(to_read)
            if not chunk:
                break
            chunks.append(chunk)
            pos += len(chunk)
            remaining -= len(chunk)

        self._pos = pos
        return b"".join(chunks)

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._fh_idx = None
        super().close()


def _extract_zip(archive, dest_dir):
    """archive: a Path, or a file-like object (_ChainedZipParts)."""
    import zipfile
    with zipfile.ZipFile(archive, "r") as zf:
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
