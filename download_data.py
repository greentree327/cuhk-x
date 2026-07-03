"""
# CONVENTION: primary — Propagating-exception convention

Downloads and extracts the CUHK-X Small Model Track dataset from Hugging
Face into the exact layout src/config.py expects. Safe to re-run — it's a
no-op if the dataset is already in place.

One-time setup before the first run:
    1. Visit https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track
       and click "Agree and access repository" (it's a gated dataset).
    2. Create a read token at https://huggingface.co/settings/tokens.
    3. Set it as an environment variable: export HF_TOKEN=hf_...
       (in Colab: %env HF_TOKEN=hf_... or via the Secrets manager)

Usage:
    python download_data.py

run_ablation.py also calls this automatically before training, so a plain
`git clone` + `python run_ablation.py` (with HF_TOKEN set) is enough end to
end — this script exists separately for explicit testing/diagnostics.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.data_bootstrap import ensure_dataset_available

if __name__ == "__main__":
    ensure_dataset_available(Config())
