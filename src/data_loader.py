"""
data_loader.py
==============
Thin wrapper for downloading / verifying the raw dataset.

The primary dataset is the Kaggle Credit Card Fraud Detection dataset
(creditcard.csv, ~150 MB).  This module checks for its presence and
prints instructions if it is missing.
"""

from __future__ import annotations

from pathlib import Path

DATASET_URL = (
    "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
)
DEFAULT_RAW_PATH = Path("data/raw/creditcard.csv")


def verify_dataset(path: str | Path = DEFAULT_RAW_PATH) -> Path:
    """
    Verify the raw dataset exists and return its resolved path.

    Raises
    ------
    FileNotFoundError
        With download instructions if the file is absent.
    """
    path = Path(path)
    if path.exists():
        return path.resolve()

    raise FileNotFoundError(
        f"\nDataset not found at: {path}\n\n"
        "Download instructions:\n"
        f"  1. Visit {DATASET_URL}\n"
        "  2. Accept the competition rules and download creditcard.csv\n"
        f"  3. Place the file at: {path.resolve()}\n"
        "\nAlternatively, use the Kaggle CLI:\n"
        "  kaggle datasets download -d mlg-ulb/creditcardfraud "
        f"--path {path.parent} --unzip\n"
    )
