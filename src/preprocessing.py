"""
preprocessing.py
================
Financial fraud data preprocessing pipeline for the QML Benchmark.

Responsibilities
----------------
1. Load raw transaction data from CSV.
2. Handle class imbalance via SMOTE or class-weight strategies.
3. Scale features (RobustScaler is preferred for financial data with outliers).
4. Reduce dimensionality via PCA to fit the qubit budget (4–8 qubits).
5. Expose a clean `preprocess()` function that returns train/test splits
   ready for both classical and quantum model pipelines.

Notes
-----
- RobustScaler is used instead of StandardScaler because financial transaction
  amounts typically contain extreme outliers that would distort z-scores.
- PCA components are fitted *only* on the training set to prevent data leakage.
- SMOTE is applied *after* the train/test split and *only* on the training set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """All tunable knobs for the preprocessing pipeline."""

    # Path to the raw CSV (e.g. Kaggle Credit Card Fraud dataset)
    data_path: str | Path = Path("data/raw/creditcard.csv")

    # Column names
    target_column: str = "Class"

    # Train / test / validation split
    test_size: float = 0.20
    val_size: float = 0.15   # fraction of training data held out pre-SMOTE
    random_state: int = 42

    # Imbalance strategy: "smote" | "class_weight" | "none"
    imbalance_strategy: Literal["smote", "class_weight", "none"] = "smote"

    # SMOTE parameters (ignored when strategy != "smote")
    smote_k_neighbors: int = 5

    # PCA — number of components == number of qubits to target
    n_qubits: int = 8
    apply_pca: bool = True

    # Columns to drop before modelling (e.g. identifiers, timestamps)
    drop_columns: list[str] = field(default_factory=lambda: ["Time"])


# ---------------------------------------------------------------------------
# Pipeline result container
# ---------------------------------------------------------------------------

@dataclass
class PreprocessedData:
    """Holds all artefacts produced by the preprocessing pipeline."""

    X_train: np.ndarray
    X_val: np.ndarray          # pre-SMOTE, real class distribution — for threshold tuning
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    # Fitted transformers — kept for later inference / explainability
    scaler: RobustScaler
    pca: PCA | None

    # Metadata
    n_features_original: int
    n_features_final: int
    class_counts_original: dict[int, int]
    class_counts_after_resampling: dict[int, int]

    @property
    def class_weight_dict(self) -> dict[int, float]:
        """Inverse-frequency class weights for use with sklearn estimators."""
        total = sum(self.class_counts_original.values())
        n_classes = len(self.class_counts_original)
        return {
            cls: total / (n_classes * count)
            for cls, count in self.class_counts_original.items()
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_raw(config: PreprocessingConfig) -> pd.DataFrame:
    path = Path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Download the Kaggle Credit Card Fraud dataset and place it at "
            "data/raw/creditcard.csv, or update PreprocessingConfig.data_path."
        )
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Raw shape: %s | Fraud rate: %.4f%%",
                df.shape,
                df[config.target_column].mean() * 100)
    return df


def _drop_and_split_xy(
    df: pd.DataFrame, config: PreprocessingConfig
) -> tuple[pd.DataFrame, pd.Series]:
    cols_to_drop = [c for c in config.drop_columns if c in df.columns]
    if cols_to_drop:
        logger.info("Dropping columns: %s", cols_to_drop)
    X = df.drop(columns=cols_to_drop + [config.target_column], errors="ignore")
    y = df[config.target_column]
    return X, y


def _apply_smote(
    X: np.ndarray, y: np.ndarray, config: PreprocessingConfig
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Applying SMOTE (k_neighbors=%d) ...", config.smote_k_neighbors)
    smote = SMOTE(
        k_neighbors=config.smote_k_neighbors,
        random_state=config.random_state,
    )
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(
        "After SMOTE — class distribution: %s",
        dict(zip(*np.unique(y_res, return_counts=True))),
    )
    return X_res, y_res


def _fit_scale(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
    """Fit RobustScaler on train, transform both splits."""
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def _fit_pca(
    X_train: np.ndarray, X_test: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Fit PCA on the training set only.

    The number of components is capped at min(n_components, n_features, n_samples)
    to avoid errors on small datasets.
    """
    n_components = min(n_components, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    explained = pca.explained_variance_ratio_.sum() * 100
    logger.info(
        "PCA: %d components retain %.2f%% of variance", n_components, explained
    )
    return X_train_pca, X_test_pca, pca


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(config: PreprocessingConfig | None = None) -> PreprocessedData:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    config : PreprocessingConfig, optional
        Pipeline configuration. Defaults are tuned for the Kaggle Credit Card
        Fraud dataset with an 8-qubit quantum backend.

    Returns
    -------
    PreprocessedData
        Train/test arrays plus fitted transformers and metadata.
    """
    if config is None:
        config = PreprocessingConfig()

    # 1. Load ---------------------------------------------------------------
    df = _load_raw(config)

    # 2. Separate features / target -----------------------------------------
    X, y = _drop_and_split_xy(df, config)
    n_features_original = X.shape[1]
    class_counts_original = dict(zip(*np.unique(y, return_counts=True)))
    logger.info("Original class counts: %s", class_counts_original)

    # 3. Train / test split (stratified to preserve fraud ratio) ------------
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    # 3b. Validation split — carved off BEFORE SMOTE to keep real distribution
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=config.val_size,
        stratify=y_train,
        random_state=config.random_state,
    )
    logger.info(
        "Split — train: %d | val: %d | test: %d samples",
        len(X_train), len(X_val), len(X_test),
    )

    # 4. Scale (fit on train only, transform val + test) --------------------
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # 5. Imbalance handling (training set only) ------------------------------
    if config.imbalance_strategy == "smote":
        X_train, y_train = _apply_smote(X_train, y_train, config)
    elif config.imbalance_strategy == "class_weight":
        logger.info(
            "Imbalance strategy: class_weight — weights will be returned in "
            "PreprocessedData.class_weight_dict for use in model training."
        )
    else:
        logger.warning(
            "Imbalance strategy: none — model may be biased towards majority class."
        )

    class_counts_after = dict(zip(*np.unique(y_train, return_counts=True)))

    # 6. PCA → qubit-ready feature space ------------------------------------
    pca_model: PCA | None = None
    if config.apply_pca:
        n_components = min(config.n_qubits, X_train.shape[1], X_train.shape[0])
        pca_model = PCA(n_components=n_components, random_state=42)
        X_train = pca_model.fit_transform(X_train)
        X_val   = pca_model.transform(X_val)
        X_test  = pca_model.transform(X_test)
        explained = pca_model.explained_variance_ratio_.sum() * 100
        logger.info("PCA: %d components retain %.2f%% of variance",
                    n_components, explained)

    n_features_final = X_train.shape[1]
    logger.info("Preprocessing complete. Final feature dim: %d", n_features_final)

    return PreprocessedData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
        pca=pca_model,
        n_features_original=n_features_original,
        n_features_final=n_features_final,
        class_counts_original=class_counts_original,
        class_counts_after_resampling=class_counts_after,
    )


# ---------------------------------------------------------------------------
# Quick sanity-check (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    cfg = PreprocessingConfig(
        data_path="data/raw/creditcard.csv",
        n_qubits=8,
        imbalance_strategy="smote",
    )
    data = preprocess(cfg)

    print("\n=== Preprocessing Summary ===")
    print(f"  Original features : {data.n_features_original}")
    print(f"  Final features    : {data.n_features_final}")
    print(f"  X_train shape     : {data.X_train.shape}")
    print(f"  X_test shape      : {data.X_test.shape}")
    print(f"  Class weights     : {data.class_weight_dict}")
    print(f"  PCA variance kept : "
          f"{data.pca.explained_variance_ratio_.sum() * 100:.2f}%"
          if data.pca else "  PCA               : disabled")
