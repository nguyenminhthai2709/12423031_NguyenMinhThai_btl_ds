"""Central configuration for the Credit Card Fraud Detection project.

Every path, random seed, split ratio and modelling option that more than one
module needs lives here, so that:

* nothing is hard-coded to a single machine (all paths are derived from the
  location of this file, so the project works on Windows, Linux and macOS);
* experiments are reproducible (one seed, one set of split ratios);
* a decision made once (e.g. 60/20/20 split) is changed in exactly one place.

Nothing in this module reads the dataset or trains anything. Importing it has
no side effects (it does not create folders); call ``ensure_directories()``
explicitly when folders are needed.

Quick check:  ``python -m fraud.config``
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Paths (all relative to the project root, never absolute)
# ---------------------------------------------------------------------------
# This file is <root>/src/fraud/config.py  ->  parents[2] is <root>.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

DATASET_FILENAME: str = "creditcard.csv"
# Optional override, e.g. in PowerShell:  $env:FRAUD_DATA_PATH = "D:\data\creditcard.csv"
RAW_DATA_PATH: Path = Path(
    os.environ.get("FRAUD_DATA_PATH", RAW_DATA_DIR / DATASET_FILENAME)
).expanduser()

MODELS_DIR: Path = PROJECT_ROOT / "models"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"  # chosen threshold, metrics, seed...

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"  # EDA charts, PR curves, confusion matrices
TABLES_DIR: Path = REPORTS_DIR / "tables"  # model comparison tables (csv)
REPORT_PATH: Path = REPORTS_DIR / "bao_cao.md"  # Vietnamese report (written in the last phase)

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"  # scratch space, git-ignored

# ---------------------------------------------------------------------------
# 2. Dataset schema (what we EXPECT; the EDA phase must verify it on real data)
# ---------------------------------------------------------------------------
TARGET_COL: str = "Class"  # 0 = legitimate, 1 = fraud
POSITIVE_LABEL: int = 1
TIME_COL: str = "Time"  # seconds elapsed since the first transaction in the dataset
AMOUNT_COL: str = "Amount"
PCA_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]  # anonymised PCA components

FEATURE_COLUMNS: list[str] = [TIME_COL, *PCA_FEATURES, AMOUNT_COL]
EXPECTED_COLUMNS: list[str] = [*FEATURE_COLUMNS, TARGET_COL]

# Columns that are NOT already PCA output and live on very different scales.
# Scaling is applied to these inside a sklearn Pipeline (fit on train only).
SCALED_COLUMNS: list[str] = [TIME_COL, AMOUNT_COL]

# ---------------------------------------------------------------------------
# 3. Reproducibility and data splitting
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# Train / validation / test proportions (must sum to 1.0).
TRAIN_SIZE: float = 0.60
VAL_SIZE: float = 0.20
TEST_SIZE: float = 0.20

# Two-step splitting: first cut off TEST_SIZE, then split the remainder into
# train and validation. The validation share *of the remainder* is derived:
VAL_SIZE_WITHIN_TRAIN_VAL: float = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)

# Main experiment: stratified random split (keeps the fraud ratio in each part).
# Extra experiment: temporal split (sort by TIME_COL, oldest -> train, newest -> test),
# which is closer to real deployment because the future is not seen during training.
SPLIT_STRATEGY: str = "stratified"
TEMPORAL_SPLIT_SORT_COL: str = TIME_COL

# ---------------------------------------------------------------------------
# 4. Models and class-imbalance handling
# ---------------------------------------------------------------------------
MODEL_NAMES: tuple[str, ...] = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
)
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

# Each model is compared with and without imbalance handling:
#   baseline     -> no weighting (default behaviour)
#   class_weight -> class_weight="balanced"-style weighting for sklearn models,
#                   scale_pos_weight = n_negative / n_positive (computed on TRAIN only) for XGBoost
IMBALANCE_STRATEGIES: tuple[str, ...] = ("baseline", "class_weight")

N_JOBS: int = -1  # use all CPU cores where supported

# ---------------------------------------------------------------------------
# 5. Evaluation and threshold selection (all done on the VALIDATION set)
# ---------------------------------------------------------------------------
# Model-selection metric: PR-AUC computed as average precision
# (sklearn.metrics.average_precision_score). Accuracy is NOT used.
MODEL_SELECTION_METRIC: str = "average_precision"

# Primary threshold criterion: threshold that maximises F1 on validation.
THRESHOLD_CRITERION: str = "max_f1"
# Secondary criterion reported for comparison: highest precision subject to a
# minimum recall. Adjustable business assumption, NOT an experimental result.
MIN_RECALL_TARGET: float = 0.80

FIGURE_DPI: int = 150


# ---------------------------------------------------------------------------
# 6. Helpers
# ---------------------------------------------------------------------------
def ensure_directories() -> None:
    """Create the folders that generated files are written to (idempotent)."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        NOTEBOOKS_DIR,
        OUTPUTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def validate_config() -> None:
    """Raise ``ValueError`` if the configuration is internally inconsistent."""
    total = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"TRAIN/VAL/TEST sizes must sum to 1.0, got {total}")
    if min(TRAIN_SIZE, VAL_SIZE, TEST_SIZE) <= 0:
        raise ValueError("TRAIN_SIZE, VAL_SIZE and TEST_SIZE must all be > 0")
    if not 0.0 < MIN_RECALL_TARGET <= 1.0:
        raise ValueError("MIN_RECALL_TARGET must be in (0, 1]")
    if len(set(EXPECTED_COLUMNS)) != len(EXPECTED_COLUMNS):
        raise ValueError("EXPECTED_COLUMNS contains duplicates")
    if set(MODEL_NAMES) != set(MODEL_DISPLAY_NAMES):
        raise ValueError("MODEL_NAMES and MODEL_DISPLAY_NAMES are out of sync")


if __name__ == "__main__":
    validate_config()
    print(f"PROJECT_ROOT   : {PROJECT_ROOT}")
    print(f"RAW_DATA_PATH  : {RAW_DATA_PATH}  (exists: {RAW_DATA_PATH.exists()})")
    print(f"MODELS_DIR     : {MODELS_DIR}")
    print(f"FIGURES_DIR    : {FIGURES_DIR}")
    print(f"TABLES_DIR     : {TABLES_DIR}")
    print(f"RANDOM_SEED    : {RANDOM_SEED}")
    print(f"SPLIT (tr/va/te): {TRAIN_SIZE}/{VAL_SIZE}/{TEST_SIZE}")
    print(f"Models         : {', '.join(MODEL_DISPLAY_NAMES[m] for m in MODEL_NAMES)}")
    print("Configuration is consistent.")
