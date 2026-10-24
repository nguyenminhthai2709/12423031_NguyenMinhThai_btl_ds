"""Basic consistency tests for fraud.config (no dataset required)."""

import importlib
import os

from fraud import config


def test_split_sizes_sum_to_one():
    assert abs(config.TRAIN_SIZE + config.VAL_SIZE + config.TEST_SIZE - 1.0) < 1e-9


def test_val_share_within_train_val_is_consistent():
    # Re-applying the two-step split must reproduce the 60/20/20 proportions.
    remaining = 1.0 - config.TEST_SIZE
    assert abs(remaining * config.VAL_SIZE_WITHIN_TRAIN_VAL - config.VAL_SIZE) < 1e-9
    assert abs(remaining * (1 - config.VAL_SIZE_WITHIN_TRAIN_VAL) - config.TRAIN_SIZE) < 1e-9


def test_validate_config_passes():
    config.validate_config()


def test_expected_schema_shape():
    # 1 Time + 28 PCA components + 1 Amount + 1 Class = 31 columns.
    assert len(config.EXPECTED_COLUMNS) == 31
    assert len(set(config.EXPECTED_COLUMNS)) == 31
    assert config.EXPECTED_COLUMNS[-1] == config.TARGET_COL
    assert config.TARGET_COL not in config.FEATURE_COLUMNS
    assert set(config.SCALED_COLUMNS) <= set(config.FEATURE_COLUMNS)


def test_paths_are_inside_project_and_portable():
    root = config.PROJECT_ROOT
    assert (root / "requirements.txt").is_file()
    assert (root / "src" / "fraud").is_dir()
    for path in (
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
        config.BEST_MODEL_PATH,
        config.MODEL_METADATA_PATH,
        config.FIGURES_DIR,
        config.TABLES_DIR,
        config.REPORT_PATH,
        config.NOTEBOOKS_DIR,
        config.OUTPUTS_DIR,
    ):
        assert path.is_relative_to(root), f"{path} is outside the project root"


def test_env_override_of_dataset_path():
    previous = os.environ.get("FRAUD_DATA_PATH")
    try:
        os.environ["FRAUD_DATA_PATH"] = os.path.join("some", "other", "creditcard.csv")
        reloaded = importlib.reload(config)
        assert reloaded.RAW_DATA_PATH.name == "creditcard.csv"
        assert "other" in reloaded.RAW_DATA_PATH.parts
    finally:
        if previous is None:
            os.environ.pop("FRAUD_DATA_PATH", None)
        else:
            os.environ["FRAUD_DATA_PATH"] = previous
        importlib.reload(config)
