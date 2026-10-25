"""Kiểm thử đơn vị cho module src/preprocessing.py."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import preprocessing
from fraud import config


def make_fixture(n_rows: int = 1000, n_fraud: int = 50, seed: int = 42) -> pd.DataFrame:
    """Tạo bảng dữ liệu nhỏ giả lập có đúng cấu trúc để kiểm thử các hàm logic."""
    rng = np.random.default_rng(seed)
    data = {config.TIME_COL: np.sort(rng.uniform(0, 48 * 3600, n_rows))}
    for col in config.PCA_FEATURES:
        data[col] = rng.normal(size=n_rows)
    data[config.AMOUNT_COL] = rng.lognormal(mean=3, sigma=1, size=n_rows)
    labels = np.zeros(n_rows, dtype=int)
    labels[rng.choice(n_rows, n_fraud, replace=False)] = 1
    data[config.TARGET_COL] = labels
    return pd.DataFrame(data)[config.EXPECTED_COLUMNS]


def test_separate_features_and_target():
    df = make_fixture(n_rows=100, n_fraud=10)
    X, y = preprocessing.separate_features_and_target(df)
    assert config.TARGET_COL not in X.columns
    assert len(X.columns) == 30
    assert y.name == config.TARGET_COL
    assert len(X) == len(y) == 100


def test_data_quality_detects_missing_and_duplicates():
    df = make_fixture(n_rows=100, n_fraud=10)
    df.loc[5, "V3"] = np.nan
    df.loc[12, "Amount"] = np.nan
    # Thêm 2 dòng trùng lặp
    df = pd.concat([df, df.iloc[[0]], df.iloc[[1]]], ignore_index=True)

    quality = preprocessing.check_data_quality(df)
    assert quality["total_missing_values"] == 2
    assert quality["columns_with_missing"] == {"V3": 1, "Amount": 1}
    assert quality["n_duplicate_rows"] == 2
    assert quality["has_missing_values"] is True


def test_split_proportions_and_stratification():
    df = make_fixture(n_rows=1000, n_fraud=100)
    X, y = preprocessing.separate_features_and_target(df)

    splits = preprocessing.split_data(
        X, y, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42
    )

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    # Kiểm tra kích thước chính xác (70% = 700, 15% = 150, 15% = 150)
    assert len(X_train) == 700
    assert len(X_val) == 150
    assert len(X_test) == 150

    # Kiểm tra stratify bảo toàn tỷ lệ gian lận (~10%)
    assert y_train.sum() == 70  # 70 / 700 = 10%
    assert y_val.sum() == 15    # 15 / 150 = 10%
    assert y_test.sum() == 15   # 15 / 150 = 10%


def test_calculate_imbalance_weights():
    # Giả sử 900 mẫu hợp lệ, 100 mẫu gian lận
    y_dummy = pd.Series([0] * 900 + [1] * 100)
    weights = preprocessing.calculate_imbalance_weights(y_dummy)

    # scale_pos_weight = 900 / 100 = 9.0
    assert weights["xgboost"]["value"] == 9.0
    assert weights["train_samples"]["n_negative"] == 900
    assert weights["train_samples"]["n_positive"] == 100


def test_preprocessor_scales_only_time_and_amount():
    df = make_fixture(n_rows=200, n_fraud=20)
    X, y = preprocessing.separate_features_and_target(df)
    splits = preprocessing.split_data(X, y)

    preprocessor = preprocessing.build_preprocessor()
    scaling_info = preprocessing.fit_and_verify_scaling(
        preprocessor, splits["X_train"], splits["X_val"], splits["X_test"]
    )

    assert scaling_info["scaler_fitted_on"] == "X_train_only"
    assert set(scaling_info["scaled_columns"]) == {"Time", "Amount"}
    assert "Time" in scaling_info["fitted_parameters"]
    assert "Amount" in scaling_info["fitted_parameters"]


def test_load_data_missing_file_raises_informative_error():
    fake_path = Path("khong_co_file_nay_dau.csv")
    with pytest.raises(FileNotFoundError) as exc_info:
        preprocessing.load_data(fake_path)
    assert "Kaggle" in str(exc_info.value) or "creditcard.csv" in str(exc_info.value)


def test_run_preprocessing_end_to_end_on_fixture():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_file = tmp_path / "creditcard.csv"
        out_summary = tmp_path / "preprocessing_summary.json"

        # Tạo file csv tạm
        df = make_fixture(n_rows=500, n_fraud=20)
        df.to_csv(data_file, index=False)

        summary = preprocessing.run_preprocessing(
            data_path=data_file, output_summary_path=out_summary
        )

        assert out_summary.is_file()
        with out_summary.open(encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["dataset_info"]["n_rows"] == 500
        assert loaded["split_configuration"]["train_ratio"] == 0.70
        assert "scale_pos_weight" in loaded["imbalance_handling"]["weights"]["xgboost"]["parameter"]
