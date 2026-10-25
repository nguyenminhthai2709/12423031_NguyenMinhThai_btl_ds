"""Kiểm thử đơn vị cho module src/train.py."""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import train
from fraud import config


def make_fixture(n_rows: int = 400, n_fraud: int = 20, seed: int = 42) -> pd.DataFrame:
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


def test_build_model_candidates():
    candidates = train.build_model_candidates(scale_pos_weight=15.5)
    # 4 thuật toán x 2 chiến lược = 8 mô hình
    assert len(candidates) == 8

    # Kiểm tra Logistic Regression sử dụng Pipeline có StandardScaler
    lr_base = candidates["logistic_regression_baseline"]["estimator"]
    assert isinstance(lr_base, Pipeline)
    assert "scaler" in lr_base.named_steps
    assert "classifier" in lr_base.named_steps

    # Kiểm tra XGBoost weighted nhận scale_pos_weight
    xgb_weighted = candidates["xgboost_weighted"]["estimator"]
    assert xgb_weighted.scale_pos_weight == 15.5


def test_evaluate_predictions():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.4, 0.9])

    metrics = train.evaluate_predictions(y_true, y_prob, threshold=0.5)
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert metrics["precision"] == 1.0  # Tất cả mẫu có prob >= 0.5 đều là nhãn 1
    assert metrics["recall"] == 1.0


def test_train_and_evaluate_end_to_end_on_fixture():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_path = tmp_path / "creditcard.csv"
        models_dir = tmp_path / "models"
        metrics_dir = tmp_path / "outputs" / "metrics"
        tables_dir = tmp_path / "reports" / "tables"

        df = make_fixture(n_rows=300, n_fraud=15)
        df.to_csv(csv_path, index=False)

        df_comparison, summary = train.train_and_evaluate_all(
            data_path=csv_path,
            models_dir=models_dir,
            metrics_dir=metrics_dir,
            tables_dir=tables_dir,
        )

        # Kiểm tra bảng so sánh có đủ 8 dòng
        assert len(df_comparison) == 8
        assert "pr_auc" in df_comparison.columns
        assert "f1_score" in df_comparison.columns

        # Kiểm tra các file kết quả được tạo ra
        assert (metrics_dir / "model_comparison.csv").is_file()
        assert (tables_dir / "model_comparison.csv").is_file()
        assert (metrics_dir / "val_predictions.joblib").is_file()
        assert (metrics_dir / "training_summary.json").is_file()

        # Kiểm tra file val_predictions có cấu trúc hợp lệ
        val_preds = joblib.load(metrics_dir / "val_predictions.joblib")
        assert "y_val_true" in val_preds
        assert "probabilities" in val_preds
        assert len(val_preds["probabilities"]) == 8
