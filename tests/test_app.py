"""Kiểm thử đơn vị cho ứng dụng Streamlit (app/app.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Thêm thư mục app và src vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import app  # noqa: E402
from fraud import config  # noqa: E402


def test_load_trained_model_and_metadata():
    model, metadata, err = app.load_trained_model()
    assert err is None, f"Lỗi nạp model: {err}"
    assert model is not None
    assert hasattr(model, "predict_proba")
    assert metadata is not None
    assert "best_model_name" in metadata
    assert abs(metadata.get("chosen_threshold", 0) - 0.5701) < 1e-4


def test_load_metrics_data():
    data = app.load_metrics_data()
    assert "preprocessing" in data
    assert "comparison" in data
    assert "threshold_optimization" in data
    assert "evaluation" in data

    # Kiểm tra DataFrame so sánh có đúng 8 mô hình
    df_comp = data["comparison"]
    assert len(df_comp) == 8


def test_single_prediction_logic():
    model, metadata, _ = app.load_trained_model()
    threshold = float(metadata.get("chosen_threshold", 0.5701))

    # Tạo 1 dòng dữ liệu hợp lệ
    dummy_input = pd.DataFrame([{col: 0.0 for col in config.FEATURE_COLUMNS}])
    dummy_input["Amount"] = 100.0
    dummy_input["Time"] = 50000.0

    probs = model.predict_proba(dummy_input)[:, 1]
    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0

    pred_class = int(probs[0] >= threshold)
    assert pred_class in (0, 1)


def test_batch_prediction_logic():
    model, metadata, _ = app.load_trained_model()
    threshold = float(metadata.get("chosen_threshold", 0.5701))

    sample_csv = PROJECT_ROOT / "data" / "sample_batch_transactions.csv"
    if sample_csv.is_file():
        df_batch = pd.read_csv(sample_csv)
        assert len(df_batch) == 20

        # Dự đoán lô
        probs = model.predict_proba(df_batch[config.FEATURE_COLUMNS])[:, 1]
        preds = (probs >= threshold).astype(int)

        assert len(probs) == 20
        assert len(preds) == 20
        # Mẫu chứa cả giao dịch hợp lệ và gian lận
        assert (preds == 1).sum() > 0
        assert (preds == 0).sum() > 0

        # Kiểm tra bảng kết quả chứa đầy đủ 30 đặc trưng đầu vào + Predicted_Probability + Status
        df_result = df_batch.copy()
        df_result["Predicted_Probability"] = np.round(probs, 4)
        df_result["Predicted_Class"] = preds
        df_result["Status"] = np.where(preds == 1, "🔴 GIAN LẬN", "🟢 Hợp lệ")

        for col in config.FEATURE_COLUMNS:
            assert col in df_result.columns, f"Thiếu cột đặc trưng {col} trong kết quả"
        assert "Predicted_Probability" in df_result.columns
        assert "Status" in df_result.columns
        assert len([c for c in df_result.columns if c.startswith("V")]) == 28



def test_preset_test_samples_in_app():
    """Kiểm tra hai mẫu thực tế trên giao diện app.py:

    - SAMPLE_LEGIT_TRANSACTION: đầy đủ 30 đặc trưng, P(Fraud) < threshold -> Hợp lệ
    - SAMPLE_FRAUD_TRANSACTION: đầy đủ 30 đặc trưng, P(Fraud) >= threshold -> Gian lận (khoảng 73%)
    """
    model, metadata, _ = app.load_trained_model()
    threshold = float(metadata.get("chosen_threshold", 0.5701))

    # Kiểm tra đủ 30 đặc trưng
    for col in config.FEATURE_COLUMNS:
        assert col in app.SAMPLE_LEGIT_TRANSACTION, f"Thiếu {col} trong SAMPLE_LEGIT_TRANSACTION"
        assert col in app.SAMPLE_FRAUD_TRANSACTION, f"Thiếu {col} trong SAMPLE_FRAUD_TRANSACTION"

    # 1. Kiểm tra mẫu hợp lệ
    df_legit = pd.DataFrame([app.SAMPLE_LEGIT_TRANSACTION])[config.FEATURE_COLUMNS]
    p_legit = float(model.predict_proba(df_legit)[0, 1])
    assert p_legit < threshold, f"Mẫu hợp lệ có xác suất gian lận {p_legit:.4f} >= threshold {threshold}"
    assert int(p_legit >= threshold) == 0

    # 2. Kiểm tra mẫu gian lận thực tế từ tập test
    df_fraud = pd.DataFrame([app.SAMPLE_FRAUD_TRANSACTION])[config.FEATURE_COLUMNS]
    p_fraud = float(model.predict_proba(df_fraud)[0, 1])
    assert p_fraud >= threshold, f"Mẫu gian lận có xác suất {p_fraud:.4f} < threshold {threshold}"
    assert int(p_fraud >= threshold) == 1
    # Kiểm tra xác suất rơi vào khoảng 70% - 76% (chính xác là 73.00%)
    assert 0.70 <= p_fraud <= 0.76

