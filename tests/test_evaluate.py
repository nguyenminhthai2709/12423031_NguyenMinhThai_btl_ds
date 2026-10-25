"""Kiểm thử đơn vị cho module src/evaluate.py."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import evaluate


def test_compute_metrics_at_threshold():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])

    metrics = evaluate.compute_metrics_at_threshold(y_true, y_prob, threshold=0.5)
    assert metrics["threshold"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["confusion_matrix"]["tp"] == 3
    assert metrics["confusion_matrix"]["tn"] == 3
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0


def test_find_optimal_threshold_max_f1():
    # Trường hợp ngưỡng 0.5 bỏ lọt nhãn 1 (y_prob = 0.4)
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.15, 0.2, 0.35, 0.8])

    # Tại 0.5: chỉ bắt được 1 mẫu (tp=1, fn=1, fp=0 -> recall=0.5, f1=0.6667)
    res_def = evaluate.compute_metrics_at_threshold(y_true, y_prob, threshold=0.5)

    # Sau tối ưu threshold: ngưỡng hạ xuống ~0.35 để bắt cả 2 mẫu
    res_opt = evaluate.find_optimal_threshold_max_f1(y_true, y_prob)

    assert res_opt["f1_score"] >= res_def["f1_score"]
    assert 0.0 <= res_opt["threshold"] <= 1.0


def test_find_threshold_target_recall():
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.9])

    # Yêu cầu Recall >= 0.80 (ít nhất bắt 3/3 hoặc 3 mẫu nếu có thể)
    res_target = evaluate.find_threshold_target_recall(y_true, y_prob, min_recall=0.80)
    assert res_target["recall"] >= 0.80


def test_optimize_thresholds_for_all_models():
    y_true = np.array([0] * 90 + [1] * 10)
    probs = {
        "model_a_baseline": np.linspace(0.05, 0.95, 100),
        "model_b_weighted": np.linspace(0.1, 0.99, 100),
    }

    df_opt = evaluate.optimize_thresholds_for_all_models(y_true, probs)
    assert len(df_opt) == 2
    assert "opt_thresh_f1" in df_opt.columns
    assert "f1_improvement" in df_opt.columns
    assert "pr_auc" in df_opt.columns


def test_plots_generation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        y_true = np.array([0] * 80 + [1] * 20)
        probs = {
            "model_1_baseline": np.random.uniform(0, 1, 100),
            "model_2_weighted": np.random.uniform(0, 1, 100),
        }
        opt_thresh = {"model_1_baseline": 0.5, "model_2_weighted": 0.4}

        pr_file = tmp_path / "pr.png"
        roc_file = tmp_path / "roc.png"
        cm_file = tmp_path / "cm.png"

        evaluate.plot_pr_curves(y_true, probs, opt_thresh, [pr_file])
        evaluate.plot_roc_curves(y_true, probs, [roc_file])
        cm_dummy = {"tn": 80, "fp": 0, "fn": 2, "tp": 18}
        evaluate.plot_confusion_matrices(cm_dummy, cm_dummy, "TestModel", 0.5, [cm_file])

        assert pr_file.is_file() and pr_file.stat().st_size > 0
        assert roc_file.is_file() and roc_file.stat().st_size > 0
        assert cm_file.is_file() and cm_file.stat().st_size > 0
