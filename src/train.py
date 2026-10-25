"""Huấn luyện và so sánh mô hình (Model Training) - Phase 4.
Bài tập lớn: Phát hiện gian lận thẻ tín dụng (Credit Card Fraud Detection).
Sinh viên: Nguyễn Minh Thái - MSSV: 12423031.

Mục tiêu của script:
1. Nạp các tập Train (70%), Validation (15%), Test (15%) từ Phase 3.
2. Huấn luyện 4 thuật toán Machine Learning:
   - Logistic Regression (kết hợp StandardScaler thông qua Pipeline)
   - Decision Tree (Cây quyết định)
   - Random Forest (Rừng ngẫu nhiên)
   - XGBoost (Extreme Gradient Boosting)
3. So sánh 2 chiến lược xử lý dữ liệu mất cân bằng cho từng mô hình:
   - Baseline: Không sử dụng trọng số lớp (mặc định)
   - Weighted: Sử dụng trọng số lớp (class_weight='balanced' hoặc scale_pos_weight)
4. Đánh giá hiệu năng trên tập Validation bằng các thước đo:
   - PR-AUC (Average Precision score) - Thước đo chính cho bài toán mất cân bằng cực đoan
   - Precision (Độ chuẩn xác)
   - Recall (Độ thu hồi / độ nhạy)
   - F1-score (Trung bình điều hòa giữa Precision và Recall)
   - ROC-AUC (Diện tích dưới đường cong ROC)
   (Sử dụng ngưỡng quyết định mặc định threshold = 0.5)
5. Tuyệt đối KHÔNG sử dụng tập Test để huấn luyện hoặc lựa chọn mô hình.
6. Lưu trữ:
   - Các mô hình đã huấn luyện vào thư mục models/
   - Bảng so sánh vào outputs/metrics/model_comparison.csv
   - Xác suất dự đoán trên tập Validation vào outputs/metrics/val_predictions.joblib (phục vụ Phase 5)
   - Báo cáo tổng kết vào outputs/metrics/training_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Đảm bảo hiển thị tiếng Việt trơn tru trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo import được package fraud khi chạy từ bất kỳ thư mục nào
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fraud import config  # noqa: E402
from preprocessing import build_preprocessor, load_and_split_data  # noqa: E402


# ===========================================================================
# 1. Khởi tạo danh sách các mô hình ứng viên
# ===========================================================================
def build_model_candidates(scale_pos_weight: float) -> dict[str, dict[str, Any]]:
    """Tạo từ điển chứa 8 cấu hình mô hình (4 thuật toán x 2 chiến lược trọng số).

    Các tham số được tinh chỉnh vừa phải để chạy mượt mà trên máy tính cá nhân
    nhưng vẫn đảm bảo hiệu năng và tính tổng quát hóa cao.
    """
    models: dict[str, dict[str, Any]] = {
        # --- 1. Logistic Regression ---
        "logistic_regression_baseline": {
            "display_name": "Logistic Regression",
            "strategy": "baseline",
            "estimator": Pipeline(
                [
                    ("scaler", build_preprocessor()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=config.RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            "description": "Hồi quy Logistic chuẩn hóa StandardScaler, không gán trọng số lớp.",
        },
        "logistic_regression_weighted": {
            "display_name": "Logistic Regression",
            "strategy": "class_weight",
            "estimator": Pipeline(
                [
                    ("scaler", build_preprocessor()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=config.RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            "description": "Hồi quy Logistic chuẩn hóa StandardScaler, class_weight='balanced'.",
        },
        # --- 2. Decision Tree ---
        "decision_tree_baseline": {
            "display_name": "Decision Tree",
            "strategy": "baseline",
            "estimator": DecisionTreeClassifier(
                max_depth=10,
                random_state=config.RANDOM_SEED,
            ),
            "description": "Cây quyết định (max_depth=10), không gán trọng số lớp.",
        },
        "decision_tree_weighted": {
            "display_name": "Decision Tree",
            "strategy": "class_weight",
            "estimator": DecisionTreeClassifier(
                max_depth=10,
                class_weight="balanced",
                random_state=config.RANDOM_SEED,
            ),
            "description": "Cây quyết định (max_depth=10), class_weight='balanced'.",
        },
        # --- 3. Random Forest ---
        "random_forest_baseline": {
            "display_name": "Random Forest",
            "strategy": "baseline",
            "estimator": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
            ),
            "description": "Rừng ngẫu nhiên (100 cây, max_depth=12), không gán trọng số.",
        },
        "random_forest_weighted": {
            "display_name": "Random Forest",
            "strategy": "class_weight",
            "estimator": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
            ),
            "description": "Rừng ngẫu nhiên (100 cây, max_depth=12), class_weight='balanced'.",
        },
        # --- 4. XGBoost ---
        "xgboost_baseline": {
            "display_name": "XGBoost",
            "strategy": "baseline",
            "estimator": XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=config.RANDOM_SEED,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            "description": "XGBoost (100 cây, depth=5, lr=0.1), scale_pos_weight=1.",
        },
        "xgboost_weighted": {
            "display_name": "XGBoost",
            "strategy": "class_weight",
            "estimator": XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=config.RANDOM_SEED,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            "description": f"XGBoost (100 cây, depth=5, lr=0.1), scale_pos_weight={scale_pos_weight:.2f}.",
        },
    }
    return models


# ===========================================================================
# 2. Đánh giá dự đoán trên tập Validation
# ===========================================================================
def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Tính toán các chỉ số đánh giá phân loại tại ngưỡng threshold cho trước."""
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "threshold": threshold,
    }


def safe_relative_path(path: Path) -> str:
    """Trả về đường dẫn tương đối so với project root nếu có thể, ngược lại trả về str(path)."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except (ValueError, RuntimeError):
        return str(path)


# ===========================================================================
# 3. Quy trình huấn luyện và tổng hợp kết quả
# ===========================================================================
def train_and_evaluate_all(
    data_path: Path = config.RAW_DATA_PATH,
    models_dir: Path = config.MODELS_DIR,
    metrics_dir: Path = config.METRICS_DIR,
    tables_dir: Path = config.TABLES_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Chạy toàn bộ quy trình huấn luyện 4 mô hình (8 phiên bản) và lưu artifacts."""
    print("=" * 80)
    print("PHASE 4: MODEL TRAINING (HUẤN LUYỆN VÀ SO SÁNH 4 MÔ HÌNH)")
    print("=" * 80)

    # Đảm bảo các thư mục đầu ra tồn tại
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tải và phân chia dữ liệu từ Phase 3
    print("\n[Bước 1/4] Đọc dữ liệu và chia tập (Train 70% - Validation 15% - Test 15%)...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(data_path)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    print(f"  - Tập Train      : {len(X_train):,} mẫu (Hợp lệ: {n_neg:,}, Gian lận: {n_pos:,})")
    print(f"  - Tập Validation : {len(X_val):,} mẫu (Dùng để so sánh mô hình)")
    print(f"  - Tập Test       : {len(X_test):,} mẫu (GIỮ NGUYÊN VẸN, tuyệt đối không dùng trong Phase này)")
    print(f"  - scale_pos_weight tính trên Train = {scale_pos_weight:.2f}")

    # 2. Xây dựng danh sách mô hình
    candidates = build_model_candidates(scale_pos_weight)
    print(f"\n[Bước 2/4] Khởi tạo {len(candidates)} cấu hình mô hình:")
    for key, info in candidates.items():
        print(f"  + [{key:28s}] {info['description']}")

    # 3. Vòng lặp huấn luyện và đánh giá
    print("\n[Bước 3/4] Bắt đầu huấn luyện và đánh giá trên tập Validation...")
    results_list: list[dict[str, Any]] = []
    val_predictions: dict[str, Any] = {
        "y_val_true": y_val.to_numpy(),
        "probabilities": {},
    }

    start_total_time = time.perf_counter()

    for idx, (model_key, model_info) in enumerate(candidates.items(), 1):
        display_name = model_info["display_name"]
        strategy = model_info["strategy"]
        estimator = model_info["estimator"]

        print(f"\n({idx}/{len(candidates)}) Đang huấn luyện: {display_name} [{strategy}]...")
        t0 = time.perf_counter()

        # Huấn luyện CHỈ trên Train
        estimator.fit(X_train, y_train)
        fit_duration = time.perf_counter() - t0

        # Dự đoán xác suất trên Validation
        y_val_prob = estimator.predict_proba(X_val)[:, 1]

        # Đánh giá với ngưỡng mặc định 0.5
        metrics = evaluate_predictions(y_val, y_val_prob, threshold=0.5)

        # Lưu model đã train ra file
        model_save_path = models_dir / f"{model_key}.joblib"
        joblib.dump(estimator, model_save_path)

        # Lưu xác suất dự đoán để Phase 5 dùng tiếp (tránh phải fit lại)
        val_predictions["probabilities"][model_key] = y_val_prob

        # Ghi nhận kết quả
        res_record = {
            "model_key": model_key,
            "model_name": display_name,
            "strategy": strategy,
            "pr_auc": metrics["pr_auc"],
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "roc_auc": metrics["roc_auc"],
            "threshold": metrics["threshold"],
            "fit_time_seconds": round(fit_duration, 2),
            "model_path": safe_relative_path(model_save_path),
        }
        results_list.append(res_record)

        print(
            f"      Xong ({fit_duration:.2f}s) -> "
            f"PR-AUC: {metrics['pr_auc']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f}"
        )

    total_training_time = time.perf_counter() - start_total_time

    # 4. Tạo DataFrame so sánh và sắp xếp
    df_comparison = pd.DataFrame(results_list)
    # Sắp xếp theo PR-AUC giảm dần (thước đo chuẩn cho bài toán mất cân bằng)
    df_comparison.sort_values(by=["pr_auc", "f1_score"], ascending=[False, False], inplace=True)
    df_comparison.reset_index(drop=True, inplace=True)

    # 5. Lưu các file kết quả
    print("\n[Bước 4/4] Lưu kết quả và xác suất dự đoán...")

    # a. Bảng so sánh CSV
    csv_metric_path = metrics_dir / "model_comparison.csv"
    csv_report_path = tables_dir / "model_comparison.csv"
    df_comparison.to_csv(csv_metric_path, index=False, encoding="utf-8")
    df_comparison.to_csv(csv_report_path, index=False, encoding="utf-8")
    print(f"  - Đã lưu bảng so sánh: {safe_relative_path(csv_metric_path)}")

    # b. Lưu xác suất dự đoán tập Validation (joblib)
    val_pred_path = metrics_dir / "val_predictions.joblib"
    joblib.dump(val_predictions, val_pred_path)
    print(f"  - Đã lưu xác suất dự đoán Validation: {safe_relative_path(val_pred_path)}")

    # c. Lưu file tóm tắt JSON
    summary_path = metrics_dir / "training_summary.json"
    summary_dict = {
        "task": "Model Training & Comparison (Phase 4)",
        "dataset": data_path.name,
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "n_test_samples": len(X_test),
        "scale_pos_weight_train": round(scale_pos_weight, 4),
        "total_training_time_seconds": round(total_training_time, 2),
        "default_threshold": 0.5,
        "primary_selection_metric": "PR-AUC (Average Precision)",
        "test_set_policy": "Untouched (strictly reserved for final Phase evaluation)",
        "models_evaluated": df_comparison.to_dict(orient="records"),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=2)
    print(f"  - Đã lưu tóm tắt JSON: {safe_relative_path(summary_path)}")

    # In bảng kết quả tổng quan ra màn hình
    print("\n" + "=" * 90)
    print("BẢNG KẾT QUẢ SO SÁNH TRÊN TẬP VALIDATION (NGƯỠNG MẶC ĐỊNH 0.5)")
    print("=" * 90)
    display_df = df_comparison[
        ["model_name", "strategy", "pr_auc", "f1_score", "precision", "recall", "roc_auc", "fit_time_seconds"]
    ].copy()
    display_df.columns = [
        "Mô hình",
        "Chiến lược",
        "PR-AUC",
        "F1-score",
        "Precision",
        "Recall",
        "ROC-AUC",
        "Thời gian (s)",
    ]
    print(display_df.to_string(index=False))
    print("=" * 90)
    print(f"Tổng thời gian huấn luyện: {total_training_time:.2f} giây.")
    print("Ghi chú: Kết quả trên sử dụng threshold mặc định 0.5. Ở Phase 5, ta sẽ tinh chỉnh ngưỡng")
    print("để tối ưu F1 và Recall dựa trên file xác suất đã lưu.")

    return df_comparison, summary_dict


# ===========================================================================
# 4. Giao diện dòng lệnh (CLI)
# ===========================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện và so sánh 4 mô hình Machine Learning cho phát hiện gian lận thẻ."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=config.RAW_DATA_PATH,
        help="Đường dẫn file creditcard.csv (mặc định lấy từ config)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=config.MODELS_DIR,
        help="Thư mục lưu các model joblib (mặc định: models/)",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=config.METRICS_DIR,
        help="Thư mục lưu metrics đầu ra (mặc định: outputs/metrics/)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        train_and_evaluate_all(
            data_path=args.data_path,
            models_dir=args.models_dir,
            metrics_dir=args.metrics_dir,
        )
        return 0
    except FileNotFoundError as exc:
        print(f"\n[LỖI]: Không tìm thấy file dữ liệu: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[LỖI KHÔNG MONG MUỐN]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
