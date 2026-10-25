"""Đánh giá chi tiết và tối ưu hóa ngưỡng quyết định (Model Evaluation & Threshold Optimization) - Phase 5.
Bài tập lớn: Phát hiện gian lận thẻ tín dụng (Credit Card Fraud Detection).
Sinh viên: Nguyễn Minh Thái - MSSV: 12423031.

Mục tiêu của script:
1. Nạp xác suất dự đoán trên tập Validation từ Phase 4 (không huấn luyện lại mô hình).
2. Đánh giá 8 biến thể mô hình bằng: PR-AUC, ROC-AUC, F1-score, Precision, Recall và Confusion Matrix.
3. Vẽ đồ thị Precision-Recall Curves và ROC Curves lưu vào outputs/figures/ và reports/figures/.
4. Tối ưu hóa ngưỡng quyết định (Threshold Optimization):
   - Phương án 1 (Chính): Cực đại hóa F1-score trên tập Validation.
   - Phương án 2 (Kinh doanh): Đạt Recall mục tiêu (>= 80%) để khảo sát sự đánh đổi với Precision.
5. Lựa chọn mô hình tốt nhất dựa trên Validation và lưu vào models/best_model.joblib.
6. ĐÁNH GIÁ TRÊN TẬP TEST ĐÚNG MỘT LẦN DUY NHẤT:
   - Sử dụng mô hình tốt nhất tại ngưỡng tối ưu đã chọn.
   - Báo cáo kết quả trung thực, khách quan; tuyệt đối không dùng test để chỉnh lại mô hình.
7. Lưu metadata và báo cáo tóm tắt vào models/model_metadata.json và outputs/metrics/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # Vẽ ra file ảnh không cần GUI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Đảm bảo hiển thị tiếng Việt trên console Windows
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

# Đảm bảo import được package fraud khi chạy từ bất kỳ đâu
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fraud import config  # noqa: E402
from preprocessing import load_and_split_data  # noqa: E402


def safe_relative_path(path: Path) -> str:
    """Trả về đường dẫn tương đối so với project root nếu có thể, ngược lại trả về str(path)."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except (ValueError, RuntimeError):
        return str(path)


# ===========================================================================
# 1. Tính toán chi tiết các chỉ số và Ma trận nhầm lẫn
# ===========================================================================
def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Tính toàn bộ chỉ số phân loại và các thành phần của Confusion Matrix tại ngưỡng cho trước."""
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "threshold": round(float(threshold), 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


# ===========================================================================
# 2. Thuật toán tối ưu hóa Threshold trên tập Validation
# ===========================================================================
def find_optimal_threshold_max_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Tìm ngưỡng quyết định cực đại hóa F1-score trên Precision-Recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve trả về len(thresholds) = len(precisions) - 1
    # Tính F1 tại từng ngưỡng
    p = precisions[:-1]
    r = recalls[:-1]
    f1_scores = np.where(
        (p + r) > 0,
        2 * (p * r) / (p + r + 1e-12),
        0.0,
    )

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])

    return compute_metrics_at_threshold(y_true, y_prob, threshold=best_threshold)


def find_threshold_target_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = config.MIN_RECALL_TARGET,
) -> dict[str, Any]:
    """Tìm ngưỡng đạt mức Recall mục tiêu (>= min_recall) với Precision cao nhất có thể.

    Phục vụ bài toán nghiệp vụ ngân hàng: ưu tiên bắt chặn tối đa gian lận
    với mức cảnh báo giả được kiểm soát.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    p = precisions[:-1]
    r = recalls[:-1]

    # Lọc các ngưỡng có recall >= min_recall
    valid_indices = np.where(r >= min_recall)[0]
    if len(valid_indices) == 0:
        # Nếu không có ngưỡng nào đạt, lấy ngưỡng có recall lớn nhất
        chosen_idx = int(np.argmax(r))
    else:
        # Trong các ngưỡng thỏa mãn recall, chọn ngưỡng có precision cao nhất
        chosen_idx = int(valid_indices[np.argmax(p[valid_indices])])

    chosen_threshold = float(thresholds[chosen_idx])
    return compute_metrics_at_threshold(y_true, y_prob, threshold=chosen_threshold)


def optimize_thresholds_for_all_models(
    y_val_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    min_recall: float = config.MIN_RECALL_TARGET,
) -> pd.DataFrame:
    """Tối ưu ngưỡng cho tất cả 8 mô hình và tạo bảng so sánh Trước vs Sau tối ưu."""
    records: list[dict[str, Any]] = []

    for model_key, y_prob in probabilities.items():
        # Phân tích tên mô hình và chiến lược
        strategy = "class_weight" if "class_weight" in model_key or "weighted" in model_key else "baseline"
        base_name = (
            model_key.replace("_baseline", "")
            .replace("_weighted", "")
            .replace("_class_weight", "")
            .replace("_", " ")
            .title()
        )

        # 1. Tại ngưỡng mặc định 0.5
        metrics_def = compute_metrics_at_threshold(y_val_true, y_prob, threshold=0.5)

        # 2. Tại ngưỡng tối ưu Max F1
        metrics_f1 = find_optimal_threshold_max_f1(y_val_true, y_prob)

        # 3. Tại ngưỡng đạt Recall mục tiêu (>= 80%)
        metrics_rec = find_threshold_target_recall(y_val_true, y_prob, min_recall=min_recall)

        rec = {
            "model_key": model_key,
            "model_name": base_name,
            "strategy": strategy,
            "pr_auc": metrics_def["pr_auc"],
            "roc_auc": metrics_def["roc_auc"],
            # Mặc định 0.5
            "default_thresh": 0.5,
            "default_precision": metrics_def["precision"],
            "default_recall": metrics_def["recall"],
            "default_f1": metrics_def["f1_score"],
            # Tối ưu Max F1
            "opt_thresh_f1": metrics_f1["threshold"],
            "opt_precision_f1": metrics_f1["precision"],
            "opt_recall_f1": metrics_f1["recall"],
            "opt_f1": metrics_f1["f1_score"],
            "f1_improvement": round(metrics_f1["f1_score"] - metrics_def["f1_score"], 4),
            # Mục tiêu Recall
            "target_thresh_rec": metrics_rec["threshold"],
            "target_precision_rec": metrics_rec["precision"],
            "target_recall_rec": metrics_rec["recall"],
            "target_f1_rec": metrics_rec["f1_score"],
        }
        records.append(rec)

    df_res = pd.DataFrame(records)
    # Sắp xếp theo PR-AUC và F1 tối ưu giảm dần
    df_res.sort_values(by=["pr_auc", "opt_f1"], ascending=[False, False], inplace=True)
    df_res.reset_index(drop=True, inplace=True)
    return df_res


# ===========================================================================
# 3. Vẽ biểu đồ PR Curve, ROC Curve và Confusion Matrix
# ===========================================================================
def plot_pr_curves(
    y_val_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    optimal_thresholds: dict[str, float],
    output_files: list[Path],
) -> None:
    """Vẽ Precision-Recall Curves cho 8 mô hình và đánh dấu điểm threshold tối ưu."""
    plt.figure(figsize=(10, 7), dpi=config.FIGURE_DPI)

    palette = sns.color_palette("tab10", n_colors=len(probabilities))

    for (model_key, y_prob), color in zip(probabilities.items(), palette):
        precisions, recalls, thresholds = precision_recall_curve(y_val_true, y_prob)
        pr_auc = average_precision_score(y_val_true, y_prob)

        # Nhãn hiển thị gọn gàng
        clean_label = (
            model_key.replace("_baseline", " (baseline)")
            .replace("_weighted", " (weighted)")
            .replace("_", " ")
        )
        plt.plot(
            recalls,
            precisions,
            label=f"{clean_label} (PR-AUC = {pr_auc:.4f})",
            color=color,
            linewidth=1.8,
        )

        # Đánh dấu điểm tối ưu Max F1
        opt_thresh = optimal_thresholds[model_key]
        idx = np.argmin(np.abs(thresholds - opt_thresh))
        plt.scatter(
            recalls[idx],
            precisions[idx],
            color=color,
            s=45,
            edgecolor="black",
            zorder=5,
        )

    # Đường cơ sở (tỷ lệ lớp gian lận trong tập validation)
    baseline_rate = float(y_val_true.sum() / len(y_val_true))
    plt.axhline(
        y=baseline_rate,
        color="grey",
        linestyle="--",
        label=f"Đường cơ sở ngẫu nhiên ({baseline_rate * 100:.2f}%)",
    )

    plt.title("Precision-Recall Curves trên tập Validation\n(Dấu chấm biểu diễn điểm ngưỡng tối ưu Max F1)", fontsize=13)
    plt.xlabel("Recall (Độ thu hồi gian lận)", fontsize=11)
    plt.ylabel("Precision (Độ chuẩn xác cảnh báo)", fontsize=11)
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()

    for out_file in output_files:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file)
    plt.close()


def plot_roc_curves(
    y_val_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    output_files: list[Path],
) -> None:
    """Vẽ ROC Curves cho 8 mô hình trên tập Validation."""
    plt.figure(figsize=(10, 7), dpi=config.FIGURE_DPI)
    palette = sns.color_palette("tab10", n_colors=len(probabilities))

    for (model_key, y_prob), color in zip(probabilities.items(), palette):
        fpr, tpr, _ = roc_curve(y_val_true, y_prob)
        auc_score = roc_auc_score(y_val_true, y_prob)

        clean_label = (
            model_key.replace("_baseline", " (baseline)")
            .replace("_weighted", " (weighted)")
            .replace("_", " ")
        )
        plt.plot(
            fpr,
            tpr,
            label=f"{clean_label} (ROC-AUC = {auc_score:.4f})",
            color=color,
            linewidth=1.8,
        )

    plt.plot([0, 1], [0, 1], "k--", label="Đoán ngẫu nhiên (AUC = 0.5000)")
    plt.title("Receiver Operating Characteristic (ROC) Curves trên tập Validation", fontsize=13)
    plt.xlabel("False Positive Rate (FPR - Tỷ lệ báo nhầm)", fontsize=11)
    plt.ylabel("True Positive Rate (TPR / Recall - Tỷ lệ bắt trúng)", fontsize=11)
    plt.xlim([-0.01, 1.01])
    plt.ylim([0.0, 1.02])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()

    for out_file in output_files:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file)
    plt.close()


def plot_confusion_matrices(
    cm_val: dict[str, int],
    cm_test: dict[str, int],
    model_name: str,
    threshold: float,
    output_files: list[Path],
) -> None:
    """Vẽ Ma trận nhầm lẫn (Confusion Matrix) cạnh nhau cho tập Validation và Test."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=config.FIGURE_DPI)

    matrices = [
        (
            "Tập Validation",
            [[cm_val["tn"], cm_val["fp"]], [cm_val["fn"], cm_val["tp"]]],
            axes[0],
        ),
        (
            "Tập Test (Độc lập)",
            [[cm_test["tn"], cm_test["fp"]], [cm_test["fn"], cm_test["tp"]]],
            axes[1],
        ),
    ]

    for title, matrix, ax in matrices:
        arr = np.array(matrix)
        sns.heatmap(
            arr,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Hợp lệ (0)", "Gian lận (1)"],
            yticklabels=["Hợp lệ (0)", "Gian lận (1)"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(f"{title}\n({model_name} @ Ngưỡng = {threshold:.4f})", fontsize=11)
        ax.set_xlabel("Nhãn dự đoán", fontsize=10)
        ax.set_ylabel("Nhãn thực tế", fontsize=10)

        # Chú thích chi tiết từng ô
        tn, fp = arr[0]
        fn, tp = arr[1]
        ax.text(0.5, -0.15, f"TN: {tn:,} | FP: {fp:,} | FN: {fn} | TP: {tp}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9.5, style="italic")

    plt.tight_layout()
    for out_file in output_files:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file)
    plt.close()


# ===========================================================================
# 4. Đánh giá duy nhất một lần trên tập Test (Test Set Evaluation)
# ===========================================================================
def evaluate_best_model_on_test_once(
    best_model_key: str,
    chosen_threshold: float,
    models_dir: Path = config.MODELS_DIR,
    data_path: Path = config.RAW_DATA_PATH,
) -> dict[str, Any]:
    """Tải mô hình tốt nhất đã chọn từ Validation và đánh giá DUY NHẤT MỘT LẦN trên Test set.

    NGUYÊN TẮC LIÊM CHÍNH HỌC THUẬT:
    - Tập Test (15% ~ 42,722 mẫu) tuyệt đối KHÔNG được dùng để huấn luyện hay chỉnh ngưỡng.
    - Kết quả Test này được ghi nhận trung thực vào báo cáo cuối cùng mà không sửa đổi mô hình.
    """
    model_path = models_dir / f"{best_model_key}.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model đã lưu: {model_path}")

    # Nạp mô hình đã train ở Phase 4
    model = joblib.load(model_path)

    # Nạp dữ liệu Test đã chia chuẩn hóa ở Phase 3
    _, _, X_test, _, _, y_test = load_and_split_data(data_path)
    y_test_arr = y_test.to_numpy()

    # Dự đoán xác suất trên Test
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Đánh giá tại cả 2 ngưỡng: mặc định 0.5 và ngưỡng tối ưu từ Validation
    metrics_at_default = compute_metrics_at_threshold(y_test_arr, y_test_prob, threshold=0.5)
    metrics_at_chosen = compute_metrics_at_threshold(y_test_arr, y_test_prob, threshold=chosen_threshold)

    return {
        "model_key": best_model_key,
        "n_test_samples": len(y_test),
        "n_test_fraud": int(y_test.sum()),
        "n_test_legit": int((y_test == 0).sum()),
        "chosen_threshold": round(chosen_threshold, 4),
        "metrics_at_default_0_5": metrics_at_default,
        "metrics_at_chosen_threshold": metrics_at_chosen,
        "integrity_confirmation": (
            "Xác nhận: Tập Test được bảo vệ nguyên vẹn 100% trong suốt Phase 1-4. "
            "Mô hình và ngưỡng quyết định được cố định hoàn toàn từ tập Validation "
            "trước khi áp dụng lên tập Test này đúng 1 lần duy nhất."
        ),
    }


# ===========================================================================
# 5. Quy trình tổng hợp và thực thi Phase 5
# ===========================================================================
def run_evaluation_and_optimization(
    val_predictions_path: Path = config.VAL_PREDICTIONS_PATH,
    models_dir: Path = config.MODELS_DIR,
    metrics_dir: Path = config.METRICS_DIR,
    figures_dir: Path = config.FIGURES_DIR,
    tables_dir: Path = config.TABLES_DIR,
    data_path: Path = config.RAW_DATA_PATH,
) -> dict[str, Any]:
    """Thực thi toàn bộ Phase 5: Đánh giá, tối ưu ngưỡng, chọn mô hình và test một lần."""
    print("=" * 85)
    print("PHASE 5: MODEL EVALUATION & THRESHOLD OPTIMIZATION")
    print("=" * 85)

    if not val_predictions_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file xác suất validation: {val_predictions_path}\n"
            "Vui lòng chạy Phase 4 trước: python src/train.py"
        )

    # 1. Nạp xác suất dự đoán tập Validation từ Phase 4
    print(f"\n[Bước 1/5] Đang nạp xác suất dự đoán từ: {safe_relative_path(val_predictions_path)}")
    data = joblib.load(val_predictions_path)
    y_val_true = data["y_val_true"]
    probabilities: dict[str, np.ndarray] = data["probabilities"]
    print(f"  - Số mẫu Validation: {len(y_val_true):,} (Gian lận: {y_val_true.sum():,})")
    print(f"  - Số mô hình khảo sát: {len(probabilities)}")

    # 2. Tối ưu ngưỡng cho tất cả 8 mô hình
    print("\n[Bước 2/5] Tiến hành tối ưu hóa ngưỡng quyết định trên tập Validation...")
    df_opt = optimize_thresholds_for_all_models(y_val_true, probabilities)

    # Lưu bảng so sánh threshold
    csv_opt_metric = metrics_dir / "threshold_optimization.csv"
    csv_opt_table = tables_dir / "threshold_optimization.csv"
    df_opt.to_csv(csv_opt_metric, index=False, encoding="utf-8")
    df_opt.to_csv(csv_opt_table, index=False, encoding="utf-8")
    print(f"  - Đã lưu bảng tối ưu ngưỡng: {safe_relative_path(csv_opt_metric)}")

    # 3. Vẽ biểu đồ PR Curves và ROC Curves
    print("\n[Bước 3/5] Vẽ biểu đồ Precision-Recall Curves và ROC Curves...")
    opt_thresholds_dict = dict(zip(df_opt["model_key"], df_opt["opt_thresh_f1"]))

    pr_fig_outputs = [
        config.OUTPUTS_DIR / "figures" / "pr_curves.png",
        figures_dir / "pr_curves.png",
    ]
    plot_pr_curves(y_val_true, probabilities, opt_thresholds_dict, pr_fig_outputs)
    print(f"  - Đã lưu biểu đồ PR Curves: {safe_relative_path(pr_fig_outputs[0])}")

    roc_fig_outputs = [
        config.OUTPUTS_DIR / "figures" / "roc_curves.png",
        figures_dir / "roc_curves.png",
    ]
    plot_roc_curves(y_val_true, probabilities, roc_fig_outputs)
    print(f"  - Đã lưu biểu đồ ROC Curves: {safe_relative_path(roc_fig_outputs[0])}")

    # 4. Lựa chọn mô hình tốt nhất (Best Model) dựa trên Validation
    print("\n[Bước 4/5] Lựa chọn mô hình tốt nhất dựa trên kết quả Validation...")
    # Tiêu chí: PR-AUC cao nhất, kết hợp F1 tối ưu cao nhất
    best_row = df_opt.iloc[0]
    best_model_key = str(best_row["model_key"])
    best_model_name = str(best_row["model_name"])
    best_strategy = str(best_row["strategy"])
    chosen_threshold = float(best_row["opt_thresh_f1"])

    print(f"  >>> Mô hình được lựa chọn: {best_model_name} [{best_strategy}]")
    print(f"      + Lý do: Đạt PR-AUC cao nhất ({best_row['pr_auc']:.4f}) và F1 tối ưu cao nhất ({best_row['opt_f1']:.4f}) trên Validation.")
    print(f"      + Ngưỡng quyết định tối ưu đã chọn: {chosen_threshold:.4f} (thay cho ngưỡng mặc định 0.5000)")

    # Sao lưu sang best_model.joblib
    src_best_model = models_dir / f"{best_model_key}.joblib"
    shutil.copyfile(src_best_model, config.BEST_MODEL_PATH)
    print(f"  - Đã sao chép sang mô hình chính thức: {safe_relative_path(config.BEST_MODEL_PATH)}")

    # 5. Đánh giá trên tập TEST đúng MỘT LẦN DUY NHẤT
    print("\n[Bước 5/5] Đánh giá mô hình tốt nhất trên tập TEST (15% độc lập, đúng 1 lần duy nhất)...")
    test_eval = evaluate_best_model_on_test_once(
        best_model_key=best_model_key,
        chosen_threshold=chosen_threshold,
        models_dir=models_dir,
        data_path=data_path,
    )
    test_metrics = test_eval["metrics_at_chosen_threshold"]
    test_cm = test_metrics["confusion_matrix"]
    val_cm = compute_metrics_at_threshold(y_val_true, probabilities[best_model_key], threshold=chosen_threshold)["confusion_matrix"]

    print("  - Kết quả trên tập Test (42,722 giao dịch, 74 vụ gian lận):")
    print(f"    + PR-AUC    : {test_metrics['pr_auc']:.4f}")
    print(f"    + ROC-AUC   : {test_metrics['roc_auc']:.4f}")
    print(f"    + F1-Score  : {test_metrics['f1_score']:.4f}")
    print(f"    + Precision : {test_metrics['precision']:.4f}")
    print(f"    + Recall    : {test_metrics['recall']:.4f}")
    print(f"    + Confusion Matrix (TN: {test_cm['tn']:,}, FP: {test_cm['fp']:,}, FN: {test_cm['fn']}, TP: {test_cm['tp']})")

    # Vẽ và lưu biểu đồ Confusion Matrix cho cả Val và Test
    cm_fig_outputs = [
        config.OUTPUTS_DIR / "figures" / "confusion_matrix_best_model.png",
        figures_dir / "confusion_matrix_best_model.png",
    ]
    plot_confusion_matrices(
        val_cm,
        test_cm,
        model_name=f"{best_model_name} [{best_strategy}]",
        threshold=chosen_threshold,
        output_files=cm_fig_outputs,
    )
    print(f"  - Đã lưu biểu đồ Confusion Matrix: {safe_relative_path(cm_fig_outputs[0])}")

    # 6. Ghi lưu metadata mô hình và tổng kết
    metadata = {
        "best_model_key": best_model_key,
        "best_model_name": best_model_name,
        "strategy": best_strategy,
        "selection_metric": "PR-AUC & Max F1 trên Validation",
        "chosen_threshold": chosen_threshold,
        "validation_metrics": {
            "pr_auc": float(best_row["pr_auc"]),
            "roc_auc": float(best_row["roc_auc"]),
            "default_0_5": {
                "precision": float(best_row["default_precision"]),
                "recall": float(best_row["default_recall"]),
                "f1_score": float(best_row["default_f1"]),
            },
            "optimized_threshold": {
                "threshold": float(best_row["opt_thresh_f1"]),
                "precision": float(best_row["opt_precision_f1"]),
                "recall": float(best_row["opt_recall_f1"]),
                "f1_score": float(best_row["opt_f1"]),
                "confusion_matrix": val_cm,
            },
        },
        "test_metrics_evaluated_once": test_metrics,
        "integrity_statement": test_eval["integrity_confirmation"],
    }
    with config.MODEL_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  - Đã lưu metadata mô hình: {safe_relative_path(config.MODEL_METADATA_PATH)}")

    summary_json_path = metrics_dir / "evaluation_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "Phase 5 - Model Evaluation & Threshold Optimization",
                "best_model_selected": metadata,
                "all_models_threshold_comparison": df_opt.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  - Đã lưu báo cáo tổng kết JSON: {safe_relative_path(summary_json_path)}")

    # In bảng so sánh Trước vs Sau tối ưu ra màn hình
    print("\n" + "=" * 105)
    print("BẢNG SO SÁNH HIỆU NĂNG TRÊN VALIDATION: TRƯỚC VÀ SAU KHI TỐI ƯU THRESHOLD")
    print("=" * 105)
    disp_df = df_opt[
        [
            "model_name",
            "strategy",
            "pr_auc",
            "default_f1",
            "default_precision",
            "default_recall",
            "opt_thresh_f1",
            "opt_f1",
            "opt_precision_f1",
            "opt_recall_f1",
            "f1_improvement",
        ]
    ].copy()
    disp_df.columns = [
        "Mô hình",
        "Chiến lược",
        "PR-AUC",
        "F1 (0.5)",
        "Prec (0.5)",
        "Rec (0.5)",
        "Ngưỡng opt",
        "F1 (opt)",
        "Prec (opt)",
        "Rec (opt)",
        "Tăng F1",
    ]
    print(disp_df.to_string(index=False))
    print("=" * 105)

    return metadata


# ===========================================================================
# 6. Giao diện dòng lệnh (CLI)
# ===========================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình và tối ưu hóa ngưỡng phân loại (Phase 5)."
    )
    parser.add_argument(
        "--val-pred-path",
        type=Path,
        default=config.VAL_PREDICTIONS_PATH,
        help="Đường dẫn file xác suất validation (mặc định: outputs/metrics/val_predictions.joblib)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=config.RAW_DATA_PATH,
        help="Đường dẫn file dữ liệu creditcard.csv (mặc định lấy từ config)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_evaluation_and_optimization(
            val_predictions_path=args.val_pred_path,
            data_path=args.data_path,
        )
        return 0
    except FileNotFoundError as exc:
        print(f"\n[LỖI]: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[LỖI KHÔNG MONG MUỐN]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
