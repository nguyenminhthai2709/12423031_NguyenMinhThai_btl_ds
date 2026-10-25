"""Tiền xử lý dữ liệu (Data Preprocessing) - Phase 3.
Bài tập lớn: Phát hiện gian lận thẻ tín dụng (Credit Card Fraud Detection).
Sinh viên: Nguyễn Minh Thái - MSSV: 12423031.

Mục tiêu của script:
1. Đọc và kiểm tra chất lượng dữ liệu (giá trị thiếu, trùng lặp, kiểu dữ liệu).
2. Tách đặc trưng X và nhãn y (cột Class).
3. Phân chia dữ liệu theo tỷ lệ 70% Train - 15% Validation - 15% Test (Stratified Random Split).
4. Thống kê phân phối lớp (hợp lệ vs gian lận) trên từng tập dữ liệu.
5. Chuẩn bị pipeline tiền xử lý: StandardScaler cho Logistic Regression, giữ nguyên cho mô hình cây.
   (Chỉ fit scaler trên tập Train để chống rò rỉ dữ liệu - Data Leakage).
6. Tính toán trọng số xử lý mất cân bằng lớp (scale_pos_weight cho XGBoost, class_weight cho sklearn).
7. Xuất báo cáo tóm tắt thực tế vào outputs/metrics/preprocessing_summary.json.

Cách chạy:
    python src/preprocessing.py
    python src/preprocessing.py --data-path data/raw/creditcard.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# Đảm bảo import được package fraud khi chạy trực tiếp từ thư mục bất kỳ
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(CURRENT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR.parent))

from fraud import config  # noqa: E402


# ===========================================================================
# 1. Đọc dữ liệu và kiểm tra chất lượng
# ===========================================================================
def download_instructions(data_path: Path) -> str:
    """Thông báo hướng dẫn tải dữ liệu khi không tìm thấy file."""
    return (
        f"Không tìm thấy file dữ liệu: {data_path}\n\n"
        "Hướng dẫn tải dataset từ Kaggle (KHÔNG dùng dữ liệu giả lập):\n"
        "  1. Truy cập: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        "  2. Tải về và giải nén file 'creditcard.csv'.\n"
        f"  3. Đặt file vào thư mục: {config.RAW_DATA_DIR}\n"
        "  Hoặc chỉ định đường dẫn khi chạy: python src/preprocessing.py --data-path <đường_dẫn>\n"
    )


def load_data(data_path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Đọc file CSV và kiểm tra tính hợp lệ cơ bản của dữ liệu."""
    if not data_path.is_file():
        raise FileNotFoundError(download_instructions(data_path))
    if data_path.stat().st_size == 0:
        raise ValueError(f"File dữ liệu rỗng (0 byte): {data_path}")

    df = pd.read_csv(data_path)
    return df


def check_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Kiểm tra giá trị thiếu (missing), dòng trùng lặp (duplicates) và kiểu dữ liệu (dtypes).

    Lưu ý: Không tự ý xóa các dòng trùng lặp vì trong dữ liệu giao dịch thẻ (với các biến
    PCA ẩn danh V1-V28), các giao dịch có thể có số tiền và đặc trưng tương đồng nhưng là
    giao dịch thực tế độc lập; việc xóa dòng khi chưa có căn cứ nghiệp vụ có thể làm sai lệch
    tỷ lệ lớp thiểu số quý giá.
    """
    total_missing = int(df.isnull().sum().sum())
    missing_by_col = {
        col: int(cnt) for col, cnt in df.isnull().sum().items() if cnt > 0
    }

    n_duplicates = int(df.duplicated().sum())

    # Kiểm tra phân phối nhãn trong các dòng trùng lặp (nếu có)
    dup_class_distribution: dict[str, int] = {}
    if n_duplicates > 0:
        dup_series = df[df.duplicated(keep=False)][config.TARGET_COL].value_counts()
        dup_class_distribution = {str(k): int(v) for k, v in dup_series.items()}

    dtypes_count = {str(k): int(v) for k, v in df.dtypes.value_counts().items()}

    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "total_missing_values": total_missing,
        "columns_with_missing": missing_by_col,
        "has_missing_values": total_missing > 0,
        "n_duplicate_rows": n_duplicates,
        "duplicate_class_distribution": dup_class_distribution,
        "dtypes_summary": dtypes_count,
        "handling_decision": (
            "Giữ nguyên dữ liệu gốc: không xóa các dòng trùng lặp và không áp dụng thêm PCA "
            "vì V1-V28 đã qua chuẩn hóa PCA; không có giá trị khuyết thiếu cần điền."
        ),
    }


# ===========================================================================
# 2. Tách đặc trưng X và nhãn y
# ===========================================================================
def separate_features_and_target(
    df: pd.DataFrame, target_col: str = config.TARGET_COL
) -> tuple[pd.DataFrame, pd.Series]:
    """Tách tập dữ liệu thành ma trận đặc trưng X và vector nhãn y."""
    if target_col not in df.columns:
        raise KeyError(f"Cột nhãn '{target_col}' không tồn tại trong dữ liệu.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


# ===========================================================================
# 3. Phân chia Train - Validation - Test (Stratified 70 / 15 / 15)
# ===========================================================================
def calculate_class_distribution(y: pd.Series, split_name: str = "") -> dict[str, Any]:
    """Thống kê số lượng, tỷ lệ phần trăm và tỷ lệ mất cân bằng giữa hai lớp."""
    counts = y.value_counts().to_dict()
    n_legit = int(counts.get(0, 0))
    n_fraud = int(counts.get(1, 0))
    total = len(y)

    pct_legit = float((n_legit / total) * 100) if total > 0 else 0.0
    pct_fraud = float((n_fraud / total) * 100) if total > 0 else 0.0
    imbalance_ratio = float(n_legit / n_fraud) if n_fraud > 0 else 0.0

    return {
        "name": split_name,
        "total_samples": total,
        "counts": {"0_legitimate": n_legit, "1_fraud": n_fraud},
        "percentage": {
            "0_legitimate": round(pct_legit, 4),
            "1_fraud": round(pct_fraud, 4),
        },
        "imbalance_ratio_legit_per_fraud": round(imbalance_ratio, 2),
    }


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = config.TRAIN_SIZE,
    val_size: float = config.VAL_SIZE,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_SEED,
) -> dict[str, Any]:
    """Chia dữ liệu thành 3 tập: Train (70%), Validation (15%), Test (15%).

    Sử dụng kỹ thuật chia 2 bước với `stratify=y` để giữ nguyên tỷ lệ gian lận:
    - Bước 1: Tách test_size (15%) khỏi toàn bộ dữ liệu -> thu được (Train+Val) và Test.
    - Bước 2: Tách val_size (15% trên tổng ban đầu) từ (Train+Val) -> thu được Train và Val.
      Tỷ lệ validation trên tập con = val_size / (train_size + val_size).
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Tổng tỷ lệ các tập phải bằng 1.0, hiện tại là: {total}")

    # Bước 1: Tách tập Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Bước 2: Tách tập Validation từ (Train + Val)
    # Tỷ lệ của tập Val trên tập (Train + Val) là 0.15 / (0.70 + 0.15) ≈ 17.65%
    val_relative_size = val_size / (train_size + val_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        stratify=y_train_val,
        random_state=random_state,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "split_ratios": {
            "train": train_size,
            "validation": val_size,
            "test": test_size,
        },
    }


# ===========================================================================
# 4. Chuẩn bị Tiền xử lý (Scaling & Pipeline) cho 4 mô hình
# ===========================================================================
def build_preprocessor(
    scaled_columns: list[str] = config.SCALED_COLUMNS,
) -> ColumnTransformer:
    """Tạo ColumnTransformer chuẩn hóa các cột số học chưa qua PCA.

    - Cột 'Time' và 'Amount' có biên độ biến thiên rất lớn (Time lên tới ~172,000,
      Amount lên tới ~25,000) nên cần áp dụng `StandardScaler`.
    - Các cột 'V1' đến 'V28' vốn đã là kết quả của PCA (trung bình xấp xỉ 0, độ lệch chuẩn xấp xỉ 1),
      nên được giữ nguyên thông qua `remainder='passthrough'`.
    """
    return ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), scaled_columns),
        ],
        remainder="passthrough",
    )


def create_model_pipeline(
    model_name: str,
    estimator: Any = None,
    scaled_columns: list[str] = config.SCALED_COLUMNS,
) -> Pipeline | Any:
    """Tạo Pipeline tiền xử lý kết hợp mô hình.

    Quy tắc nghiệp vụ:
    - Logistic Regression: Mô hình tuyến tính rất nhạy cảm với thang đo -> bắt buộc chuẩn hóa.
    - Decision Tree, Random Forest, XGBoost: Mô hình cây phân chia theo ngưỡng (rule-based split),
      bất biến với phép co giãn đơn điệu (monotonic transformation) -> không bắt buộc chuẩn hóa.
    """
    if model_name == "logistic_regression":
        preprocessor = build_preprocessor(scaled_columns)
        if estimator is not None:
            return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        return Pipeline([("preprocessor", preprocessor)])

    # Các mô hình cây có thể nhận trực tiếp dữ liệu thô hoặc pipeline passthrough
    if estimator is not None:
        return Pipeline([("classifier", estimator)])
    return None


def fit_and_verify_scaling(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict[str, Any]:
    """Fit scaler CHỈ trên tập Train, sau đó transform sang Val và Test.

    QUY TẮC CHỐNG RÒ RỈ DỮ LIỆU (DATA LEAKAGE):
    - Scaler tính trung bình (mean) và phương sai (std) duy nhất trên tập Train.
    - Tuyệt đối không dùng thông tin từ Validation hoặc Test trong quá trình tính mean/std.
    """
    # Fit CHỈ trên train
    preprocessor.fit(X_train)

    # Lấy thông số mean và std của các cột được scale
    scaler: StandardScaler = preprocessor.named_transformers_["scaler"]
    scaled_stats = {
        col: {
            "mean": float(mean),
            "std": float(scale),
        }
        for col, mean, scale in zip(
            config.SCALED_COLUMNS, scaler.mean_, scaler.scale_
        )
    }

    # Transform thử nghiệm để kiểm chứng
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    return {
        "scaler_fitted_on": "X_train_only",
        "scaled_columns": config.SCALED_COLUMNS,
        "fitted_parameters": scaled_stats,
        "transformed_shapes": {
            "train": list(X_train_trans.shape),
            "val": list(X_val_trans.shape),
            "test": list(X_test_trans.shape),
        },
    }


# ===========================================================================
# 5. Xử lý mất cân bằng lớp (Imbalance Weights)
# ===========================================================================
def calculate_imbalance_weights(y_train: pd.Series) -> dict[str, Any]:
    """Tính toán các tham số trọng số lớp phục vụ huấn luyện (Phase 4).

    Lưu ý:
    - Chưa áp dụng các phương pháp tái lấy mẫu (như SMOTE) trong giai đoạn này để giữ nguyên
      phân phối tự nhiên của tập dữ liệu và không làm biến dạng tập kiểm định/kiểm thử.
    - Đối với XGBoost: tham số `scale_pos_weight = n_negative / n_positive` (chỉ tính trên tập train).
    - Đối với Scikit-learn (Logistic Regression, Random Forest): tham số `class_weight='balanced'`
      tự động gán trọng số nghịch đảo với tần suất xuất hiện của lớp.
    """
    counts = y_train.value_counts().to_dict()
    n_neg = int(counts.get(0, 0))
    n_pos = int(counts.get(1, 0))

    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    # Trọng số tính theo chuẩn 'balanced' của sklearn: n_samples / (n_classes * n_samples_c)
    total_samples = len(y_train)
    weight_0 = float(total_samples / (2.0 * n_neg)) if n_neg > 0 else 1.0
    weight_1 = float(total_samples / (2.0 * n_pos)) if n_pos > 0 else 1.0

    return {
        "computed_on": "y_train_only",
        "train_samples": {
            "n_negative": n_neg,
            "n_positive": n_pos,
        },
        "xgboost": {
            "parameter": "scale_pos_weight",
            "value": round(scale_pos_weight, 4),
            "formula": "n_negative / n_positive (tính trên tập Train)",
        },
        "sklearn": {
            "parameter": "class_weight='balanced'",
            "effective_weights": {
                "0_legitimate": round(weight_0, 4),
                "1_fraud": round(weight_1, 4),
            },
        },
    }


# ===========================================================================
# 6. Tiện ích xuất dữ liệu và tóm tắt
# ===========================================================================
def to_jsonable(obj: Any) -> Any:
    """Chuyển numpy/pandas sang kiểu dữ liệu JSON chuẩn."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    return obj


def print_section(title: str) -> None:
    """In tiêu đề mục phân cách rõ ràng trên console."""
    print(f"\n{'=' * 75}\n{title}\n{'=' * 75}")


def run_preprocessing(
    data_path: Path = config.RAW_DATA_PATH,
    output_summary_path: Path = config.PREPROCESSING_SUMMARY_PATH,
) -> dict[str, Any]:
    """Thực thi toàn bộ quy trình tiền xử lý Phase 3 và lưu báo cáo tóm tắt."""
    print_section("GIAI ĐOẠN 3: DATA PREPROCESSING (TIỀN XỬ LÝ DỮ LIỆU)")
    print(f"File dữ liệu đầu vào : {data_path}")
    print(f"File tóm tắt đầu ra  : {output_summary_path}")

    # 1. Đọc dữ liệu
    print_section("1. ĐỌC DỮ LIỆU VÀ KIỂM TRA CHẤT LƯỢNG")
    df = load_data(data_path)
    print(f"Đã nạp thành công: {len(df):,} dòng, {len(df.columns)} cột.")

    quality = check_data_quality(df)
    print(f"  - Giá trị khuyết thiếu (Missing values): {quality['total_missing_values']}")
    print(f"  - Dòng dữ liệu trùng lặp (Duplicates)  : {quality['n_duplicate_rows']:,}")
    if quality["duplicate_class_distribution"]:
        print(f"    Phân bố dòng trùng theo nhãn: {quality['duplicate_class_distribution']}")
    print(f"  - Kiểu dữ liệu (Dtypes)                : {quality['dtypes_summary']}")
    print(f"  -> Quyết định: {quality['handling_decision']}")

    # 2. Tách đặc trưng X và nhãn y
    print_section("2. TÁCH ĐẶC TRƯNG X VÀ NHÃN Y")
    X, y = separate_features_and_target(df)
    print(f"  - Ma trận đặc trưng X: {X.shape[0]:,} dòng, {X.shape[1]} cột ({', '.join(X.columns[:3])}... {X.columns[-1]})")
    print(f"  - Vector nhãn y       : {len(y):,} dòng (cột '{config.TARGET_COL}')")

    # 3. Phân chia Train - Validation - Test (70% - 15% - 15%)
    print_section("3. PHÂN CHIA TẬP DỮ LIỆU (STRATIFIED SPLIT 70% / 15% / 15%)")
    split_res = split_data(X, y)
    X_train, y_train = split_res["X_train"], split_res["y_train"]
    X_val, y_val = split_res["X_val"], split_res["y_val"]
    X_test, y_test = split_res["X_test"], split_res["y_test"]

    dist_full = calculate_class_distribution(y, "Toàn bộ dữ liệu (Full)")
    dist_train = calculate_class_distribution(y_train, "Tập Huấn luyện (Train - 70%)")
    dist_val = calculate_class_distribution(y_val, "Tập Kiểm định (Validation - 15%)")
    dist_test = calculate_class_distribution(y_test, "Tập Kiểm thử (Test - 15%)")

    for dist in [dist_full, dist_train, dist_val, dist_test]:
        print(f"\n[{dist['name']}] Tổng: {dist['total_samples']:,} mẫu")
        print(f"  - Hợp lệ (0)  : {dist['counts']['0_legitimate']:,} ({dist['percentage']['0_legitimate']}%)")
        print(f"  - Gian lận (1): {dist['counts']['1_fraud']:,} ({dist['percentage']['1_fraud']}%)")
        print(f"  - Tỷ lệ 0 / 1 : {dist['imbalance_ratio_legit_per_fraud']:.1f} : 1")

    # 4. Chuẩn bị Tiền xử lý & Tránh rò rỉ dữ liệu
    print_section("4. THIẾT LẬP CHUẨN HÓA (STANDARDSCALER) & PHÒNG CHỐNG DATA LEAKAGE")
    preprocessor = build_preprocessor()
    scaling_info = fit_and_verify_scaling(preprocessor, X_train, X_val, X_test)
    print("  - Phương pháp chuẩn hóa: StandardScaler áp dụng cho ['Time', 'Amount']")
    print("  - Các đặc trưng PCA (V1-V28): Giữ nguyên (passthrough)")
    print("  - Nguyên tắc chống rò rỉ: Scaler CHỈ được fit trên X_train")
    print(f"  - Tham số học được trên Train:")
    for col, stats in scaling_info["fitted_parameters"].items():
        print(f"    + {col:6s}: mean = {stats['mean']:10.2f}, std = {stats['std']:10.2f}")

    # 5. Tính toán trọng số xử lý mất cân bằng
    print_section("5. CHUẨN BỊ TRỌNG SỐ MẤT CÂN BẰNG LỚP (IMBALANCE WEIGHTS)")
    weights_info = calculate_imbalance_weights(y_train)
    print(f"  - Tính toán dựa trên: {weights_info['computed_on']}")
    print(f"  - XGBoost (scale_pos_weight) = {weights_info['xgboost']['value']} "
          f"({weights_info['xgboost']['formula']})")
    print(f"  - Scikit-learn (class_weight='balanced'):")
    print(f"    + Trọng số lớp 0 (Hợp lệ): {weights_info['sklearn']['effective_weights']['0_legitimate']}")
    print(f"    + Trọng số lớp 1 (Gian lận): {weights_info['sklearn']['effective_weights']['1_fraud']}")

    # 6. Tổng hợp và lưu tóm tắt
    print_section("6. LƯU BÁO CÁO TÓM TẮT THỰC TẾ")
    summary = {
        "dataset_info": {
            "file_name": data_path.name,
            "file_size_bytes": int(data_path.stat().st_size),
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "features_count": X.shape[1],
            "feature_names": list(X.columns),
            "target_name": config.TARGET_COL,
        },
        "data_quality": quality,
        "split_configuration": {
            "random_seed": config.RANDOM_SEED,
            "train_ratio": config.TRAIN_SIZE,
            "val_ratio": config.VAL_SIZE,
            "test_ratio": config.TEST_SIZE,
            "strategy": "Stratified random split (train_test_split 2 bước)",
            "roles_explanation": {
                "train_70pct": "Dùng để huấn luyện mô hình học quy luật và fit các bộ chuẩn hóa (scaler).",
                "validation_15pct": "Dùng để tinh chỉnh siêu tham số, so sánh giữa các thuật toán và tìm ngưỡng quyết định (threshold tuning) tối ưu F1/Recall.",
                "test_15pct": "Dữ liệu kiểm thử độc lập đóng vai trò tương lai chưa từng thấy, chỉ đánh giá 1 lần duy nhất để báo cáo hiệu năng tổng quát.",
            },
            "temporal_split_limitation_note": (
                "Hạn chế của chia ngẫu nhiên khi có cột Time: Giao dịch thẻ mang tính chất chuỗi thời gian liên tục. "
                "Khi chia ngẫu nhiên (dù có stratify), dữ liệu tương lai có thể lọt vào tập train và dữ liệu quá khứ rơi vào val/test, "
                "gây ra hiện tượng Temporal Data Leakage (nhìn trước tương lai). Trong ứng dụng thực tế ngân hàng, "
                "mô hình chỉ có thể học từ quá khứ và dự đoán cho tương lai."
            ),
        },
        "class_distributions": {
            "full_dataset": dist_full,
            "train_set": dist_train,
            "validation_set": dist_val,
            "test_set": dist_test,
        },
        "preprocessing_pipeline": {
            "scaled_columns": config.SCALED_COLUMNS,
            "passthrough_columns": config.PCA_FEATURES,
            "scaler_type": "StandardScaler",
            "model_specific_strategies": {
                "logistic_regression": "Bắt buộc chuẩn hóa qua Pipeline với StandardScaler (do nhạy cảm với thang đo lớn của Time và Amount).",
                "decision_tree": "Không bắt buộc chuẩn hóa (thuật toán cây bất biến với phép biến đổi đơn điệu).",
                "random_forest": "Không bắt buộc chuẩn hóa (tương tự cây quyết định).",
                "xgboost": "Không bắt buộc chuẩn hóa (cây tăng cường dựa trên phân chia ngưỡng đơn điệu).",
            },
            "leakage_prevention": "Chỉ gọi scaler.fit() trên X_train. X_val và X_test chỉ được transform().",
            "fitted_scaler_parameters": scaling_info["fitted_parameters"],
        },
        "imbalance_handling": {
            "smote_applied": False,
            "reason_for_no_smote": "Giữ nguyên phân phối tự nhiên thực tế của tập validation và test; chuẩn bị dùng trọng số lớp (class weights) để chống thiên vị lớp đa số mà không tạo mẫu tổng hợp giả.",
            "weights": weights_info,
        },
    }

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output_summary_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, ensure_ascii=False, indent=2)

    print(f"Đã lưu thành công tóm tắt vào: {output_summary_path}")
    print("\n[HOÀN TẤT GIAI ĐOẠN 3] Dữ liệu và pipeline đã sẵn sàng cho giai đoạn 4 (Huấn luyện mô hình).")
    return summary


# ===========================================================================
# 7. Hàm tiện ích cho các module sau nạp dữ liệu nhanh
# ===========================================================================
def load_and_split_data(
    data_path: Path = config.RAW_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Hàm tiện ích dành cho Phase 4 & 5 để lấy nhanh các tập dữ liệu đã chia đúng tỷ lệ."""
    df = load_data(data_path)
    X, y = separate_features_and_target(df)
    splits = split_data(X, y)
    return (
        splits["X_train"],
        splits["X_val"],
        splits["X_test"],
        splits["y_train"],
        splits["y_val"],
        splits["y_test"],
    )


# ===========================================================================
# 8. Giao diện dòng lệnh (CLI)
# ===========================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiền xử lý dữ liệu cho bài toán Credit Card Fraud Detection."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=config.RAW_DATA_PATH,
        help="Đường dẫn tới file creditcard.csv (mặc định lấy từ config)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=config.PREPROCESSING_SUMMARY_PATH,
        help="Đường dẫn lưu file JSON tóm tắt (mặc định: outputs/metrics/preprocessing_summary.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_preprocessing(data_path=args.data_path, output_summary_path=args.output_path)
        return 0
    except FileNotFoundError as exc:
        print(f"\n[LỖI]: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[LỖI KHÔNG MONG MUỐN]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
