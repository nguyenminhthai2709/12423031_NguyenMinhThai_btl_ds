"""Phân tích khám phá dữ liệu (EDA) cho bài toán Credit Card Fraud Detection.

Script này CHỈ khám phá và mô tả dữ liệu. Nó không huấn luyện mô hình, không
xóa dòng, không điền giá trị thiếu và không sửa file dữ liệu gốc.

Mọi con số trong kết quả đều được TÍNH TRỰC TIẾP từ file creditcard.csv thật;
không có số liệu nào được ghi sẵn trong code.

Cách chạy (từ thư mục gốc của dự án, PowerShell):
    python src/eda.py
    python src/eda.py --data-path D:\\datasets\\creditcard.csv
    python src/eda.py --output-dir outputs_thu_nghiem

Đầu ra:
    outputs/figures/*.png            các biểu đồ EDA
    outputs/metrics/eda_summary.json tóm tắt toàn bộ kết quả EDA
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # vẽ ra file, không cần cửa sổ giao diện (chạy được trên mọi máy)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from fraud import config  # noqa: E402  (src/ nằm sẵn trong sys.path khi chạy `python src/eda.py`)

# ---------------------------------------------------------------------------
# Hằng số dùng chung cho việc vẽ
# ---------------------------------------------------------------------------
CLASS_LABELS = {0: "Hợp lệ (0)", 1: "Gian lận (1)"}
CLASS_COLORS = {0: "#4C72B0", 1: "#C44E52"}  # xanh = hợp lệ, đỏ = gian lận (nhất quán ở mọi biểu đồ)
TIME_BIN_SECONDS = 2 * 3600  # chia trục thời gian thành các khoảng 2 giờ
OUTLIER_IQR_FACTOR = 1.5  # quy tắc Tukey: ngoài [Q1 - 1.5*IQR, Q3 + 1.5*IQR]


# ===========================================================================
# 1. Đọc dữ liệu và kiểm tra schema
# ===========================================================================
def download_instructions(data_path: Path) -> str:
    """Thông báo hướng dẫn tải dữ liệu khi không tìm thấy file."""
    return (
        f"Không tìm thấy file dữ liệu: {data_path}\n\n"
        "Cách lấy dataset thật từ Kaggle (KHÔNG tạo dữ liệu giả thay thế):\n"
        "  Cách A - tải thủ công:\n"
        "    1. Đăng nhập Kaggle, mở https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        "    2. Bấm Download, giải nén file zip.\n"
        f"    3. Đặt file creditcard.csv vào: {config.RAW_DATA_DIR}\n"
        "  Cách B - Kaggle CLI (token để NGOÀI repo, ở %USERPROFILE%\\.kaggle\\kaggle.json):\n"
        "    pip install kaggle\n"
        "    kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip\n"
        "  Hoặc đặt file ở nơi khác rồi chạy: python src/eda.py --data-path <đường dẫn>\n"
    )


def load_data(data_path: Path) -> pd.DataFrame:
    """Đọc file CSV; báo lỗi rõ ràng nếu file không tồn tại hoặc rỗng."""
    if not data_path.is_file():
        raise FileNotFoundError(download_instructions(data_path))
    if data_path.stat().st_size == 0:
        raise ValueError(f"File dữ liệu rỗng (0 byte): {data_path}")
    return pd.read_csv(data_path)


def check_schema(df: pd.DataFrame) -> dict:
    """So sánh cột thực tế với schema kỳ vọng trong config (chỉ báo cáo, không sửa dữ liệu)."""
    expected = config.EXPECTED_COLUMNS
    target = config.TARGET_COL
    target_values = sorted(df[target].dropna().unique().tolist()) if target in df.columns else None
    return {
        "missing_expected_columns": [c for c in expected if c not in df.columns],
        "unexpected_columns": [c for c in df.columns if c not in expected],
        "same_order_as_expected": list(df.columns) == expected,
        "non_numeric_columns": [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])],
        "target_unique_values": target_values,
        "target_is_binary_0_1": target_values is not None and set(target_values) <= {0, 1},
    }


# ===========================================================================
# 2. Tổng quan dữ liệu
# ===========================================================================
def describe_overview(df: pd.DataFrame) -> dict:
    """Số dòng, số cột, tên cột, kiểu dữ liệu, bộ nhớ và thống kê mô tả."""
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "describe": df.describe().T.to_dict(orient="index"),
    }


# ===========================================================================
# 3. Chất lượng dữ liệu (chỉ báo cáo, KHÔNG xóa/sửa gì)
# ===========================================================================
def iqr_outlier_percent(series: pd.Series) -> float:
    """Tỷ lệ % giá trị nằm ngoài khoảng Tukey [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:  # cột gần như hằng số: định nghĩa IQR không còn ý nghĩa
        return float("nan")
    lower, upper = q1 - OUTLIER_IQR_FACTOR * iqr, q3 + OUTLIER_IQR_FACTOR * iqr
    return float(((series < lower) | (series > upper)).mean() * 100)


def data_quality_report(df: pd.DataFrame) -> dict:
    """Giá trị thiếu, dòng trùng lặp, giá trị vô cực/âm/bằng 0 và ngoại lai."""
    target = config.TARGET_COL
    numeric = df.select_dtypes(include="number")

    missing = df.isna().sum()
    duplicated_mask = df.duplicated()  # mặc định: giữ lần xuất hiện đầu, đánh dấu các bản sao sau
    duplicates_by_class = (
        df.loc[duplicated_mask, target].value_counts().sort_index().to_dict()
        if target in df.columns
        else {}
    )

    report = {
        "total_missing_values": int(missing.sum()),
        "columns_with_missing": {c: int(v) for c, v in missing.items() if v > 0},
        "n_duplicate_rows": int(duplicated_mask.sum()),
        "duplicate_rows_percent": float(duplicated_mask.mean() * 100),
        "duplicate_rows_by_class": {str(k): int(v) for k, v in duplicates_by_class.items()},
        "n_infinite_values": int(np.isinf(numeric).sum().sum()),
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
    }

    # Kiểm tra thêm cho Amount và Time
    if config.AMOUNT_COL in df.columns:
        amount = df[config.AMOUNT_COL]
        report["n_negative_amount"] = int((amount < 0).sum())
        report["n_zero_amount"] = int((amount == 0).sum())
        report["amount_iqr_outlier_percent"] = iqr_outlier_percent(amount)
        if target in df.columns:
            report["amount_iqr_outlier_percent_by_class"] = {
                str(k): iqr_outlier_percent(g[config.AMOUNT_COL])
                for k, g in df.groupby(target)
            }
    if config.TIME_COL in df.columns:
        time = df[config.TIME_COL]
        report["n_negative_time"] = int((time < 0).sum())
        # Quan trọng cho việc chia dữ liệu theo thời gian ở các giai đoạn sau
        report["rows_sorted_by_time"] = bool(time.is_monotonic_increasing)

    # Ngoại lai theo quy tắc IQR cho các đặc trưng (không phải nhãn)
    feature_cols = [c for c in numeric.columns if c != target]
    outlier_pct = pd.Series({c: iqr_outlier_percent(df[c]) for c in feature_cols}).dropna()
    report["top_10_features_by_iqr_outlier_percent"] = (
        outlier_pct.sort_values(ascending=False).head(10).round(4).to_dict()
    )
    return report


# ===========================================================================
# 4. Phân phối lớp (mất cân bằng)
# ===========================================================================
def class_distribution(df: pd.DataFrame) -> dict:
    """Số lượng, tỷ lệ % từng lớp, tỷ lệ mất cân bằng và Accuracy của mô hình 'luôn đoán lớp đa số'."""
    counts = df[config.TARGET_COL].value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    n_legit, n_fraud = int(counts[0]), int(counts[1])
    return {
        "counts": {"0": n_legit, "1": n_fraud},
        "percent": {"0": n_legit / total * 100, "1": n_fraud / total * 100},
        "imbalance_ratio_legit_per_fraud": (n_legit / n_fraud) if n_fraud else None,
        # Accuracy nếu mô hình luôn dự đoán lớp chiếm đa số (tính từ dữ liệu, không ghi sẵn)
        "majority_class_baseline_accuracy": max(n_legit, n_fraud) / total,
    }


def plot_class_distribution(dist: dict, out_path: Path) -> None:
    """Vẽ 2 biểu đồ cột cạnh nhau: thang tuyến tính (thấy rõ chênh lệch) và thang log (thấy rõ lớp nhỏ)."""
    classes = [0, 1]
    counts = [dist["counts"][str(c)] for c in classes]
    percents = [dist["percent"][str(c)] for c in classes]
    labels = [CLASS_LABELS[c] for c in classes]
    colors = [CLASS_COLORS[c] for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, use_log in zip(axes, (False, True)):
        bars = ax.bar(labels, counts, color=colors)
        for bar, n, pct in zip(bars, counts, percents):
            ax.annotate(
                f"{n:,}\n({pct:.3f}%)",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        if use_log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1, top=max(counts) * 12)  # đáy = 1 giao dịch, để chiều cao cột có nghĩa
            ax.set_title("Thang logarit (nhìn rõ lớp thiểu số)")
        else:
            ax.set_ylim(top=max(counts) * 1.2)
            ax.set_title("Thang tuyến tính (nhìn rõ độ chênh lệch)")
        ax.set_ylabel("Số giao dịch")
    fig.suptitle("Phân phối hai lớp giao dịch", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.FIGURE_DPI)
    plt.close(fig)


# ===========================================================================
# 5. Phân tích Amount
# ===========================================================================
def describe_series(series: pd.Series) -> dict:
    """Các thống kê chính của một cột số."""
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "q25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "q75": float(series.quantile(0.75)),
        "q95": float(series.quantile(0.95)),
        "q99": float(series.quantile(0.99)),
        "max": float(series.max()),
        "skewness": float(series.skew()),
    }


def amount_statistics(df: pd.DataFrame) -> dict:
    """Thống kê Amount toàn bộ và theo từng lớp."""
    amount_col, target = config.AMOUNT_COL, config.TARGET_COL
    return {
        "overall": describe_series(df[amount_col]),
        "by_class": {str(k): describe_series(g[amount_col]) for k, g in df.groupby(target)},
    }


def plot_amount_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Phân phối Amount: dữ liệu gốc (trục y log) và sau biến đổi log(1 + Amount)."""
    amount = df[config.AMOUNT_COL]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].hist(amount, bins=60, color="#55A868")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Amount")
    axes[0].set_ylabel("Số giao dịch (thang log)")
    axes[0].set_title("Amount gốc")

    axes[1].hist(np.log1p(amount.clip(lower=0)), bins=60, color="#55A868")
    axes[1].set_xlabel("log(1 + Amount)")
    axes[1].set_ylabel("Số giao dịch")
    axes[1].set_title("Amount sau biến đổi log")

    fig.suptitle("Phân phối số tiền giao dịch (Amount)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.FIGURE_DPI)
    plt.close(fig)


def plot_amount_by_class(df: pd.DataFrame, out_path: Path) -> None:
    """So sánh Amount giữa hai lớp: boxplot và histogram mật độ (đã chuẩn hóa theo từng lớp)."""
    data = pd.DataFrame(
        {
            "log1p_amount": np.log1p(df[config.AMOUNT_COL].clip(lower=0)),
            "Lớp": df[config.TARGET_COL].map(CLASS_LABELS),
        }
    )
    palette = {CLASS_LABELS[c]: CLASS_COLORS[c] for c in (0, 1)}
    order = [CLASS_LABELS[0], CLASS_LABELS[1]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sns.boxplot(
        data=data, x="Lớp", y="log1p_amount", hue="Lớp", order=order, hue_order=order,
        palette=palette, legend=False, ax=axes[0], fliersize=1.5,
    )
    axes[0].set_ylabel("log(1 + Amount)")
    axes[0].set_xlabel("")
    axes[0].set_title("Boxplot")

    # common_norm=False: mỗi lớp được chuẩn hóa riêng, nếu không lớp gian lận sẽ gần như vô hình
    sns.histplot(
        data=data, x="log1p_amount", hue="Lớp", hue_order=order, palette=palette,
        stat="density", common_norm=False, bins=50, element="step", ax=axes[1],
    )
    axes[1].set_xlabel("log(1 + Amount)")
    axes[1].set_ylabel("Mật độ")
    axes[1].set_title("Mật độ (mỗi lớp chuẩn hóa riêng)")

    fig.suptitle("So sánh số tiền giao dịch giữa hai lớp", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.FIGURE_DPI)
    plt.close(fig)


# ===========================================================================
# 6. Phân tích Time
# ===========================================================================
def time_statistics(df: pd.DataFrame) -> dict:
    """Thống kê Time (giây) và tỷ lệ gian lận theo từng khoảng thời gian."""
    time_col, target = config.TIME_COL, config.TARGET_COL
    time = df[time_col]

    max_time = float(time.max())
    # Số khoảng chọn sao cho mốc cuối LUÔN lớn hơn max_time (tránh rơi ra ngoài khoảng cuối)
    n_bins = int(max_time // TIME_BIN_SECONDS) + 1
    edges = np.arange(n_bins + 1) * TIME_BIN_SECONDS
    bins = pd.cut(time, bins=edges, right=False, include_lowest=True)
    grouped = df.groupby(bins, observed=False)[target].agg(["size", "sum"])

    fraud_rate_by_bin = [
        {
            "bin_start_hour": float(interval.left / 3600),
            "bin_end_hour": float(interval.right / 3600),
            "n_transactions": int(row["size"]),
            "n_fraud": int(row["sum"]),
            "fraud_rate_percent": float(row["sum"] / row["size"] * 100) if row["size"] else None,
        }
        for interval, row in grouped.iterrows()
    ]
    return {
        "overall_seconds": describe_series(time),
        "span_hours": max_time / 3600,
        "by_class_seconds": {str(k): describe_series(g[time_col]) for k, g in df.groupby(target)},
        "fraud_rate_by_2h_bin": fraud_rate_by_bin,
    }


def plot_time_by_class(df: pd.DataFrame, time_stats: dict, out_path: Path) -> None:
    """Trái: mật độ Time theo lớp. Phải: tỷ lệ gian lận (%) theo từng khoảng 2 giờ."""
    data = pd.DataFrame(
        {
            "hour": df[config.TIME_COL] / 3600,
            "Lớp": df[config.TARGET_COL].map(CLASS_LABELS),
        }
    )
    palette = {CLASS_LABELS[c]: CLASS_COLORS[c] for c in (0, 1)}
    order = [CLASS_LABELS[0], CLASS_LABELS[1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    sns.histplot(
        data=data, x="hour", hue="Lớp", hue_order=order, palette=palette,
        stat="density", common_norm=False, bins=48, element="step", ax=axes[0],
    )
    axes[0].set_xlabel("Time (giờ kể từ giao dịch đầu tiên của dataset)")
    axes[0].set_ylabel("Mật độ")
    axes[0].set_title("Mật độ theo thời gian (mỗi lớp chuẩn hóa riêng)")

    bins = time_stats["fraud_rate_by_2h_bin"]
    starts = [b["bin_start_hour"] for b in bins]
    rates = [b["fraud_rate_percent"] or 0.0 for b in bins]
    axes[1].bar(starts, rates, width=1.8, align="edge", color=CLASS_COLORS[1])
    axes[1].set_xlabel("Time (giờ kể từ giao dịch đầu tiên của dataset)")
    axes[1].set_ylabel("Tỷ lệ gian lận (%)")
    axes[1].set_title("Tỷ lệ gian lận theo từng khoảng 2 giờ")

    fig.suptitle("Phân tích Time theo hai lớp", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.FIGURE_DPI)
    plt.close(fig)


# ===========================================================================
# 7. Tương quan
# ===========================================================================
def correlation_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ma trận tương quan Pearson, tương quan với Class và cặp V-V tương quan mạnh nhất."""
    target = config.TARGET_COL
    corr = df.corr(numeric_only=True)

    with_class = corr[target].drop(target)
    top_with_class = with_class.reindex(with_class.abs().sort_values(ascending=False).index).head(10)

    pca_cols = [c for c in config.PCA_FEATURES if c in corr.columns]
    pca_corr = corr.loc[pca_cols, pca_cols].to_numpy()
    upper = np.triu_indices_from(pca_corr, k=1)
    pair_values = np.abs(pca_corr[upper])
    summary = {
        "top_10_abs_correlation_with_class": top_with_class.round(4).to_dict(),
        "max_abs_correlation_between_pca_features": float(pair_values.max()) if pair_values.size else None,
        "n_pca_pairs_with_abs_corr_above_0.5": int((pair_values > 0.5).sum()),
    }
    if pair_values.size:
        k = int(pair_values.argmax())
        summary["most_correlated_pca_pair"] = [pca_cols[upper[0][k]], pca_cols[upper[1][k]]]
    return corr, summary


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    """Heatmap tương quan giữa tất cả các cột (gồm cả Class)."""
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True,
        xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.7}, ax=ax,
    )
    ax.tick_params(labelsize=7)
    ax.set_title("Ma trận tương quan Pearson", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.FIGURE_DPI)
    plt.close(fig)


def plot_correlation_with_class(corr: pd.DataFrame, out_path: Path) -> None:
    """Thanh ngang: tương quan của từng đặc trưng với Class (sắp xếp tăng dần)."""
    with_class = corr[config.TARGET_COL].drop(config.TARGET_COL).sort_values()
    colors = [CLASS_COLORS[1] if v > 0 else CLASS_COLORS[0] for v in with_class]

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(with_class.index, with_class.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hệ số tương quan Pearson với Class")
    ax.set_title("Tương quan của từng đặc trưng với Class")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=config.FIGURE_DPI)
    plt.close(fig)


# ===========================================================================
# 8. Lưu kết quả và in báo cáo
# ===========================================================================
def to_jsonable(obj):
    """Chuyển numpy/pandas sang kiểu Python thuần; NaN/inf -> None (JSON chuẩn không có NaN)."""
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
        value = float(obj)
        return value if np.isfinite(value) else None
    return obj


def save_summary(summary: dict, out_path: Path) -> None:
    """Ghi tóm tắt EDA ra file JSON (UTF-8)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, ensure_ascii=False, indent=2)


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def display_path(path: Path) -> str:
    """Hiển thị đường dẫn tương đối so với thư mục gốc dự án (không lộ đường dẫn cá nhân)."""
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_eda(data_path: Path, figures_dir: Path, summary_path: Path) -> dict:
    """Chạy toàn bộ quy trình EDA, trả về dict tóm tắt (đồng thời lưu biểu đồ và JSON)."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Đọc dữ liệu -------------------------------------------------------
    df = load_data(data_path)

    print_section("1. TỔNG QUAN DỮ LIỆU")
    print("5 dòng đầu tiên:")
    print(df.head())
    overview = describe_overview(df)
    print(f"\nSố dòng: {overview['n_rows']:,} | Số cột: {overview['n_columns']}")
    print(f"Tên cột: {', '.join(overview['columns'])}")
    print("\nKiểu dữ liệu:")
    print(df.dtypes.to_string())
    print("\nThống kê mô tả:")
    print(df.describe().T)

    schema = check_schema(df)
    print("\nKiểm tra schema so với kỳ vọng:")
    for key, value in schema.items():
        print(f"  - {key}: {value}")
    if schema["missing_expected_columns"] or not schema["target_is_binary_0_1"]:
        raise ValueError(
            "Schema không đúng kỳ vọng (thiếu cột hoặc Class không phải 0/1) - "
            "hãy kiểm tra lại file dữ liệu, xem chi tiết ở phần trên."
        )

    # --- Chất lượng dữ liệu ----------------------------------------------
    print_section("2. CHẤT LƯỢNG DỮ LIỆU (chỉ báo cáo, không sửa dữ liệu)")
    quality = data_quality_report(df)
    for key, value in quality.items():
        print(f"  - {key}: {value}")

    # --- Mất cân bằng lớp -------------------------------------------------
    print_section("3. PHÂN PHỐI LỚP")
    dist = class_distribution(df)
    if dist["counts"]["0"] == 0 or dist["counts"]["1"] == 0:
        raise ValueError("Dữ liệu chỉ có một lớp, không thể phân tích bài toán phân loại nhị phân.")
    print(f"  Hợp lệ (0)  : {dist['counts']['0']:,} ({dist['percent']['0']:.4f}%)")
    print(f"  Gian lận (1): {dist['counts']['1']:,} ({dist['percent']['1']:.4f}%)")
    print(f"  Tỷ lệ hợp lệ : gian lận = {dist['imbalance_ratio_legit_per_fraud']:.1f} : 1")
    print(f"  Accuracy nếu luôn đoán 'hợp lệ' = {dist['majority_class_baseline_accuracy']:.6f}")
    plot_class_distribution(dist, figures_dir / "class_distribution.png")

    # --- Amount -------------------------------------------------------------
    print_section("4. PHÂN TÍCH AMOUNT")
    amount_stats = amount_statistics(df)
    print(pd.DataFrame({"overall": amount_stats["overall"],
                        **{f"class {k}": v for k, v in amount_stats["by_class"].items()}}))
    plot_amount_distribution(df, figures_dir / "amount_distribution.png")
    plot_amount_by_class(df, figures_dir / "amount_by_class.png")

    # --- Time ---------------------------------------------------------------
    print_section("5. PHÂN TÍCH TIME")
    time_stats = time_statistics(df)
    print(f"  Khoảng thời gian bao phủ: {time_stats['span_hours']:.2f} giờ")
    print(f"  Dữ liệu đã sắp xếp theo Time: {quality.get('rows_sorted_by_time')}")
    plot_time_by_class(df, time_stats, figures_dir / "time_by_class.png")

    # --- Tương quan ----------------------------------------------------------
    print_section("6. TƯƠNG QUAN")
    corr, corr_summary = correlation_analysis(df)
    for key, value in corr_summary.items():
        print(f"  - {key}: {value}")
    plot_correlation_heatmap(corr, figures_dir / "correlation_heatmap.png")
    plot_correlation_with_class(corr, figures_dir / "correlation_with_class.png")

    # --- Lưu tóm tắt ---------------------------------------------------------
    summary = {
        "data_file": {"name": data_path.name, "size_bytes": int(data_path.stat().st_size)},
        "libraries": {"pandas": pd.__version__, "numpy": np.__version__,
                      "matplotlib": matplotlib.__version__, "seaborn": sns.__version__},
        "overview": overview,
        "schema_check": schema,
        "data_quality": quality,
        "class_distribution": dist,
        "amount": amount_stats,
        "time": time_stats,
        "correlation": corr_summary,
    }
    save_summary(summary, summary_path)

    print_section("HOÀN TẤT")
    print(f"  Biểu đồ : {display_path(figures_dir)}")
    for fig_path in sorted(figures_dir.glob("*.png")):
        print(f"    - {fig_path.name}")
    print(f"  Tóm tắt : {display_path(summary_path)}")
    return summary


# ===========================================================================
# 9. Chạy từ dòng lệnh
# ===========================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA cho dataset Credit Card Fraud Detection.")
    parser.add_argument("--data-path", type=Path, default=config.RAW_DATA_PATH,
                        help="đường dẫn tới creditcard.csv (mặc định lấy từ config)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="thư mục gốc cho đầu ra; sẽ tạo figures/ và metrics/ bên trong "
                             "(mặc định: outputs/)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir is None:
        figures_dir, summary_path = config.EDA_FIGURES_DIR, config.EDA_SUMMARY_PATH
    else:
        figures_dir = args.output_dir / "figures"
        summary_path = args.output_dir / "metrics" / "eda_summary.json"

    try:
        run_eda(args.data_path, figures_dir, summary_path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)  # đã chứa hướng dẫn tải dữ liệu
        return 1
    except ValueError as exc:
        print(f"LỖI DỮ LIỆU: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
