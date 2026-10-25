"""Ứng dụng Web Demo Phát hiện Gian lận Thẻ Tín dụng (Streamlit App) - Phase 6.
Bài tập lớn: Phát hiện gian lận thẻ tín dụng (Credit Card Fraud Detection).
Sinh viên thực hiện: Nguyễn Minh Thái - MSSV: 12423031.

Các chức năng chính:
1. Tổng quan dữ liệu: Phân tích kích thước, chất lượng dữ liệu, tỷ lệ mất cân bằng và phân chia tập.
2. Biểu đồ EDA: Hiển thị 6 biểu đồ phân tích khám phá dữ liệu từ Phase 2.
3. So sánh 4 mô hình: Hiển thị bảng so sánh thực tế, tối ưu hóa ngưỡng, PR Curves, ROC Curves và Confusion Matrix.
4. Dự đoán 1 giao dịch: Nhập thủ công hoặc chọn mẫu thực tế, dự đoán xác suất và phân loại theo ngưỡng 0.5701.
5. Dự đoán hàng loạt từ CSV: Tải file CSV lên, kiểm tra schema, dự đoán hàng loạt và tải file kết quả.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Thiết lập đường dẫn import
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud import config  # noqa: E402


# ===========================================================================
# 1. Cấu hình trang và nạp tài nguyên (Cache Resource / Cache Data)
# ===========================================================================
st.set_page_config(
    page_title="Hệ thống Phát hiện Gian lận Thẻ Tín dụng",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_trained_model() -> tuple[Any | None, dict[str, Any] | None, str | None]:
    """Nạp mô hình tốt nhất và metadata; không nạp lại khi người dùng tương tác."""
    model_path = config.BEST_MODEL_PATH
    meta_path = config.MODEL_METADATA_PATH

    if not model_path.is_file():
        return None, None, f"Không tìm thấy file mô hình tại `{model_path}`. Vui lòng chạy Phase 4 & 5 trước."
    if not meta_path.is_file():
        return None, None, f"Không tìm thấy file metadata tại `{meta_path}`. Vui lòng chạy Phase 5 trước."

    try:
        model = joblib.load(model_path)
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return model, metadata, None
    except Exception as exc:
        return None, None, f"Lỗi khi nạp mô hình: {exc}"


@st.cache_data
def load_metrics_data() -> dict[str, Any]:
    """Nạp các bảng số liệu và báo cáo JSON đã lưu từ các Phase trước."""
    data: dict[str, Any] = {}

    # 1. Preprocessing Summary
    prep_path = config.PREPROCESSING_SUMMARY_PATH
    if prep_path.is_file():
        with prep_path.open("r", encoding="utf-8") as f:
            data["preprocessing"] = json.load(f)

    # 2. Model Comparison CSV
    comp_path = config.MODEL_COMPARISON_CSV
    if comp_path.is_file():
        data["comparison"] = pd.read_csv(comp_path)

    # 3. Threshold Optimization CSV
    thresh_path = config.THRESHOLD_OPTIMIZATION_CSV
    if thresh_path.is_file():
        data["threshold_optimization"] = pd.read_csv(thresh_path)

    # 4. Evaluation Summary
    eval_path = config.EVALUATION_SUMMARY_PATH
    if eval_path.is_file():
        with eval_path.open("r", encoding="utf-8") as f:
            data["evaluation"] = json.load(f)

    return data


# ===========================================================================
# Mẫu dữ liệu chuẩn trích xuất trực tiếp từ tập kiểm thử độc lập (Test Set)
# ===========================================================================
SAMPLE_LEGIT_TRANSACTION: dict[str, float] = {
    "Time": 148719.0,
    "V1": -2.249486,
    "V2": 0.081739,
    "V3": 0.441904,
    "V4": 1.721977,
    "V5": -0.180580,
    "V6": -0.706050,
    "V7": -0.312720,
    "V8": 0.690355,
    "V9": 0.233285,
    "V10": -0.439941,
    "V11": -1.256395,
    "V12": 0.783574,
    "V13": -0.075629,
    "V14": 0.197456,
    "V15": -0.969207,
    "V16": -0.735345,
    "V17": 0.710080,
    "V18": -0.764719,
    "V19": 1.147030,
    "V20": -1.164660,
    "V21": -0.516668,
    "V22": -0.579222,
    "V23": 1.407223,
    "V24": 0.336601,
    "V25": -0.806343,
    "V26": -1.031327,
    "V27": 0.054091,
    "V28": -0.113703,
    "Amount": 21.85,
}

SAMPLE_FRAUD_TRANSACTION: dict[str, float] = {
    "Time": 41743.0,
    "V1": -2.144411,
    "V2": 1.073499,
    "V3": -2.773663,
    "V4": 1.384394,
    "V5": -4.015477,
    "V6": 0.747235,
    "V7": 1.525638,
    "V8": 0.192988,
    "V9": -2.431862,
    "V10": -4.155838,
    "V11": 4.354775,
    "V12": -2.872400,
    "V13": 0.468875,
    "V14": -4.555667,
    "V15": 0.699671,
    "V16": -2.901141,
    "V17": -4.674667,
    "V18": -2.213623,
    "V19": 1.296165,
    "V20": 0.668124,
    "V21": 0.144653,
    "V22": -0.885682,
    "V23": 0.625099,
    "V24": 0.096527,
    "V25": -0.189455,
    "V26": 0.532908,
    "V27": 0.423045,
    "V28": -0.210266,
    "Amount": 802.52,
}


# ===========================================================================
# 2. Giao diện và Sidebar thông tin dự án
# ===========================================================================
model, metadata, model_err = load_trained_model()
metrics_store = load_metrics_data()

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/bank-card-back-side.png",
        width=80,
    )
    st.title("FraudShield AI")
    st.caption("Credit Card Fraud Detection System")
    st.markdown("---")

    st.markdown("### 🎓 Thông tin đồ án")
    st.markdown("**Sinh viên:** Nguyễn Minh Thái")
    st.markdown("**MSSV:** 12423031")
    st.markdown("**Môn học:** Khoa học Dữ liệu (Data Science)")

    st.markdown("---")
    st.markdown("### ⚙️ Trạng thái mô hình")
    if model_err:
        st.error(model_err)
        CHOSEN_THRESHOLD = 0.5701
        CHOSEN_MODEL_NAME = "Random Forest (baseline)"
    else:
        CHOSEN_THRESHOLD = float(metadata.get("chosen_threshold", 0.5701))
        CHOSEN_MODEL_NAME = f"{metadata.get('best_model_name', 'Random Forest')} ({metadata.get('strategy', 'baseline')})"
        st.success(f"**Mô hình:** {CHOSEN_MODEL_NAME}")
        st.info(f"**Ngưỡng tối ưu:** `{CHOSEN_THRESHOLD:.4f}`")
        test_f1 = metadata.get("test_metrics_evaluated_once", {}).get("f1_score", 0.8358)
        test_pr_auc = metadata.get("test_metrics_evaluated_once", {}).get("pr_auc", 0.8320)
        st.metric("Test F1-Score", f"{test_f1:.4f}")
        st.metric("Test PR-AUC", f"{test_pr_auc:.4f}")

    st.markdown("---")
    st.caption("© 2026 Nguyễn Minh Thái - BTL Khoa học Dữ liệu")


# Header chính
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 22px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <h1 style="margin: 0; font-size: 28px; color: white;">💳 Hệ Thống Phát Hiện Gian Lận Thẻ Tín Dụng</h1>
        <p style="margin: 5px 0 0 0; font-size: 15px; opacity: 0.9;">
            Ứng dụng Machine Learning xử lý dữ liệu mất cân bằng cực đoan & tối ưu hóa ngưỡng phân loại trong thời gian thực.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# 3. Các Tab chức năng chính
# ===========================================================================
tab_overview, tab_eda, tab_models, tab_predict_one, tab_predict_batch = st.tabs(
    [
        "📊 Tổng quan dữ liệu",
        "📈 Biểu đồ EDA",
        "🤖 So sánh 4 Mô hình & Đánh giá",
        "🔍 Dự đoán một giao dịch",
        "📁 Dự đoán hàng loạt từ CSV",
    ]
)


# ---------------------------------------------------------------------------
# TAB 1: Tổng quan dữ liệu
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("1. Tổng quan Bộ dữ liệu Credit Card Fraud Detection")
    st.markdown(
        "Dữ liệu gồm các giao dịch thẻ tín dụng của chủ thẻ châu Âu trong 2 ngày (tháng 9/2013). "
        "Bài toán có tính chất **mất cân bằng lớp cực đoan (Extreme Class Imbalance)**."
    )

    prep_data = metrics_store.get("preprocessing", {})
    dataset_info = prep_data.get("dataset_info", {})
    quality = prep_data.get("data_quality", {})
    splits = prep_data.get("split_configuration", {})
    class_dists = prep_data.get("class_distributions", {})

    # Thẻ KPI tổng quan
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Tổng số giao dịch", f"{dataset_info.get('n_rows', 284807):,}")
    kpi2.metric("Số lượng đặc trưng", f"{dataset_info.get('features_count', 30)}")
    kpi3.metric("Giá trị khuyết thiếu (Missing)", f"{quality.get('total_missing_values', 0)}")
    kpi4.metric("Dòng trùng lặp (Giữ nguyên)", f"{quality.get('n_duplicate_rows', 1081):,}")

    st.markdown("---")

    col_dist, col_split = st.columns([1, 1])

    with col_dist:
        st.markdown("#### ⚖️ Phân phối lớp (Class Imbalance)")
        full_dist = class_dists.get("full_dataset", {})
        counts = full_dist.get("counts", {"0_legitimate": 284315, "1_fraud": 492})
        pcts = full_dist.get("percentage", {"0_legitimate": 99.8273, "1_fraud": 0.1727})

        df_pie = pd.DataFrame(
            {
                "Loại giao dịch": ["Hợp lệ (0)", "Gian lận (1)"],
                "Số lượng": [counts.get("0_legitimate", 284315), counts.get("1_fraud", 492)],
                "Tỷ lệ (%)": [pcts.get("0_legitimate", 99.83), pcts.get("1_fraud", 0.17)],
            }
        )
        fig_pie = px.pie(
            df_pie,
            names="Loại giao dịch",
            values="Số lượng",
            color="Loại giao dịch",
            color_discrete_map={"Hợp lệ (0)": "#2b5c8f", "Gian lận (1)": "#d9534f"},
            hole=0.45,
            title="Tỷ lệ giao dịch Hợp lệ vs Gian lận (0.17% gian lận)",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.info(
            f"**Tỷ lệ mất cân bằng:** Cứ **{full_dist.get('imbalance_ratio_legit_per_fraud', 577.9):.1f}** giao dịch hợp lệ "
            "mới có 1 giao dịch gian lận. Do đó, độ chính xác (Accuracy) không phản ánh đúng hiệu quả của mô hình!"
        )

    with col_split:
        st.markdown("#### 🔀 Phân chia dữ liệu (Stratified Split 70% / 15% / 15%)")
        split_table = pd.DataFrame(
            [
                {
                    "Tập dữ liệu": "Train (Huấn luyện)",
                    "Tỷ lệ": "70%",
                    "Số mẫu": f"{class_dists.get('train_set', {}).get('total_samples', 199364):,}",
                    "Hợp lệ (0)": f"{class_dists.get('train_set', {}).get('counts', {}).get('0_legitimate', 199020):,}",
                    "Gian lận (1)": f"{class_dists.get('train_set', {}).get('counts', {}).get('1_fraud', 344):,}",
                    "Tỷ lệ gian lận": f"{class_dists.get('train_set', {}).get('percentage', {}).get('1_fraud', 0.1725):.4f}%",
                },
                {
                    "Tập dữ liệu": "Validation (Kiểm định)",
                    "Tỷ lệ": "15%",
                    "Số mẫu": f"{class_dists.get('validation_set', {}).get('total_samples', 42721):,}",
                    "Hợp lệ (0)": f"{class_dists.get('validation_set', {}).get('counts', {}).get('0_legitimate', 42647):,}",
                    "Gian lận (1)": f"{class_dists.get('validation_set', {}).get('counts', {}).get('1_fraud', 74):,}",
                    "Tỷ lệ gian lận": f"{class_dists.get('validation_set', {}).get('percentage', {}).get('1_fraud', 0.1732):.4f}%",
                },
                {
                    "Tập dữ liệu": "Test (Kiểm thử độc lập)",
                    "Tỷ lệ": "15%",
                    "Số mẫu": f"{class_dists.get('test_set', {}).get('total_samples', 42722):,}",
                    "Hợp lệ (0)": f"{class_dists.get('test_set', {}).get('counts', {}).get('0_legitimate', 42648):,}",
                    "Gian lận (1)": f"{class_dists.get('test_set', {}).get('counts', {}).get('1_fraud', 74):,}",
                    "Tỷ lệ gian lận": f"{class_dists.get('test_set', {}).get('percentage', {}).get('1_fraud', 0.1732):.4f}%",
                },
            ]
        )
        st.dataframe(split_table, use_container_width=True, hide_index=True)

        st.warning(
            "**Nguyên tắc bảo vệ tập Test:** Tập Test (15%) được cô lập hoàn toàn trong suốt quá trình huấn luyện "
            "và lựa chọn threshold; chỉ được đánh giá đúng một lần duy nhất ở cuối Phase 5."
        )


# ---------------------------------------------------------------------------
# TAB 2: Biểu đồ EDA
# ---------------------------------------------------------------------------
with tab_eda:
    st.subheader("2. Khám phá & Trực quan hóa Dữ liệu (EDA Visualizations)")
    st.markdown("Các biểu đồ được sinh tự động từ Phase 2 dựa trên 284,807 giao dịch thực tế.")

    eda_tab_sel = st.radio(
        "Chọn chủ đề phân tích:",
        ["Phân phối Số tiền (Amount)", "Phân phối Thời gian (Time)", "Tương quan Đặc trưng (Correlation)"],
        horizontal=True,
    )

    fig_dir = config.OUTPUTS_DIR / "figures"

    if eda_tab_sel == "Phân phối Số tiền (Amount)":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Phân phối Số tiền Giao dịch (Tổng quan)")
            img1 = fig_dir / "amount_distribution.png"
            if img1.is_file():
                st.image(str(img1), use_container_width=True)
            else:
                st.warning("Chưa có ảnh amount_distribution.png.")
            st.caption("Số tiền giao dịch tập trung nhiều ở giá trị nhỏ (< 100 USD), có đuôi dài lệch phải mạnh.")
        with c2:
            st.markdown("##### Phân phối Số tiền theo Nhãn (Hợp lệ vs Gian lận)")
            img2 = fig_dir / "amount_by_class.png"
            if img2.is_file():
                st.image(str(img2), use_container_width=True)
            else:
                st.warning("Chưa có ảnh amount_by_class.png.")
            st.caption("Giao dịch gian lận có giá trị trung bình ~122 USD nhưng không có các giao dịch cực đại > 3,000 USD.")

    elif eda_tab_sel == "Phân phối Thời gian (Time)":
        st.markdown("##### Tỷ lệ Gian lận theo các Khoảng Thời gian (2 giờ)")
        img3 = fig_dir / "time_by_class.png"
        if img3.is_file():
            st.image(str(img3), use_container_width=True)
        else:
            st.warning("Chưa có ảnh time_by_class.png.")
        st.caption("Thời gian giao dịch trải dài 48 giờ. Tỷ lệ gian lận tăng đột biến vào các khung giờ đêm (ít giao dịch hợp lệ).")

    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Ma trận Tương quan Đặc trưng PCA")
            img4 = fig_dir / "correlation_heatmap.png"
            if img4.is_file():
                st.image(str(img4), use_container_width=True)
            else:
                st.warning("Chưa có ảnh correlation_heatmap.png.")
            st.caption("Các đặc trưng V1-V28 độc lập tuyến tính với nhau do là thành phần chính từ PCA.")
        with c2:
            st.markdown("##### Tương quan giữa Đặc trưng và Biến Mục tiêu (Class)")
            img5 = fig_dir / "correlation_with_class.png"
            if img5.is_file():
                st.image(str(img5), use_container_width=True)
            else:
                st.warning("Chưa có ảnh correlation_with_class.png.")
            st.caption("Các biến V17, V14, V12 có tương quan nghịch mạnh nhất; V11, V4 có tương quan thuận rõ rệt nhất với gian lận.")


# ---------------------------------------------------------------------------
# TAB 3: So sánh 4 Mô hình & Đánh giá
# ---------------------------------------------------------------------------
with tab_models:
    st.subheader("3. So sánh 4 Mô hình Học máy & Tối ưu hóa Ngưỡng (Threshold)")
    st.markdown(
        "So sánh 4 thuật toán: **Logistic Regression, Decision Tree, Random Forest, XGBoost** "
        "trên 2 chiến lược: `baseline` (không trọng số) và `class_weight` (có gán trọng số bù mất cân bằng)."
    )

    df_comp = metrics_store.get("comparison")
    df_opt = metrics_store.get("threshold_optimization")

    if df_comp is not None and not df_comp.empty:
        st.markdown("#### 📋 Bảng So sánh 8 Biến thể Mô hình (Tại ngưỡng mặc định 0.5)")
        st.dataframe(
            df_comp[
                ["model_name", "strategy", "pr_auc", "f1_score", "precision", "recall", "roc_auc", "fit_time_seconds"]
            ].rename(
                columns={
                    "model_name": "Mô hình",
                    "strategy": "Chiến lược",
                    "pr_auc": "PR-AUC",
                    "f1_score": "F1-Score",
                    "precision": "Precision",
                    "recall": "Recall",
                    "roc_auc": "ROC-AUC",
                    "fit_time_seconds": "Thời gian fit (s)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    if df_opt is not None and not df_opt.empty:
        st.markdown("#### 🎯 Bảng Tối ưu hóa Ngưỡng Quyết định trên tập Validation")
        st.dataframe(
            df_opt[
                [
                    "model_name",
                    "strategy",
                    "pr_auc",
                    "default_f1",
                    "opt_thresh_f1",
                    "opt_f1",
                    "opt_precision_f1",
                    "opt_recall_f1",
                    "f1_improvement",
                ]
            ].rename(
                columns={
                    "model_name": "Mô hình",
                    "strategy": "Chiến lược",
                    "pr_auc": "PR-AUC",
                    "default_f1": "F1 (ngưỡng 0.5)",
                    "opt_thresh_f1": "Ngưỡng tối ưu",
                    "opt_f1": "F1 tối ưu",
                    "opt_precision_f1": "Precision (opt)",
                    "opt_recall_f1": "Recall (opt)",
                    "f1_improvement": "Mức tăng F1",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("#### 📈 Đường cong Precision-Recall & Ma trận Nhầm lẫn")

    c_pr, c_roc = st.columns(2)
    with c_pr:
        img_pr = fig_dir / "pr_curves.png"
        if img_pr.is_file():
            st.image(str(img_pr), caption="Precision-Recall Curves (Đánh dấu điểm tối ưu Max F1)", use_container_width=True)
        else:
            st.info("Chưa có biểu đồ PR curves.")

    with c_roc:
        img_roc = fig_dir / "roc_curves.png"
        if img_roc.is_file():
            st.image(str(img_roc), caption="ROC Curves trên tập Validation", use_container_width=True)
        else:
            st.info("Chưa có biểu đồ ROC curves.")

    st.markdown("---")
    st.markdown("#### 🏆 Kết quả Kiểm thử của Mô hình Được chọn trên tập Test Độc lập")
    col_cm, col_summary = st.columns([1.2, 1])

    with col_cm:
        img_cm = fig_dir / "confusion_matrix_best_model.png"
        if img_cm.is_file():
            st.image(str(img_cm), caption="Ma trận nhầm lẫn Validation vs Test (Random Forest @ 0.5701)", use_container_width=True)
        else:
            st.info("Chưa có biểu đồ Confusion Matrix.")

    with col_summary:
        st.success("### 🥇 Mô hình được lựa chọn: Random Forest (baseline)")
        st.markdown(
            """
            - **Ngưỡng tối ưu đã chọn:** `0.5701` (dựa trên Max F1 của tập Validation).
            - **Kết quả thực tế trên tập Test độc lập (42,722 giao dịch):**
                - **PR-AUC:** `0.8320`
                - **F1-Score:** `0.8358` (rất sát với Validation 0.8676 $\rightarrow$ không Overfit).
                - **Precision:** `93.33%` (93.33% cảnh báo là gian lận thật).
                - **Recall:** `75.68%` (Bắt trúng **56 / 74** vụ gian lận).
                - **False Positives (Báo nhầm):** Chỉ **4** trường hợp trên 42,648 giao dịch hợp lệ!
            """
        )


# ---------------------------------------------------------------------------
# TAB 4: Dự đoán một giao dịch
# ---------------------------------------------------------------------------
with tab_predict_one:
    st.subheader("4. Dự đoán Trực tuyến cho Một Giao dịch (Single Prediction)")
    st.markdown(
        "Nhập thông tin giao dịch hoặc chọn mẫu thử thực tế để mô hình phân loại gian lận theo ngưỡng tối ưu **0.5701**."
    )

    if model is None:
        st.error("Chưa nạp được mô hình. Vui lòng kiểm tra lại file `models/best_model.joblib`.")
    else:
        # Tải mẫu từ test set nếu có
        sample_choice = st.selectbox(
            "Chọn kịch bản kiểm thử nhanh:",
            [
                "Mẫu 1: Giao dịch HỢP LỆ thực tế (Từ tập kiểm thử)",
                "Mẫu 2: Giao dịch GIAN LẬN thực tế (Từ tập kiểm thử)",
                "Mẫu 3: Tự điều chỉnh thủ công",
            ],
            key="sample_scenario_choice",
        )

        if "HỢP LỆ" in sample_choice:
            scenario_key = "legit"
            active_vals = SAMPLE_LEGIT_TRANSACTION
        elif "GIAN LẬN" in sample_choice:
            scenario_key = "fraud"
            active_vals = SAMPLE_FRAUD_TRANSACTION
        else:
            scenario_key = "custom"
            active_vals = SAMPLE_LEGIT_TRANSACTION

        with st.form(f"single_transaction_form_{scenario_key}"):
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                input_time = st.number_input(
                    "Time (Số giây trôi qua)",
                    value=float(active_vals["Time"]),
                    min_value=0.0,
                    key=f"{scenario_key}_Time",
                )
            with col_t2:
                input_amount = st.number_input(
                    "Amount (Số tiền giao dịch - USD)",
                    value=float(active_vals["Amount"]),
                    min_value=0.0,
                    key=f"{scenario_key}_Amount",
                )

            st.markdown("##### Các đặc trưng PCA ẩn danh (V1 - V28):")
            pca_cols = st.columns(7)
            user_features: dict[str, float] = {"Time": input_time, "Amount": input_amount}

            for i in range(1, 29):
                v_name = f"V{i}"
                with pca_cols[(i - 1) % 7]:
                    user_features[v_name] = st.number_input(
                        v_name,
                        value=float(active_vals.get(v_name, 0.0)),
                        format="%.4f",
                        key=f"{scenario_key}_{v_name}",
                    )

            submitted = st.form_submit_button("🚀 Phân tích giao dịch", use_container_width=True)

        if submitted:
            # Tạo DataFrame đúng thứ tự các cột
            feature_order = config.FEATURE_COLUMNS
            df_input = pd.DataFrame([user_features])[feature_order]

            # Dự đoán xác suất
            fraud_prob = float(model.predict_proba(df_input)[0, 1])
            is_fraud = fraud_prob >= CHOSEN_THRESHOLD

            st.markdown("---")
            st.markdown("### 📊 Kết quả Phân tích từ Mô hình")

            c_gauge, c_result = st.columns([1, 1.2])

            with c_gauge:
                # Vẽ thước đo Gauge Plotly
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=fraud_prob * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Xác suất Gian lận (%)", "font": {"size": 18}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1},
                            "bar": {"color": "#d9534f" if is_fraud else "#2b5c8f"},
                            "steps": [
                                {"range": [0, CHOSEN_THRESHOLD * 100], "color": "#e8f4f8"},
                                {"range": [CHOSEN_THRESHOLD * 100, 100], "color": "#fde8e8"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 3},
                                "thickness": 0.8,
                                "value": CHOSEN_THRESHOLD * 100,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(height=260, margin=dict(l=15, r=15, t=35, b=15))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with c_result:
                st.markdown(f"**Xác suất gian lận tính toán:** `{fraud_prob * 100:.2f}%`")
                st.markdown(f"**Ngưỡng phân định (Threshold):** `{CHOSEN_THRESHOLD * 100:.2f}%`")

                if is_fraud:
                    st.error(
                        "### 🔴 CẢNH BÁO: GIAN LẬN (FRAUD DETECTED)\n\n"
                        f"Giao dịch có xác suất gian lận vượt quá ngưỡng an toàn ({CHOSEN_THRESHOLD:.4f}).\n\n"
                        "**Hành động khuyến nghị:**\n"
                        "- Chặn giao dịch ngay lập tức.\n"
                        "- Tạm khóa thẻ và gửi thông báo cảnh báo qua ứng dụng ngân hàng / SMS.\n"
                        "- Yêu cầu xác thực bảo mật đa nhân tố (MFA / Smart OTP)."
                    )
                else:
                    st.success(
                        "### 🟢 KẾT QUẢ: HỢP LỆ (LEGITIMATE)\n\n"
                        f"Giao dịch an toàn, xác suất gian lận thấp hơn ngưỡng cảnh báo ({CHOSEN_THRESHOLD:.4f}).\n\n"
                        "**Hành động khuyến nghị:**\n"
                        "- Cho phép giao dịch hoàn tất bình thường."
                    )


# ---------------------------------------------------------------------------
# TAB 5: Dự đoán hàng loạt từ CSV
# ---------------------------------------------------------------------------
with tab_predict_batch:
    st.subheader("5. Dự đoán Hàng loạt từ File CSV (Batch Prediction)")
    st.markdown(
        "Tải lên file CSV chứa danh sách giao dịch để mô hình tự động quét và phân loại theo lô. "
        "File CSV cần chứa đủ 30 cột: `Time`, `V1` đến `V28`, và `Amount`."
    )

    col_btn1, col_btn2 = st.columns([1, 1])

    # Nút tải file mẫu
    sample_csv_path = config.PROJECT_ROOT / "data" / "sample_batch_transactions.csv"
    if sample_csv_path.is_file():
        with open(sample_csv_path, "rb") as f:
            col_btn1.download_button(
                label="📥 Tải file CSV mẫu (20 giao dịch thử nghiệm)",
                data=f,
                file_name="sample_transactions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    use_sample_data = col_btn2.button("⚡ Chạy thử ngay với 20 giao dịch mẫu", use_container_width=True)

    uploaded_file = st.file_uploader("Hoặc tải file CSV giao dịch của bạn lên:", type=["csv"])

    df_to_predict = None

    if use_sample_data and sample_csv_path.is_file():
        df_to_predict = pd.read_csv(sample_csv_path)
        st.info("Đang sử dụng tập 20 giao dịch mẫu được trích xuất từ dữ liệu kiểm thử.")
    elif uploaded_file is not None:
        try:
            df_to_predict = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Không thể đọc file CSV: {e}")

    if df_to_predict is not None and model is not None:
        st.markdown(f"**Dữ liệu nạp vào:** `{len(df_to_predict)}` dòng, `{len(df_to_predict.columns)}` cột.")

        # Kiểm tra tính hợp lệ của các cột
        expected_cols = config.FEATURE_COLUMNS
        missing_cols = [c for c in expected_cols if c not in df_to_predict.columns]

        if missing_cols:
            st.error(f"File CSV thiếu các cột bắt buộc: `{', '.join(missing_cols)}`")
        else:
            with st.spinner("Đang xử lý dự đoán cho toàn bộ lô giao dịch..."):
                X_batch = df_to_predict[expected_cols]
                probs = model.predict_proba(X_batch)[:, 1]
                preds = (probs >= CHOSEN_THRESHOLD).astype(int)

                df_result = df_to_predict.copy()
                df_result["Predicted_Probability"] = np.round(probs, 4)
                df_result["Predicted_Class"] = preds
                df_result["Status"] = np.where(preds == 1, "🔴 GIAN LẬN", "🟢 Hợp lệ")

            # Thống kê kết quả
            n_fraud_found = int((preds == 1).sum())
            n_legit_found = int((preds == 0).sum())
            pct_fraud_found = (n_fraud_found / len(preds)) * 100

            st.markdown("### 📊 Tổng kết kết quả phân loại:")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tổng giao dịch đã quét", f"{len(preds):,}")
            m2.metric("Giao dịch Hợp lệ", f"{n_legit_found:,}", delta=f"{(n_legit_found/len(preds))*100:.1f}%")
            m3.metric("Gian lận phát hiện", f"{n_fraud_found:,}", delta=f"{pct_fraud_found:.1f}%", delta_color="inverse")
            m4.metric("Ngưỡng áp dụng", f"{CHOSEN_THRESHOLD:.4f}")

            # Sắp xếp các cột: Status và Xác suất lên trước, theo sau là đầy đủ 30 đặc trưng đầu vào
            ordered_cols = [
                "Status",
                "Predicted_Probability",
                "Time",
                "Amount",
            ] + [f"V{i}" for i in range(1, 29)]
            remaining_cols = [c for c in df_result.columns if c not in ordered_cols]
            display_columns = ordered_cols + remaining_cols

            st.markdown("##### 📋 Danh sách chi tiết kết quả (Có thể cuộn ngang để xem đủ 30 đặc trưng V1 - V28):")
            st.dataframe(
                df_result[display_columns],
                use_container_width=True,
                height=450,
            )

            # Nút tải kết quả về máy (giữ nguyên đầy đủ tất cả các cột)
            csv_buffer = io.StringIO()
            df_result[display_columns].to_csv(csv_buffer, index=False)
            st.download_button(
                label="💾 Tải toàn bộ kết quả phân loại (CSV)",
                data=csv_buffer.getvalue(),
                file_name="fraud_detection_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
