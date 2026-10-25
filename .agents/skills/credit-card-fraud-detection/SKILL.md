---
name: credit-card-fraud-detection
description: >-
  Hướng dẫn và quy chuẩn chuyên biệt cho dự án Phát hiện gian lận thẻ tín dụng (Credit Card Fraud Detection).
  Kích hoạt khi cần phân tích dữ liệu, tiền xử lý, huấn luyện 4 mô hình (8 biến thể), tối ưu ngưỡng (threshold),
  đánh giá mô hình theo PR-AUC/F1, kiểm thử dự đoán trực tuyến/hàng loạt trên Streamlit và chạy unit test.
---

# Credit Card Fraud Detection - Project Skill Guide

Tài liệu hướng dẫn chuẩn hóa dành cho AI Agent khi làm việc trên dự án **Phát hiện gian lận thẻ tín dụng (Credit Card Fraud Detection)** - Bài tập lớn Khoa học Dữ liệu (MSSV: 12423031).

---

## 1. Project Overview (Tổng quan dự án)

- **Bài toán:** Phân loại nhị phân (Binary Classification) xác định giao dịch gian lận (`Class = 1`) hay hợp lệ (`Class = 0`).
- **Thách thức cốt lõi:** Dữ liệu mất cân bằng cực đoan (Extreme Class Imbalance).
  - Tổng số mẫu: 284,807 giao dịch.
  - Gian lận (`Class = 1`): 492 mẫu (~0.1727%).
  - Hợp lệ (`Class = 0`): 284,315 mẫu (~99.8273%).
  - Tỷ lệ mất cân bằng: ~578 giao dịch hợp lệ mới có 1 giao dịch gian lận.
- **Nghịch lý độ chính xác (Accuracy Paradox):** Nếu một mô hình luôn dự đoán nhãn 0, độ chính xác đạt tới 99.83% nhưng không bắt được bất kỳ vụ gian lận nào. Do đó, **Accuracy bị cấm làm thước đo chính**.
- **Thước đo đánh giá trọng tâm:** PR-AUC (Average Precision), F1-Score, Precision, Recall, và Confusion Matrix.

---

## 2. Technology Stack (Công nghệ & Thư viện)

- **Ngôn ngữ:** Python ≥ 3.10 (Môi trường thực thi hiện tại: Python 3.13.5 trên Windows).
- **Xử lý & Phân tích dữ liệu:** `pandas`, `numpy`.
- **Học máy (Machine Learning):** `scikit-learn` (Pipelines, StandardScaler, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, train_test_split, metrics), `xgboost` (XGBClassifier).
- **Trực quan hóa dữ liệu:** `matplotlib`, `seaborn`, `plotly` (Gauge charts & interactive plots).
- **Giao diện Web:** `streamlit`.
- **Lưu trữ & Tuần tự hóa mô hình:** `joblib`, `json`.
- **Kiểm thử tự động:** `pytest`, `pytest-cov`.

---

## 3. Project Structure (Cấu trúc thư mục thực tế)

```
.
├── .agents/skills/credit-card-fraud-detection/
│   └── SKILL.md            # Tài liệu Skill hướng dẫn dành cho Agent
├── app/
│   └── app.py              # Ứng dụng Streamlit Web Demo (5 Tab hoàn chỉnh)
├── data/
│   ├── raw/                # Chứa file creditcard.csv (git-ignored)
│   ├── processed/          # Chứa dữ liệu train/val/test sau khi chia
│   └── sample_batch_transactions.csv  # 20 mẫu kiểm thử lô (10 hợp lệ, 10 gian lận)
├── models/
│   ├── best_model.joblib   # Mô hình tốt nhất (Random Forest baseline)
│   └── model_metadata.json # Metadata: ngưỡng 0.5701, metrics validation và test
├── outputs/
│   ├── figures/            # Biểu đồ EDA và ma trận nhầm lẫn (Confusion Matrix)
│   └── metrics/            # Các file JSON/CSV tóm tắt từ Phase 2 đến Phase 5
├── reports/
│   ├── final_report.md     # Báo cáo tổng kết toàn diện bằng tiếng Việt
│   └── tables/             # Bảng so sánh 8 biến thể mô hình
├── src/
│   ├── eda.py              # Phase 2: Phân tích khám phá dữ liệu (6 biểu đồ)
│   ├── preprocessing.py    # Phase 3: Kiểm tra chất lượng & chia Stratified 70/15/15
│   ├── train.py            # Phase 4: Huấn luyện 4 mô hình (8 biến thể)
│   ├── evaluate.py         # Phase 5: Đánh giá, tối ưu ngưỡng theo F1 trên Validation
│   └── fraud/
│       ├── __init__.py
│       └── config.py       # Cấu hình trung tâm (đường dẫn, random_seed=42, tỷ lệ chia)
├── tests/                  # 6 file kiểm thử đơn vị (35 unit tests passing)
│   ├── test_config.py
│   ├── test_eda.py
│   ├── test_preprocessing.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   └── test_app.py
├── pyproject.toml          # Khai báo package và cấu hình pytest
├── requirements.txt        # Danh sách thư viện phụ thuộc
└── README.md               # Hướng dẫn cài đặt, chạy dự án và kiểm thử
```

---

## 4. Dataset Rules (Quy chuẩn dữ liệu)

- **Danh sách 30 đặc trưng đầu vào (`config.FEATURE_COLUMNS`):**
  - `Time`: Số giây trôi qua kể từ giao dịch đầu tiên trong tập dữ liệu.
  - `V1` đến `V28`: 28 thành phần chính thu được từ phép biến đổi PCA nhằm bảo mật danh tính khách hàng.
  - `Amount`: Số tiền thực hiện giao dịch (USD).
- **Cột nhãn mục tiêu:** `Class` (0 = Hợp lệ, 1 = Gian lận).
- **Quy tắc tính nguyên vẹn dữ liệu:**
  - Thứ tự các cột đưa vào mô hình phải khớp 100% với `config.FEATURE_COLUMNS`.
  - Tuyệt đối không tự bịa hoặc tạo ngẫu nhiên dữ liệu giả rồi gán mác "dữ liệu kiểm thử thực tế".
  - Khi cần dữ liệu mẫu, trích xuất chính xác các dòng có sẵn từ tập `test.csv` hoặc [data/sample_batch_transactions.csv](file:///data/sample_batch_transactions.csv).

---

## 5. Machine Learning Workflow (Quy trình làm việc chuẩn)

1. **Phase 2 - EDA (`src/eda.py`):** Kiểm tra schema, giá trị khuyết thiếu (0 missing values), tỷ lệ mất cân bằng, phân phối `Time`, phân phối log-transformed `Amount`, ma trận tương quan. Không chỉnh sửa file `creditcard.csv` gốc.
2. **Phase 3 - Preprocessing (`src/preprocessing.py`):**
   - Phân chia tập dữ liệu 2 bước: 70% Train, 15% Validation, 15% Test với `stratify=y` và `random_state=42`.
   - Chuẩn hóa: `StandardScaler` chỉ dùng cho Logistic Regression (fit duy nhất trên Train để chống rò rỉ dữ liệu - Data Leakage).
   - Tính toán trọng số mất cân bằng: `class_weight='balanced'` và `scale_pos_weight = N_neg / N_pos ≈ 577.88`.
3. **Phase 4 - Model Training (`src/train.py`):**
   - Huấn luyện đúng 4 thuật toán với 8 biến thể:
     1. Logistic Regression (Baseline vs Balanced)
     2. Decision Tree (Baseline vs Balanced)
     3. Random Forest (Baseline vs Balanced Subsample)
     4. XGBoost (Baseline vs Weighted với scale_pos_weight)
4. **Phase 5 - Evaluation & Threshold Optimization (`src/evaluate.py`):**
   - Tính Precision, Recall, F1, PR-AUC, ROC-AUC trên tập Validation.
   - Quét ngưỡng quyết định (threshold sweep) từ 0.01 đến 0.99 để chọn ngưỡng đạt **Max F1** trên tập Validation.
   - Chọn mô hình quán quân: **Random Forest (baseline)** với ngưỡng tối ưu **`0.5701`**.
   - Kiểm thử độc lập trên tập Test đúng **1 lần duy nhất** để báo cáo kết quả khách quan.
5. **Phase 6 - Web Application (`app/app.py`):**
   - Xây dựng giao diện Streamlit với 5 tab chức năng, nạp mô hình từ `models/best_model.joblib`.
6. **Phase 7 - Comprehensive Testing (`tests/`):**
   - Đảm bảo toàn bộ 35 unit test vượt qua.

---

## 6. Model Rules (Quy chuẩn mô hình)

- **Mô hình được chọn cố định:** `Random Forest (baseline)` (không dùng trọng số cân bằng vì trọng số làm tăng False Positives trên bài toán này).
- **Ngưỡng quyết định tối ưu đã chọn:** `0.5701` (ngưỡng mặc định 0.5 cho F1 = 0.8676, ngưỡng 0.5701 duy trì F1 = 0.8676 trên Validation và đạt F1 = 0.8358 trên Test).
- **Kết quả trên tập Test độc lập (42,722 giao dịch):**
  - **PR-AUC:** `0.8320`
  - **F1-Score:** `0.8358`
  - **Precision:** `93.33%` (Báo nhầm chỉ 4 trường hợp trên 42,648 giao dịch hợp lệ)
  - **Recall:** `75.68%` (Bắt trúng 56/74 vụ gian lận)
- **Quy tắc bất biến:** Tuyệt đối không tự ý huấn luyện lại mô hình, không thay đổi thuật toán hoặc tùy tiện thay đổi ngưỡng `0.5701` khi chưa có yêu cầu từ người dùng.

---

## 7. Prediction Rules (Quy tắc dự đoán giao dịch)

- **Đầu vào:** DataFrame phải có đúng 30 cột theo thứ tự:
  ```python
  ['Time', 'V1', 'V2', ..., 'V28', 'Amount']
  ```
- **Cách lấy xác suất:** Luôn lấy xác suất của lớp 1 (Gian lận):
  ```python
  fraud_prob = float(model.predict_proba(df_input)[0, 1])
  ```
- **Phân loại nhị phân theo ngưỡng:**
  ```python
  is_fraud = fraud_prob >= CHOSEN_THRESHOLD  # 0.5701
  # is_fraud == True  -> 🔴 CẢNH BÁO: GIAN LẬN
  # is_fraud == False -> 🟢 HỢP LỆ
  ```

---

## 8. CSV Batch Prediction Rules (Quy tắc dự đoán hàng loạt)

Khi xử lý tab dự đoán theo lô từ file CSV trong [app/app.py](file:///app/app.py):
1. **Kiểm tra schema:** Xác minh file tải lên có đầy đủ 30 cột bắt buộc thuộc `config.FEATURE_COLUMNS`.
2. **Bảo toàn dữ liệu đầu vào:** Bảng kết quả hiển thị và file tải về **phải giữ nguyên toàn bộ 30 đặc trưng gốc** (`Time`, $V_1 \dots V_{28}$, `Amount`). Không được cắt bỏ các cột $V_6 \dots V_{28}$.
3. **Thêm cột kết quả:**
   - `Predicted_Probability`: Làm tròn 4 chữ số thập phân (`np.round(probs, 4)`).
   - `Predicted_Class`: Nhãn `0` hoặc `1`.
   - `Status`: Chuỗi trực quan `🔴 GIAN LẬN` hoặc `🟢 Hợp lệ`.
4. **Bố cục hiển thị:** Đặt `Status` và `Predicted_Probability` lên đầu danh sách cột để tiện theo dõi, hỗ trợ cuộn ngang (`use_container_width=True`) cho toàn bộ 28 cột PCA.
5. **Chức năng tải xuống:** Cho phép tải đầy đủ DataFrame kết quả dạng CSV qua `st.download_button`.

---

## 9. Testing Guidelines (Quy tắc kiểm thử)

- **Công cụ:** `pytest`.
- **Lệnh chạy toàn bộ kiểm thử:**
  ```powershell
  pytest
  ```
- **Lệnh kiểm thử từng file cụ thể:**
  ```powershell
  pytest tests/test_app.py
  pytest tests/test_train.py
  ```
- **Quy tắc kiểm thử:**
  - Sau mỗi lần sửa mã nguồn hoặc giao diện, bắt buộc phải chạy lại `pytest` để xác nhận không gây hồi quy (regression).
  - Không bao giờ được xóa hoặc comment bài test hiện có để "lách" lỗi; phải tìm nguyên nhân trong code và sửa triệt để.
  - Khi bổ sung tính năng mới (như mẫu dữ liệu chuẩn, xử lý batch), phải viết test tương ứng trong thư mục `tests/`.

---

## 10. Code Modification Rules (Quy tắc can thiệp mã nguồn)

- **Nguyên tắc can thiệp tối thiểu (Surgical Changes):** Chỉ sửa đúng những dòng liên quan đến lỗi hoặc yêu cầu cụ thể của người dùng. Không tự ý tái cấu trúc (refactor) các phần đang hoạt động tốt.
- **Giữ nguyên chú thích và phong cách:** Giữ toàn bộ docstring, comment tiếng Việt và cấu trúc logging hiện hành.
- **Đọc trước khi sửa:** Luôn xem nội dung file và kiểm tra các phụ thuộc trước khi thực hiện thay đổi.
- **Tương thích Windows:**
  - Khi in tiếng Việt ra console trong các file script `src/`, luôn bảo đảm:
    ```python
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ```
  - Khi tính toán đường dẫn tương đối trong test, tránh gọi `path.resolve().relative_to(PROJECT_ROOT)` trực tiếp trên thư mục tạm `tmp_path` ở ổ đĩa khác.

---

## 11. Model Evaluation Integrity (Nguyên tắc toàn vẹn đánh giá)

- **Bảo vệ tập Test độc lập (Strict Test Set Isolation):** Không bao giờ dùng tập Test để chọn siêu tham số, huấn luyện lại, hoặc tối ưu ngưỡng. Tập Test chỉ được đánh giá duy nhất một lần ở bước cuối cùng.
- **Phân biệt rạch ròi giữa các loại lỗi:**
  - **Lỗi giao diện / trạng thái (UI / State Bug):** Ví dụ xung đột `key` trong Streamlit Session State dẫn đến việc nạp nhầm dữ liệu con lai.
  - **Lỗi xử lý dữ liệu (Data Pipeline Error):** Sai thứ tự cột, thiếu đặc trưng, chuẩn hóa nhầm tập.
  - **Sai số thống kê của mô hình (Model False Negative / False Positive):** Giao dịch có đặc trưng giống thật và mô hình tính toán xác suất thực tế $< 0.5701$.
  - Tuyệt đối không đổ lỗi cho mô hình khi nguyên nhân thực sự là lỗi code giao diện hoặc dữ liệu bị truyền sai.

---

## 12. Streamlit Rules (Quy chuẩn phát triển Streamlit)

- **Tránh huấn luyện lại mỗi lần render:** Nạp mô hình và metadata qua decorator `@st.cache_resource`, nạp các số liệu đánh giá qua `@st.cache_data`.
- **Quản lý Widget Keys an toàn:**
  - Không dùng chung `key` tĩnh cho các kịch bản kiểm thử khác nhau trong cùng một form.
  - Sử dụng tiền tố động theo kịch bản: `key=f"{scenario_key}_{v_name}"`, `key=f"{scenario_key}_Time"`, và định danh form `st.form(f"single_transaction_form_{scenario_key}")`.
  - Giúp Streamlit tải ngay giá trị mặc định của kịch bản mới mà không bị cache đè giá trị của kịch bản cũ.

---

## 13. Run Commands (Các lệnh chạy thông dụng trên Windows PowerShell)

```powershell
# 1. Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt thư viện và package nội bộ
pip install -r requirements.txt
pip install -e .

# 3. Chạy từng giai đoạn xử lý
python src/eda.py             # Phase 2: Khám phá dữ liệu
python src/preprocessing.py   # Phase 3: Tiền xử lý & chia tập 70/15/15
python src/train.py           # Phase 4: Huấn luyện 4 mô hình
python src/evaluate.py        # Phase 5: Đánh giá & tối ưu ngưỡng

# 4. Khởi chạy ứng dụng Web Demo trực quan
streamlit run app/app.py

# 5. Chạy toàn bộ bộ kiểm thử tự động
pytest
```

---

## 14. Reporting Format (Định dạng báo cáo chuẩn khi hoàn thành công việc)

Mỗi khi giải quyết một vấn đề hoặc hoàn thành một yêu cầu, báo cáo phản hồi cần tuân theo cấu trúc:
1. **Nguyên nhân gốc rễ (Root Cause):** Phân tích chi tiết cơ chế phát sinh vấn đề (lỗi code, lỗi state hay lỗi dữ liệu).
2. **Các tệp đã sửa đổi (Modified Files):** Liệt kê đường dẫn tệp dạng liên kết Markdown có thể nhấp (`[path/file.py](file:///...)`) kèm tóm tắt nội dung thay đổi.
3. **Kết quả kiểm thử (Test Results):** Trích dẫn kết quả chạy `pytest` và kiểm tra logic.
4. **Hướng dẫn kiểm tra trực quan:** Các bước cụ thể để người dùng tự xác minh lại trên giao diện hoặc terminal.
