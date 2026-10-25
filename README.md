# Credit Card Fraud Detection Using Machine Learning
*Phát hiện gian lận thẻ tín dụng bằng Machine Learning*

Data Science coursework project / Bài tập lớn môn Khoa học dữ liệu.
Author / Sinh viên: Nguyen Minh Thai – 12423031 *(edit if needed / chỉnh nếu cần)*

> **Status / Trạng thái:** Phase 1 đến Phase 6 đã hoàn thành. Đã có khung dự án, EDA, Preprocessing (70/15/15),
> huấn luyện 4 mô hình (8 biến thể), tối ưu ngưỡng (chọn Random Forest baseline @ 0.5701), ứng dụng Streamlit (`app/app.py`)
> và bộ 34 unit tests kiểm thử toàn diện. / Completed Phases 1–6 (all 34 tests passing).

---

## 1. Problem / Bài toán

Card fraud is rare but costly. Given a transaction described by anonymised numeric features, predict
whether it is **fraudulent** (`Class = 1`) or **legitimate** (`Class = 0`). The core difficulty is
**extreme class imbalance**: a model that labels every transaction "legitimate" would look very accurate
yet detect no fraud. For this reason Accuracy is not used as the main metric.

Gian lận thẻ rất hiếm nhưng gây thiệt hại lớn. Khó khăn chính là **mất cân bằng lớp nghiêm trọng**,
nên Accuracy không phải thước đo chính; dự án dùng Precision, Recall, F1 và PR-AUC.

## 2. Dataset / Dữ liệu

- Kaggle: [Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), file `creditcard.csv`.
- According to the dataset page: card transactions by European cardholders over two days, features
  `V1`–`V28` are PCA components, `Time` and `Amount` are the only non-PCA features, `Class` is the label.
  *All of this — schema, number of rows/columns, class ratio, missing values — will be verified on the real
  file in the EDA phase rather than assumed.*
- The dataset is **not** included in this repository. See [`data/README.md`](data/README.md) for how to download it.
- Please follow the citation request on the Kaggle page when using the data (e.g. Dal Pozzolo, Caelen, Johnson,
  Bontempi, *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE CIDM 2015).

## 3. Objectives / Mục tiêu

Build a complete data-science workflow, from raw data to a trained model and a demo app:

1. Exploratory Data Analysis (EDA) with verified schema, class ratio and missing values.
2. Train / validation / test split (stratified; limitations discussed; temporal split as an extra experiment).
3. Preprocessing and feature scaling inside a `Pipeline` (scaler fitted on train only).
4. Compare **exactly four models**: Logistic Regression, Decision Tree, Random Forest, XGBoost —
   each as a baseline and with class-imbalance handling (`class_weight` / `scale_pos_weight`).
5. Evaluate with Precision, Recall, F1, PR-AUC (average precision), Precision-Recall curves and confusion matrices.
6. Choose the classification threshold on the **validation** set; report the **test** set only once, at the end.
7. Save the final model + threshold metadata and serve it in a Streamlit demo.

Experimental rules / Quy tắc thực nghiệm: the test set is never used to pick a model or threshold; no resampling
(e.g. SMOTE) on validation/test; no fabricated metrics.

## 4. Technology / Công nghệ

Python ≥ 3.10 · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · Plotly · Streamlit · pytest.
Exact dependency ranges are in [`requirements.txt`](requirements.txt).

## 5. Project structure / Cấu trúc dự án

```
.
├── README.md
├── requirements.txt
├── pyproject.toml          # makes `src/fraud` importable (pip install -e .) + pytest settings
├── .gitignore  .gitattributes
├── data/
│   ├── README.md           # how to obtain creditcard.csv
│   ├── raw/                # creditcard.csv goes here (git-ignored)
│   └── processed/          # derived data (git-ignored)
├── src/
│   ├── eda.py              # Phase 2: exploratory data analysis (run directly)
│   └── fraud/
│       ├── __init__.py
│       └── config.py       # paths, seed, split ratios, model/threshold options
├── notebooks/              # exploratory notebooks (optional)
├── models/                 # trained model (git-ignored) + model_metadata.json
├── outputs/
│   ├── figures/            # EDA charts (tracked, small PNGs)
│   └── metrics/            # eda_summary.json (tracked)
├── reports/
│   ├── figures/            # EDA charts, PR curves, confusion matrices (tracked)
│   └── tables/             # model comparison tables (tracked)
└── tests/
    ├── test_config.py
    └── test_eda.py
```

Added in later phases / Sẽ thêm ở các giai đoạn sau: `src/fraud/{data,split,pipelines,threshold,metrics,plots,predict}.py`,
`scripts/` (EDA, training, evaluation, temporal experiment), `app/streamlit_app.py`, `reports/bao_cao.md`.

## 6. Installation / Cài đặt (Windows PowerShell)

```powershell
git clone https://github.com/nguyenminhthai2709/12423031_NguyenMinhThai_btl_ds.git
cd 12423031_NguyenMinhThai_btl_ds

python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

On Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`, then the same `pip` commands.

Verify the setup / Kiểm tra cài đặt:

```powershell
python -m fraud.config     # prints resolved paths and checks the configuration
pytest                     # runs the tests in tests/
```

Then download the dataset as described in [`data/README.md`](data/README.md).

## 7. Run the EDA / Chạy EDA (Phase 2)

Put `creditcard.csv` in `data/raw/` first (see [`data/README.md`](data/README.md)), then:

```powershell
python src/eda.py
# or with another location:
python src/eda.py --data-path D:\datasets\creditcard.csv
```

Outputs / Đầu ra: `outputs/figures/*.png` (6 charts) and `outputs/metrics/eda_summary.json`.
The script only describes the data: it never deletes rows, imputes values or modifies the raw file.
If the file is missing it prints download instructions and exits with code 1.

## 8. Roadmap / Lộ trình

| Phase | Content | Status |
|---|---|---|
| 1 | Project scaffolding & configuration | done |
| 2 | EDA (`src/eda.py`) | done (figures & metrics generated) |
| 3 | Preprocessing & splitting 70/15/15 (`src/preprocessing.py`) | done |
| 4 | Model training: 4 models, 8 variants (`src/train.py`) | done |
| 5 | Evaluation & Threshold selection (`src/evaluate.py`) | done (Best: Random Forest @ 0.5701) |
| 6 | Streamlit web demo (`app/app.py`) | done |
| 7 | Comprehensive testing (`tests/` – 34 unit tests) | done (34 passed) |
| 8 | Reporting (`reports/final_report.md`) | done |

## 9. Execution Workflow / Hướng dẫn chạy từng giai đoạn (PowerShell)

Mở PowerShell tại thư mục gốc của dự án (`C:\Users\BOOK PRO\12423031_NguyenMinhThai_btl_ds`) và thực hiện theo thứ tự:

```powershell
# Bước 1: Khám phá và phân tích dữ liệu (Phase 2)
python src/eda.py

# Bước 2: Tiền xử lý dữ liệu và chia tập Stratified 70/15/15 (Phase 3)
python src/preprocessing.py

# Bước 3: Huấn luyện và so sánh 4 mô hình học máy (Phase 4)
python src/train.py

# Bước 4: Đánh giá mô hình, tối ưu ngưỡng và kiểm thử độc lập (Phase 5)
python src/evaluate.py

# Bước 5: Chạy ứng dụng Web Demo trực quan (Phase 6)
streamlit run app/app.py

# Bước 6: Chạy toàn bộ 34 bài kiểm thử tự động (Phase 7)
pytest
```

## 10. Run the Streamlit Demo App / Chi tiết ứng dụng Web Demo (Phase 6)

Khởi chạy giao diện trực quan bằng Streamlit:

```powershell
streamlit run app/app.py
```

Ứng dụng cung cấp 5 tab chức năng:
1. **Tổng quan dữ liệu:** Kích thước, phân phối nhãn mất cân bằng và phân chia tập 70/15/15.
2. **Biểu đồ EDA:** 6 biểu đồ phân tích trực quan từ Phase 2.
3. **So sánh 4 mô hình:** Bảng so sánh 8 biến thể, đồ thị PR curves, ROC curves và Confusion Matrix.
4. **Dự đoán một giao dịch:** Nhập tham số hoặc chọn kịch bản mẫu, phân loại theo ngưỡng 0.5701.
5. **Dự đoán hàng loạt từ CSV:** Tải file CSV lên, quét hàng loạt và tải file kết quả về máy.

## 11. Reproducibility / Tái lập kết quả

A single seed (`RANDOM_SEED = 42`) and the split ratios (70/15/15) are defined in `src/fraud/config.py`.
After the final run, exact package versions will be frozen (`pip freeze > requirements-lock.txt`) and the
commands to regenerate every figure and table will be documented here.

## 12. Security and data hygiene / An toàn dữ liệu

- Never commit `creditcard.csv`, `kaggle.json`, `.env` files or API tokens (all are git-ignored).
- Large model files are git-ignored by default; see the comment in `.gitignore` for adding the final model deliberately.
- Báo cáo tổng kết đầy đủ bằng tiếng Việt được lưu tại [`reports/final_report.md`](reports/final_report.md).
