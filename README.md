# Credit Card Fraud Detection Using Machine Learning
*Phát hiện gian lận thẻ tín dụng bằng Machine Learning*

Data Science coursework project / Bài tập lớn môn Khoa học dữ liệu.
Author / Sinh viên: Nguyen Minh Thai – 12423031 *(edit if needed / chỉnh nếu cần)*

> **Status / Trạng thái:** Phase 1 (scaffolding) done; Phase 2 (EDA) code written — it must be run on the
> real dataset by the author. No model has been trained yet and this README contains **no experimental
> results**. / Đã có khung dự án và code EDA; **chưa** huấn luyện mô hình, **chưa** có số liệu thực nghiệm.

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
| 1 | Project scaffolding (this commit) | done, awaiting review |
| 2 | EDA (`src/eda.py`) | code written, to be run on real data |
| 3 | Preprocessing and splitting | planned |
| 4 | Model training | planned |
| 5 | Evaluation and threshold selection | planned |
| 6 | Streamlit app | planned |
| 7 | Testing | planned |
| 8 | Reporting (Vietnamese Markdown report) | planned |

## 9. Reproducibility / Tái lập kết quả

A single seed (`RANDOM_SEED = 42`) and the split ratios (60/20/20) are defined in `src/fraud/config.py`.
After the final run, exact package versions will be frozen (`pip freeze > requirements-lock.txt`) and the
commands to regenerate every figure and table will be documented here.

## 10. Security and data hygiene / An toàn dữ liệu

- Never commit `creditcard.csv`, `kaggle.json`, `.env` files or API tokens (all are git-ignored).
- Large model files are git-ignored by default; see the comment in `.gitignore` for adding the final model deliberately.
