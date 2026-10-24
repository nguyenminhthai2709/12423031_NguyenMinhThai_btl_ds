# Data / Dữ liệu

The raw dataset is **not** stored in this repository (it is large and is excluded by `.gitignore`).
Dataset gốc **không** nằm trong repo (dung lượng lớn, đã được `.gitignore` loại trừ).

- Source / Nguồn: Kaggle – *Credit Card Fraud Detection* (`mlg-ulb/creditcardfraud`)
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Expected file / File cần có: `data/raw/creditcard.csv`

## Option A – Manual download (simplest) / Tải thủ công (đơn giản nhất)

1. Sign in to Kaggle and open the dataset page above. / Đăng nhập Kaggle và mở trang dataset.
2. Click **Download**, extract the zip. / Bấm **Download**, giải nén file zip.
3. Put `creditcard.csv` into `data/raw/`. / Đặt `creditcard.csv` vào thư mục `data/raw/`.

Resulting path / Đường dẫn kết quả: `data/raw/creditcard.csv`

## Option B – Kaggle CLI

```powershell
pip install kaggle
# Put the API token in %USERPROFILE%\.kaggle\kaggle.json  (OUTSIDE this repository).
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

**Never** copy `kaggle.json` into this project or commit it. `.gitignore` blocks it as a safety net,
but the token should live outside the repository.
**Không bao giờ** đặt `kaggle.json` trong project hoặc commit nó.

## Using a different location / Dùng vị trí khác

```powershell
$env:FRAUD_DATA_PATH = "D:\datasets\creditcard.csv"
```

`fraud.config.RAW_DATA_PATH` reads this environment variable.

## Expected schema (to be verified in the EDA phase) / Schema kỳ vọng

| Column | Description |
|---|---|
| `Time` | Seconds elapsed between each transaction and the first transaction in the dataset |
| `V1` … `V28` | Anonymised numerical features (PCA output; original features are not disclosed) |
| `Amount` | Transaction amount |
| `Class` | Target: `0` = legitimate, `1` = fraud |

Row count, class ratio and missing values are **not** assumed here: they will be measured
from the real file by the EDA script and reported from that output.
Số dòng, tỷ lệ lớp và giá trị thiếu sẽ được **đo trực tiếp** từ file thật ở giai đoạn EDA.

`data/processed/` is reserved for derived files (e.g. split indices); its content is git-ignored.
