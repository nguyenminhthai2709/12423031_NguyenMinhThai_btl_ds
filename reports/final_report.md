# BÁO CÁO TỔNG KẾT ĐỀ TÀI BÀI TẬP LỚN KHOA HỌC DỮ LIỆU
## ĐỀ TÀI: PHÁT HIỆN GIAN LẬN THẺ TÍN DỤNG BẰNG MACHINE LEARNING
**(Credit Card Fraud Detection Using Machine Learning)**

* **Sinh viên thực hiện:** Nguyễn Minh Thái
* **Mã số sinh viên (MSSV):** 12423031
* **Môn học:** Khoa học Dữ liệu (Data Science)
* **Thời gian hoàn thành:** Tháng 10/2026
* **Môi trường thực nghiệm:** Python 3.13.5, Windows 11, PowerShell

---

## TÓM TẮT DỰ ÁN (EXECUTIVE SUMMARY)
Đề tài xây dựng một quy trình khoa học dữ liệu hoàn chỉnh từ đầu đến cuối (End-to-End Data Science Pipeline) nhằm giải quyết bài toán phát hiện gian lận trong giao dịch thẻ tín dụng ngân hàng với đặc trưng **mất cân bằng lớp cực đoan (0.17% gian lận)**:
1. **Khám phá dữ liệu (EDA):** Xác minh 284,807 giao dịch từ Kaggle `creditcard.csv`, 30 đặc trưng (`Time`, `Amount`, `V1`–`V28`), 0 giá trị thiếu, 1,081 dòng trùng lặp được giữ nguyên vì lý do bảo toàn mẫu gian lận.
2. **Tiền xử lý & Chia tập:** Phân chia Stratified Random Split 70% Train - 15% Validation - 15% Test (seed 42); áp dụng `StandardScaler` cho `Time` và `Amount` qua Pipeline và **chỉ fit trên tập Train** để triệt tiêu hoàn toàn rò rỉ dữ liệu (Data Leakage).
3. **Huấn luyện & So sánh 4 mô hình:** Logistic Regression, Decision Tree, Random Forest, XGBoost với 2 chiến lược: `baseline` (không trọng số) và `class_weight`/`scale_pos_weight = 578.55`.
4. **Tối ưu hóa ngưỡng phân loại (Threshold Optimization):** Khảo sát đường cong Precision-Recall Curves và tinh chỉnh ngưỡng quyết định trên tập Validation. Mô hình **Random Forest (baseline)** được lựa chọn với ngưỡng tối ưu **`threshold = 0.5701`** (đạt PR-AUC = 0.8619, F1 = 0.8676 trên Validation).
5. **Kiểm thử độc lập trên tập Test:** Đánh giá đúng **1 lần duy nhất** trên 42,722 mẫu Test độc lập, đạt **PR-AUC = 0.8320, F1 = 0.8358, Precision = 93.33%, Recall = 75.68%**, chặn đứng **56 / 74** vụ gian lận và chỉ báo nhầm **4** trường hợp trên 42,648 giao dịch hợp lệ.
6. **Ứng dụng Demo Streamlit:** Xây dựng ứng dụng web tại `app/app.py` hỗ trợ trực quan hóa dữ liệu, so sánh mô hình, dự đoán đơn lẻ và dự đoán hàng loạt từ file CSV.

---

## 1. ĐẶT VẤN ĐỀ VÀ MỤC TIÊU NGHIÊN CỨU

### 1.1. Bối cảnh thực tế
Gian lận thẻ tín dụng (Credit Card Fraud) là thách thức an ninh tài chính hàng đầu của hệ thống ngân hàng toàn cầu. Các giao dịch gian lận chiếm tỷ lệ rất nhỏ nhưng gây ra thiệt hại tài chính nghiêm trọng và làm suy giảm niềm tin của khách hàng. Việc phát hiện thủ công là bất khả thi trước khối lượng hàng triệu giao dịch mỗi giây, đòi hỏi hệ thống trí tuệ nhân tạo (AI/Machine Learning) có khả năng phân loại theo thời gian thực.

### 1.2. Thách thức cốt lõi: Mất cân bằng lớp cực đoan (Extreme Imbalance)
Trong tập dữ liệu thực tế, tỷ lệ giao dịch gian lận chỉ chiếm **0.1727%** (492 vụ trên tổng số 284,807 giao dịch). Một mô hình phân loại ngây thơ (Naive Classifier) luôn đoán tất cả giao dịch là "Hợp lệ" sẽ đạt độ chính xác (Accuracy) lên tới **99.8273%** nhưng hoàn toàn vô dụng vì bỏ lọt 100% số vụ gian lận. 

Do đó, đề tài đặt ra các nguyên tắc bắt buộc:
* **Không dùng Accuracy làm thước đo chính.**
* Thước đo đánh giá cốt lõi là **PR-AUC (Area Under the Precision-Recall Curve)**, kết hợp cùng **F1-score**, **Precision** và **Recall**.
* Không bịa đặt số liệu, không dùng tập Test để chọn mô hình hay chỉnh tham số.

---

## 2. BỘ DỮ LIỆU VÀ PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)

### 2.1. Nguồn gốc và Cấu trúc dữ liệu
Bộ dữ liệu được cung cấp bởi nhóm nghiên cứu Machine Learning Group (MLG) thuộc Đại học Université Libre de Bruxelles (Bỉ) trên nền tảng Kaggle:
* **File dữ liệu:** `data/raw/creditcard.csv` (dung lượng 143.84 MB).
* **Số dòng (samples):** 284,807 dòng.
* **Số cột (features):** 31 cột, gồm:
  * `Time`: Số giây trôi qua kể từ giao dịch đầu tiên trong bộ dữ liệu (từ 0 đến 172,792 giây, tương đương 48 giờ).
  * `Amount`: Số tiền của giao dịch (USD), từ 0.00 đến 25,691.16 USD.
  * `V1` đến `V28`: 28 đặc trưng số học ẩn danh thu được từ phép biến đổi Phân tích Thành phần Chính (PCA) nhằm bảo mật thông tin cá nhân của chủ thẻ.
  * `Class`: Biến mục tiêu (nhãn), trong đó `0 = Hợp lệ (Legitimate)`, `1 = Gian lận (Fraud)`.

### 2.2. Kiểm tra chất lượng dữ liệu
* **Giá trị khuyết thiếu (Missing values):** **0** giá trị thiếu trên toàn bộ 31 cột. Dữ liệu có chất lượng kỹ thuật rất cao.
* **Dữ liệu trùng lặp (Duplicates):** Phát hiện **1,081** dòng trùng lặp. Phân tích sâu cho thấy trong các dòng trùng này có **32 dòng mang nhãn gian lận (`Class = 1`)**.
  > **Quyết định xử lý:** **Giữ nguyên 100% dữ liệu, không xóa các dòng trùng.** Trong thanh toán điện tử, các giao dịch vi mô hoặc thao tác quẹt thẻ lặp lại có số tiền và thời gian tương đồng là hiện tượng thực tế. Xóa bỏ tùy tiện sẽ làm mất đi 32 mẫu gian lận quý giá (chiếm 6.5% tổng số gian lận).

### 2.3. Các phát hiện quan trọng từ EDA
1. **Phân phối lớp:** 284,315 giao dịch hợp lệ (99.8273%) và 492 giao dịch gian lận (0.1727%). Tỷ lệ mất cân bằng là **577.9 : 1**.
2. **Phân phối số tiền (Amount):** Giao dịch gian lận có giá trị trung bình là 122.21 USD, trung vị 9.25 USD, giá trị lớn nhất là 2,125.87 USD. Không có vụ gian lận nào vượt quá 3,000 USD (trong khi giao dịch hợp lệ có số tiền lên tới 25,691 USD).
3. **Phân tích thời gian (Time):** Tỷ lệ gian lận tăng đột biến vào các khung giờ đêm (khi lượng giao dịch hợp lệ giảm mạnh nhưng các đợt tấn công gian lận tự động vẫn diễn ra).
4. **Phân tích tương quan:** Các biến $V_{17}$ ($r = -0.3265$), $V_{14}$ ($r = -0.3025$), $V_{12}$ ($r = -0.2606$) có tương quan nghịch mạnh nhất với gian lận; trong khi $V_{11}$ ($r = +0.1549$) và $V_{4}$ ($r = +0.1334$) có tương quan thuận rõ nhất.

*(Các biểu đồ tương ứng được lưu tại `outputs/figures/` và `reports/figures/`: `class_distribution.png`, `amount_distribution.png`, `amount_by_class.png`, `time_by_class.png`, `correlation_heatmap.png`, `correlation_with_class.png`).*

---

## 3. TIỀN XỬ LÝ VÀ PHÂN CHIA TẬP DỮ LIỆU (PREPROCESSING & SPLITTING)

### 3.1. Chiến lược phân chia Train - Validation - Test (70% / 15% / 15%)
Để phục vụ việc huấn luyện, lựa chọn mô hình và đánh giá khách quan, dữ liệu được phân chia theo quy trình 2 bước sử dụng `train_test_split` với `stratify=y` và cố định `random_state = 42`:
* **Bước 1:** Tách 15% làm tập Test độc lập $\rightarrow$ còn lại 85% (Train + Val).
* **Bước 2:** Tách tập Validation với tỷ lệ tương đối $0.15 / (0.70 + 0.15) \approx 17.65\%$ trên tập còn lại $\rightarrow$ thu được đúng 70% Train và 15% Validation.

| Phân vùng | Tỷ lệ (%) | Tổng số mẫu | Hợp lệ (`Class=0`) | Gian lận (`Class=1`) | Tỷ lệ gian lận (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Dataset** | 100% | 284,807 | 284,315 | 492 | 0.1727% |
| **Train Set** | **70%** | **199,364** | 199,020 | **344** | 0.1725% |
| **Validation Set** | **15%** | **42,721** | 42,647 | **74** | 0.1732% |
| **Test Set (Độc lập)** | **15%** | **42,722** | 42,648 | **74** | 0.1732% |

Tỷ lệ gian lận được bảo toàn hoàn hảo ở mức $\approx 0.173\%$ trên tất cả các tập nhờ cơ chế phân tầng (`stratify`).

### 3.2. Vai trò của từng tập và Hạn chế của phân chia ngẫu nhiên
* **Train (70%):** Học quy luật và cập nhật trọng số mô hình.
* **Validation (15%):** So sánh thuật toán và tinh chỉnh ngưỡng quyết định (threshold tuning).
* **Test (15%):** Đánh giá khách quan năng lực tổng quát hóa, chỉ mở ra đúng 1 lần ở cuối đề tài.
* **Hạn chế của Stratified Random Split khi có cột Time:** Trong thực tế ngân hàng, giao dịch là chuỗi thời gian liên tục. Chia ngẫu nhiên có thể đưa giao dịch ngày 2 vào Train và giao dịch ngày 1 vào Test/Val, gây hiện tượng **Temporal Data Leakage (nhìn trước tương lai)**. Đây là giới hạn mang tính học thuật cần lưu ý khi chuyển giao sang môi trường sản xuất thực tế.

### 3.3. Chuẩn hóa đặc trưng & Chống rò rỉ dữ liệu (No Data Leakage)
* Các đặc trưng $V_1$–$V_{28}$ đã qua PCA nên có trung bình $\approx 0$ và phương sai $\approx 1$.
* Hai đặc trưng `Time` và `Amount` có thang đo rất lớn (`Time` lên tới ~172,000, `Amount` lên tới ~25,000). Mô hình tuyến tính (Logistic Regression) rất nhạy cảm với thang đo này.
* Sử dụng `ColumnTransformer` áp dụng `StandardScaler` cho `['Time', 'Amount']` và giữ nguyên các cột PCA (`remainder='passthrough'`).
* **Nguyên tắc vàng:** Scaler **chỉ được gọi `.fit()` trên tập Train** ($\mu_{Time} = 94,910.09$, $\sigma_{Time} = 47,516.53$; $\mu_{Amount} = 88.23$, $\sigma_{Amount} = 248.38$). Tập Validation và Test tuyệt đối chỉ được `.transform()`.

---

## 4. HUẤN LUYỆN VÀ SO SÁNH 4 MÔ HÌNH HỌC MÁY (PHASE 4)

### 4.1. Bốn thuật toán được lựa chọn
1. **Logistic Regression (Hồi quy Logistic):** Mô hình tuyến tính chuẩn mực, tích hợp `StandardScaler` qua `Pipeline`.
2. **Decision Tree (Cây quyết định):** Cây phân lớp với `max_depth = 10` để tránh học vẹt.
3. **Random Forest (Rừng ngẫu nhiên):** Mô hình học kết hợp dạng Bagging gồm 100 cây (`n_estimators = 100`, `max_depth = 12`, `n_jobs = -1`).
4. **XGBoost (Extreme Gradient Boosting):** Mô hình tăng cường độ dốc gồm 100 cây (`max_depth = 5`, `learning_rate = 0.1`, `eval_metric = "logloss"`).

### 4.2. Chiến lược xử lý mất cân bằng lớp (Imbalance Handling)
* **Không dùng SMOTE:** Nhằm tránh việc nội suy ra các giao dịch gian lận phi thực tế trong không gian đa chiều, giữ nguyên phân phối tự nhiên của dữ liệu.
* **Kỹ thuật gán trọng số (Cost-sensitive learning):**
  * Mô hình Sklearn (Logistic Regression, Decision Tree, Random Forest): dùng `class_weight = "balanced"`.
  * Mô hình XGBoost: sử dụng tham số `scale_pos_weight = n_negative / n_positive = 199,020 / 344 = 578.55` (tính hoàn toàn trên tập Train).

### 4.3. Kết quả so sánh 8 biến thể mô hình (Ngưỡng mặc định 0.5 trên Validation)

| Mô hình | Chiến lược | PR-AUC | F1-Score | Precision | Recall | ROC-AUC | Thời gian fit (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | `baseline` | **0.8619** | **0.8676** | **0.9516** | 0.7973 | **0.9879** | 28.42s |
| **Random Forest** | `class_weight` | 0.8557 | 0.8333 | 0.8571 | 0.8108 | 0.9820 | 22.72s |
| **XGBoost** | `scale_pos_weight` | **0.8417** | 0.7027 | 0.5856 | **0.8784** | 0.9832 | **1.69s** |
| **XGBoost** | `baseline` | 0.8290 | 0.8120 | 0.9153 | 0.7297 | 0.9883 | **1.50s** |
| **Logistic Regression** | `baseline` | 0.7705 | 0.7097 | 0.8800 | 0.5946 | 0.9863 | **0.36s** |
| **Logistic Regression** | `class_weight` | 0.7520 | 0.1269 | 0.0681 | **0.9189** | 0.9859 | **0.86s** |
| **Decision Tree** | `baseline` | 0.6880 | 0.8227 | 0.8657 | 0.7838 | 0.8920 | 6.74s |
| **Decision Tree** | `class_weight` | 0.6638 | 0.4037 | 0.2609 | 0.8919 | 0.9454 | 5.82s |

---

## 5. ĐÁNH GIÁ CHI TIẾT VÀ TỐI ƯU HÓA NGƯỠNG PHÂN LOẠI (PHASE 5)

### 5.1. Tại sao ngưỡng 0.5 không tối ưu cho bài toán mất cân bằng?
Trong bài toán phân loại giao dịch gian lận, chi phí của hai loại sai lầm là bất đối xứng:
* Bỏ sót gian lận (False Negative - FN) làm ngân hàng mất tiền trực tiếp.
* Báo nhầm giao dịch hợp lệ (False Positive - FP) gây phiền hà khách hàng.
Tại ngưỡng mặc định 0.5, các mô hình có gán trọng số (`class_weight`) bị phạt quá nặng nên đẩy xác suất gian lận lên rất cao, dẫn tới nhiều cảnh báo giả (Precision của Logistic Regression rơi xuống 6.81%). Ngược lại, các mô hình không gán trọng số có xu hướng quá thận trọng (Recall bị thấp).

### 5.2. Bảng so sánh trước và sau khi tối ưu Threshold (Tối ưu hóa Max F1 trên Validation)

| Mô hình | Chiến lược | PR-AUC | F1 (ngưỡng 0.5) | Ngưỡng tối ưu | F1 (ngưỡng tối ưu) | Precision (opt) | Recall (opt) | Mức tăng F1 ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | `baseline` | **0.8619** | **0.8676** | **0.5701** | **0.8676** | **0.9516** | 0.7973 | +0.0000 |
| **Random Forest** | `class_weight` | 0.8557 | 0.8333 | 0.5780 | 0.8451 | 0.8824 | 0.8108 | +0.0118 |
| **XGBoost** | `scale_pos_weight` | 0.8417 | 0.7027 | 0.9639 | **0.8551** | 0.9219 | 0.7973 | **+0.1524** |
| **XGBoost** | `baseline` | 0.8290 | 0.8120 | 0.1103 | 0.8429 | 0.8939 | 0.7973 | +0.0309 |
| **Logistic Regression** | `baseline` | 0.7705 | 0.7097 | 0.0779 | **0.8312** | 0.8000 | 0.8649 | **+0.1215** |
| **Logistic Regression** | `class_weight` | 0.7520 | 0.1269 | 1.0000 | **0.8310** | 0.8676 | 0.7973 | **+0.7041** |
| **Decision Tree** | `baseline` | 0.6880 | 0.8227 | 1.0000 | 0.8227 | 0.8657 | 0.7838 | +0.0000 |
| **Decision Tree** | `class_weight` | 0.6638 | 0.4037 | 0.9994 | 0.7949 | 0.7561 | 0.8378 | **+0.3912** |

> **Nhận xét then chốt:** Việc tối ưu ngưỡng đã hồi phục hoàn toàn hiệu năng của các mô hình gán trọng số: F1 của `Logistic Regression (class_weight)` nhảy từ **0.1269 lên 0.8310**; `XGBoost (weighted)` nhảy từ **0.7027 lên 0.8551**.

### 5.3. Mô hình được chọn làm hệ thống chính thức
* **Mô hình thắng cuộc:** **Random Forest (baseline)**
* **Ngưỡng quyết định lựa chọn:** **`threshold = 0.5701`**
* **Lý do lựa chọn:** Đạt PR-AUC cao nhất (0.8619), F1 tối ưu cao nhất (0.8676) với tỷ lệ báo nhầm thấp nhất (chỉ 3 ca báo nhầm trên 42,647 giao dịch hợp lệ của tập Validation).

---

## 6. ĐÁNH GIÁ ĐỘC LẬP TRÊN TẬP TEST (FINAL TEST EVALUATION)

Để đảm bảo tính khách quan và liêm chính học thuật, mô hình Random Forest với ngưỡng $0.5701$ đã chọn từ Validation được áp dụng lên tập Test (42,722 mẫu, gồm 42,648 giao dịch hợp lệ và 74 giao dịch gian lận) đúng **MỘT LẦN DUY NHẤT**:

### 6.1. Bảng số liệu kết quả thực tế trên tập Test

| Thước đo đánh giá | Kết quả trên tập Test | So sánh với Validation | Nhận định học thuật |
| :--- | :---: | :---: | :--- |
| **PR-AUC** | **0.8320** | 0.8619 | Rất ổn định trên toàn dải ngưỡng |
| **ROC-AUC** | **0.9622** | 0.9879 | Khả năng phân tách hai lớp gần như tuyệt đối |
| **F1-Score** | **0.8358** | 0.8676 | Chênh lệch $< 3.2\%$ $\rightarrow$ **Không hề bị Overfitting** |
| **Precision** | **93.33%** | 95.16% | Cực kỳ chính xác (cứ 100 cảnh báo thì $> 93$ vụ là thật) |
| **Recall** | **75.68%** | 79.73% | Bắt trúng **56 / 74** vụ gian lận thực tế |

### 6.2. Ma trận nhầm lẫn (Confusion Matrix) trên tập Test
$$\begin{pmatrix} \text{TN: } 42,644 & \text{FP: } 4 \\ \text{FN: } 18 & \text{TP: } 56 \end{pmatrix}$$
* **True Negative (TN = 42,644):** 42,644 giao dịch hợp lệ được lưu thông trơn tru.
* **False Positive (FP = 4):** Chỉ có **4 giao dịch hợp lệ bị cảnh báo nhầm** trong tổng số 42,648 giao dịch (tỷ lệ báo động giả chỉ **0.009%** $\rightarrow$ trải nghiệm người dùng gần như hoàn hảo).
* **False Negative (FN = 18):** 18 vụ gian lận lọt lưới.
* **True Positive (TP = 56):** **Chặn đứng 56 vụ gian lận**, ngăn ngừa thất thoát tài chính lớn cho chủ thẻ và ngân hàng.

---

## 7. ỨNG DỤNG WEB DEMO TRỰC QUAN BẰNG STREAMLIT (PHASE 6)

Ứng dụng được xây dựng tại [`app/app.py`](app/app.py) và chia thành 5 phân hệ chuyên sâu:
1. **Phân hệ 1 - Tổng quan dữ liệu:** Các thẻ KPI, biểu đồ tròn phân phối nhãn mất cân bằng và bảng phân chia 70/15/15.
2. **Phân hệ 2 - Biểu đồ EDA:** Khám phá 6 biểu đồ trực quan hóa dữ liệu theo các nhóm chủ đề Amount, Time và Correlation.
3. **Phân hệ 3 - So sánh 4 mô hình & Đánh giá:** Bảng số liệu thực tế của 8 biến thể, đồ thị PR curves, ROC curves và Confusion Matrix.
4. **Phân hệ 4 - Dự đoán giao dịch đơn lẻ:** Cho phép chọn kịch bản mẫu (Hợp lệ vs Gian lận) hoặc nhập tay 30 tham số; đồng hồ đo xác suất tương tác so sánh với ngưỡng 0.5701 và đề xuất hành động nghiệp vụ.
5. **Phân hệ 5 - Dự đoán hàng loạt từ file CSV:** Tải file CSV lên, kiểm tra schema, quét tự động theo lô và xuất file kết quả phân loại dạng CSV có nhãn và xác suất.

---

## 8. HẠN CHẾ CỦA ĐỀ TÀI VÀ HƯỚNG MỞ RỘNG

### 8.1. Hạn chế hiện tại
1. **Tính chất ẩn danh của dữ liệu:** Do yêu cầu bảo mật, 28 đặc trưng $V_1$–$V_{28}$ đã qua biến đổi PCA, làm giảm khả năng diễn giải nghiệp vụ chi tiết (Feature Explainability).
2. **Hiện tượng Temporal Leakage tiềm ẩn:** Do sử dụng phân chia ngẫu nhiên (Stratified Split), mô hình có thể đã học một phần quy luật từ tương lai.
3. **Thời gian thu thập ngắn:** Dữ liệu chỉ gồm 2 ngày giao dịch (48 giờ), chưa phản ánh được sự biến động theo tuần, tháng hay mùa vụ (Seasonality).

### 8.2. Hướng phát triển tiếp theo
1. **Thử nghiệm Temporal Split:** Sắp xếp theo cột `Time` để chia: 70% thời gian đầu làm Train, 15% tiếp theo làm Validation và 15% cuối cùng làm Test (phản ánh sát nhất thực tế triển khai).
2. **Áp dụng kỹ thuật giải thích mô hình (XAI):** Tích hợp SHAP (SHapley Additive exPlanations) để giải thích lý do cụ thể vì sao một giao dịch bị gắn cờ gian lận.
3. **Mô hình học không giám sát & Bán giám sát (Unsupervised/Semi-supervised):** Kết hợp Isolation Forest hoặc AutoEncoder để phát hiện các dạng gian lận kiểu mới (Zero-day fraud attacks).

---

## 9. KẾT LUẬN

Đề tài đã hoàn thành xuất sắc toàn bộ các mục tiêu đặt ra cho bài tập lớn môn Khoa học Dữ liệu:
* Xây dựng một quy trình Machine Learning chuẩn mực, tái lập được kết quả (`seed = 42`).
* Làm chủ phương pháp xử lý dữ liệu mất cân bằng cực đoan và tối ưu hóa ngưỡng phân loại trên đồ thị Precision-Recall Curve.
* Chứng minh mô hình Random Forest (baseline) đạt hiệu năng tổng quát hóa xuất sắc trên tập Test độc lập (**F1 = 0.8358, Precision = 93.33%, Recall = 75.68%**).
* Đóng gói giải pháp thành một ứng dụng web demo hoàn chỉnh, trực quan, dễ sử dụng bằng Streamlit.
* Đảm bảo tính trung thực học thuật 100%: mọi số liệu và nhận định đều xuất phát từ thực nghiệm thực tế trên máy tính cá nhân.
