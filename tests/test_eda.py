"""Kiểm thử các hàm trong src/eda.py.

LƯU Ý: các DataFrame dưới đây là dữ liệu tổng hợp NHỎ chỉ dùng để kiểm tra code
chạy đúng (fixture của unit test). Chúng KHÔNG đại diện cho dataset thật và
không bao giờ được dùng để báo cáo kết quả EDA.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import eda
from fraud import config


def make_fixture(n_rows: int = 600, n_fraud: int = 30, seed: int = 0) -> pd.DataFrame:
    """Tạo bảng nhỏ có đúng schema của dataset (chỉ để test code)."""
    rng = np.random.default_rng(seed)
    data = {config.TIME_COL: np.sort(rng.uniform(0, 48 * 3600, n_rows))}
    for col in config.PCA_FEATURES:
        data[col] = rng.normal(size=n_rows)
    data[config.AMOUNT_COL] = rng.lognormal(mean=3, sigma=1, size=n_rows)
    labels = np.zeros(n_rows, dtype=int)
    labels[rng.choice(n_rows, n_fraud, replace=False)] = 1
    data[config.TARGET_COL] = labels
    return pd.DataFrame(data)[config.EXPECTED_COLUMNS]


def test_load_data_missing_file_gives_instructions():
    missing = Path(tempfile.gettempdir()) / "khong_ton_tai" / "creditcard.csv"
    try:
        eda.load_data(missing)
    except FileNotFoundError as exc:
        assert "kaggle" in str(exc).lower()
        assert "creditcard.csv" in str(exc)
    else:
        raise AssertionError("Phải ném FileNotFoundError khi thiếu file")


def test_main_returns_1_when_file_missing():
    with tempfile.TemporaryDirectory() as tmp:
        code = eda.main(["--data-path", str(Path(tmp) / "nope.csv"), "--output-dir", tmp])
    assert code == 1


def test_check_schema_detects_missing_column():
    df = make_fixture().drop(columns=["V5"])
    assert eda.check_schema(df)["missing_expected_columns"] == ["V5"]


def test_class_distribution_counts_and_percent():
    df = make_fixture(n_rows=200, n_fraud=10)
    dist = eda.class_distribution(df)
    assert dist["counts"] == {"0": 190, "1": 10}
    assert abs(dist["percent"]["1"] - 5.0) < 1e-9
    assert abs(dist["majority_class_baseline_accuracy"] - 0.95) < 1e-9
    assert abs(dist["imbalance_ratio_legit_per_fraud"] - 19.0) < 1e-9


def test_quality_report_finds_missing_and_duplicates():
    df = make_fixture(n_rows=100, n_fraud=5)
    df.loc[3, "V1"] = np.nan
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True)  # thêm đúng 1 dòng trùng
    quality = eda.data_quality_report(df)
    assert quality["total_missing_values"] == 1
    assert quality["columns_with_missing"] == {"V1": 1}
    assert quality["n_duplicate_rows"] == 1
    assert quality["rows_sorted_by_time"] is False or quality["rows_sorted_by_time"] is True


def test_quality_report_does_not_modify_input():
    df = make_fixture()
    before = df.copy(deep=True)
    eda.data_quality_report(df)
    pd.testing.assert_frame_equal(df, before)


def test_time_bins_cover_every_row_even_at_exact_multiple():
    df = make_fixture(n_rows=50, n_fraud=5)
    df.loc[df.index[-1], config.TIME_COL] = 4 * eda.TIME_BIN_SECONDS  # đúng bội số của khoảng chia
    stats = eda.time_statistics(df)
    assert sum(b["n_transactions"] for b in stats["fraud_rate_by_2h_bin"]) == len(df)


def test_to_jsonable_handles_numpy_and_nan():
    obj = {"a": np.int64(3), "b": np.float64("nan"), "c": np.array([1.0, 2.0]), "d": np.bool_(True)}
    assert eda.to_jsonable(obj) == {"a": 3, "b": None, "c": [1.0, 2.0], "d": True}
    json.dumps(eda.to_jsonable(obj), allow_nan=False)


def test_full_run_creates_figures_and_summary():
    expected_figures = {
        "class_distribution.png", "amount_distribution.png", "amount_by_class.png",
        "time_by_class.png", "correlation_heatmap.png", "correlation_with_class.png",
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "fixture.csv"
        make_fixture().to_csv(csv_path, index=False)
        out_dir = tmp_path / "out"
        assert eda.main(["--data-path", str(csv_path), "--output-dir", str(out_dir)]) == 0

        assert {p.name for p in (out_dir / "figures").glob("*.png")} == expected_figures
        summary = json.loads((out_dir / "metrics" / "eda_summary.json").read_text(encoding="utf-8"))
        assert summary["overview"]["n_columns"] == 31
        assert summary["class_distribution"]["counts"]["1"] == 30
