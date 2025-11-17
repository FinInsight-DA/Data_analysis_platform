from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

SENTIMENT_MAPPING: Dict[str, float] = {
    "positive": 1,
    "negative": -1,
    "neutral": 0,
    "Positive": 1,
    "Negative": -1,
    "Neutral": 0,
    "": np.nan,
    None: np.nan,
}


@dataclass
class RegressionResult:
    label: str
    n_samples: int
    summary: str
    output_path: Optional[Path] = None
    coefficients: Optional[pd.DataFrame] = None


def summarize_coefficients(model: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    result_df = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t_stat": model.tvalues,
        "p_value": model.pvalues,
    })
    conf_int = model.conf_int()
    conf_int.columns = ["ci_lower", "ci_upper"]
    result_df = pd.concat([result_df, conf_int], axis=1)
    result_df = result_df.reset_index().rename(columns={"index": "variable"})
    return result_df


def _validate_paths(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    return path


def _normalize_date(series: pd.Series, fmt: Optional[str] = None) -> pd.Series:
    if fmt:
        dt = pd.to_datetime(series, format=fmt, errors="coerce")
    else:
        dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d")


def prepare_sentiment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    value_col: Optional[str] = None
    if "sentiment" in df.columns:
        value_col = "sentiment"
    elif "predicted_sentiment" in df.columns:
        value_col = "predicted_sentiment"
    else:
        raise KeyError("`sentiment` 또는 `predicted_sentiment` 컬럼이 필요합니다.")

    df[value_col] = df[value_col].map(SENTIMENT_MAPPING).astype(float)

    missing = df[value_col].isna().sum()
    if missing:
        print(f"[WARN] 감성 수치로 변환되지 않은 행 {missing}개를 제외합니다.")
        df = df.dropna(subset=[value_col])

    if "date" not in df.columns:
        raise KeyError("감성 데이터에 `date` 컬럼이 없습니다.")

    df["date"] = _normalize_date(df["date"])
    return df


def load_sentiment_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return prepare_sentiment_dataframe(df)


def compute_sentiment_index(df: pd.DataFrame, firm_flag: str) -> pd.DataFrame:
    sentiment_col = "sentiment" if "sentiment" in df.columns else "predicted_sentiment"
    required_cols = {firm_flag, "label", "aspect_category", sentiment_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"필요 컬럼이 없습니다: {missing_cols}")

    subset = df[df[firm_flag]].copy()
    if subset.empty:
        raise ValueError(f"`{firm_flag}` 조건을 만족하는 데이터가 없습니다.")

    grouped = (
        subset.groupby(["date", "label", "aspect_category"])[sentiment_col]
        .agg(
            pos_count=lambda s: (s == 1).sum(),
            neg_count=lambda s: (s == -1).sum(),
        )
        .reset_index()
    )

    grouped["total_count"] = grouped["pos_count"] + grouped["neg_count"]
    grouped["sentiment_index"] = np.where(
        grouped["total_count"] > 0,
        (grouped["pos_count"] - grouped["neg_count"]) / grouped["total_count"],
        np.nan,
    )

    grouped["label_aspect"] = grouped["label"].astype(str) + "_" + grouped["aspect_category"].astype(str)
    return grouped


def build_sentiment_matrix(index_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        index_df.pivot(index="date", columns="label_aspect", values="sentiment_index")
        .sort_index()
    )
    pivot.columns = pivot.columns.astype(str)
    return pivot


def prepare_market_dataframe(df: pd.DataFrame, date_format: Optional[str] = None) -> pd.DataFrame:
    if "date" not in df.columns:
        raise KeyError("시장 데이터에 `date` 컬럼이 없습니다.")

    df = df.copy()
    df["date"] = _normalize_date(df["date"], fmt=date_format)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    close_col = None
    for candidate in ["종가", "close", "Close", "closing_price"]:
        if candidate in df.columns:
            close_col = candidate
            break
    if close_col is None:
        raise KeyError("시장 데이터에서 종가 컬럼을 찾을 수 없습니다.")

    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df["daily_return"] = df[close_col].pct_change()

    keep_cols = ["date", "daily_return"]

    if "환율" in df.columns:
        df["환율"] = pd.to_numeric(df["환율"], errors="coerce")
        df["환율증감률"] = df["환율"].pct_change()
        keep_cols.extend(["환율", "환율증감률"])

    market = df[keep_cols].dropna(subset=["daily_return"]).copy()
    market = market.dropna(subset=["date"])
    return market


def load_market_data(csv_path: Path, date_format: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return prepare_market_dataframe(df, date_format=date_format)


def resample_sentiment_market(
    sentiment_matrix: pd.DataFrame,
    market_df: pd.DataFrame,
    frequency: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sentiment_resampled = sentiment_matrix.resample(frequency).mean()
    market_resampled = market_df.set_index("date").resample(frequency).mean()

    if "환율" in market_resampled.columns:
        market_resampled["환율증감률"] = market_resampled["환율"].pct_change()

    market_resampled = market_resampled.dropna(subset=["daily_return"])
    market_resampled = market_resampled.dropna(how="any")
    market_resampled = market_resampled.reset_index()
    return sentiment_resampled, market_resampled


def merge_datasets(sentiment_matrix: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    sentiment_reset = sentiment_matrix.reset_index()
    merged = pd.merge(sentiment_reset, market_df, on="date", how="inner")
    merged = merged.sort_values("date")
    return merged


def fit_ols_model(merged: pd.DataFrame, add_constant: bool = True) -> sm.regression.linear_model.RegressionResultsWrapper:
    feature_cols = [col for col in merged.columns if col not in {"date", "daily_return"}]
    X = merged[feature_cols].fillna(0.0).astype(float)
    if add_constant:
        X = sm.add_constant(X, has_constant="add")
    y = merged["daily_return"].astype(float)

    if len(X) != len(y) or len(X) == 0:
        raise ValueError("회귀에 사용할 데이터가 충분하지 않습니다.")

    model = sm.OLS(y, X).fit()
    return model


def filter_period(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        start_dt = pd.to_datetime(start)
        df = df[df["date"] >= start_dt]
    if end:
        end_dt = pd.to_datetime(end)
        df = df[df["date"] <= end_dt]
    return df


def ensure_output_dir(output_dir: Optional[Path]) -> Optional[Path]:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export_summary(result: RegressionResult) -> None:
    if result.output_path is None:
        return
    if result.output_path.suffix:
        result.output_path.parent.mkdir(parents=True, exist_ok=True)
        result.output_path.write_text(result.summary, encoding="utf-8")
    else:
        raise ValueError("`output_path`는 파일 경로여야 합니다.")


def run_single_regression(
    sentiment_csv: Path,
    market_csv: Path,
    firm_flag: str,
    label: str,
    start: Optional[str],
    end: Optional[str],
    output_dir: Optional[Path],
) -> RegressionResult:
    sentiment_df = load_sentiment_data(sentiment_csv)
    sentiment_index = compute_sentiment_index(sentiment_df, firm_flag)
    sentiment_matrix = build_sentiment_matrix(sentiment_index)
    market_df = load_market_data(market_csv)
    merged = merge_datasets(sentiment_matrix, market_df)
    merged = filter_period(merged, start, end)

    if merged.empty:
        raise ValueError("조건에 맞는 병합 데이터가 없습니다. 기간 또는 입력 데이터를 확인하세요.")

    model = fit_ols_model(merged)
    summary_text = model.summary().as_text()
    coefficients = summarize_coefficients(model)

    output_path = None
    if output_dir:
        suffix = f"{label.replace(' ', '_').lower()}"
        if start or end:
            range_token = f"{start or 'start'}_{end or 'end'}".replace('-', '')
            suffix = f"{suffix}_{range_token}"
        output_path = output_dir / f"{suffix}_ols_summary.txt"

    result = RegressionResult(
        label=label,
        n_samples=len(merged),
        summary=summary_text,
        output_path=output_path,
        coefficients=coefficients,
    )
    if output_path:
        export_summary(result)
    return result


def run_with_split_year(
    sentiment_csv: Path,
    market_csv: Path,
    firm_flag: str,
    label_prefix: str,
    split_year: int,
    output_dir: Optional[Path],
) -> Iterable[RegressionResult]:
    boundary = f"{split_year}-01-01"
    full = run_single_regression(
        sentiment_csv,
        market_csv,
        firm_flag,
        f"{label_prefix} (전체)",
        None,
        None,
        output_dir,
    )
    pre = run_single_regression(
        sentiment_csv,
        market_csv,
        firm_flag,
        f"{label_prefix} ({split_year-1}년 이전)",
        None,
        f"{split_year-1}-12-31",
        output_dir,
    )
    post = run_single_regression(
        sentiment_csv,
        market_csv,
        firm_flag,
        f"{label_prefix} ({split_year}년 이후)",
        boundary,
        None,
        output_dir,
    )
    return [full, pre, post]


def run_regression_from_frames(
    sentiment_df: pd.DataFrame,
    market_df: pd.DataFrame,
    firm_flag: str,
    label: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    frequency: Optional[str] = None,
) -> RegressionResult:
    sentiment_df = prepare_sentiment_dataframe(sentiment_df.copy())
    sentiment_index = compute_sentiment_index(sentiment_df.copy(), firm_flag)
    sentiment_matrix = build_sentiment_matrix(sentiment_index)
    sentiment_matrix.index = pd.to_datetime(sentiment_matrix.index)

    market_prepared = prepare_market_dataframe(market_df.copy(), date_format=None)

    if frequency:
        sentiment_matrix, market_prepared = resample_sentiment_market(
            sentiment_matrix,
            market_prepared,
            frequency=frequency,
        )

    sentiment_for_merge = sentiment_matrix.copy()
    sentiment_for_merge.index = sentiment_for_merge.index.strftime("%Y-%m-%d")
    market_prepared["date"] = market_prepared["date"].dt.strftime("%Y-%m-%d")

    merged = merge_datasets(sentiment_for_merge, market_prepared)
    merged["date"] = pd.to_datetime(merged["date"])
    merged = filter_period(merged, start, end)

    if merged.empty:
        raise ValueError("조건에 맞는 병합 데이터가 없습니다. 기간 또는 입력 데이터를 확인하세요.")

    model = fit_ols_model(merged)
    summary_text = model.summary().as_text()
    coefficients = summarize_coefficients(model)

    return RegressionResult(
        label=label,
        n_samples=len(merged),
        summary=summary_text,
        coefficients=coefficients,
    )


def run_regression_scenarios_from_frames(
    sentiment_df: pd.DataFrame,
    market_df: pd.DataFrame,
    firm_flag: str,
    label: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    split_year: Optional[int] = 2023,
    resample_frequency: Optional[str] = "W",
) -> Iterable[Tuple[str, RegressionResult]]:
    scenarios: list[Tuple[str, RegressionResult]] = []

    base_result = run_regression_from_frames(
        sentiment_df=sentiment_df,
        market_df=market_df,
        firm_flag=firm_flag,
        label=f"{label} (일별)",
        start=start,
        end=end,
    )
    scenarios.append(("일별 OLS", base_result))

    if split_year:
        pre_end = f"{split_year - 1}-12-31"
        post_start = f"{split_year}-01-01"

        try:
            pre_result = run_regression_from_frames(
                sentiment_df=sentiment_df,
                market_df=market_df,
                firm_flag=firm_flag,
                label=f"{label} ({split_year}년 이전)",
                start=start,
                end=pre_end,
            )
            scenarios.append((f"{split_year}년 이전 OLS", pre_result))
        except ValueError:
            pass

        try:
            post_result = run_regression_from_frames(
                sentiment_df=sentiment_df,
                market_df=market_df,
                firm_flag=firm_flag,
                label=f"{label} ({split_year}년 이후)",
                start=post_start,
                end=end,
            )
            scenarios.append((f"{split_year}년 이후 OLS", post_result))
        except ValueError:
            pass

    if resample_frequency:
        try:
            resampled_result = run_regression_from_frames(
                sentiment_df=sentiment_df,
                market_df=market_df,
                firm_flag=firm_flag,
                label=f"{label} ({resample_frequency} 재샘플링)",
                start=start,
                end=end,
                frequency=resample_frequency,
            )
            scenarios.append((f"{resample_frequency} 재샘플링 OLS", resampled_result))
        except ValueError:
            pass

    return scenarios


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="감성-주가 OLS 회귀 분석 스크립트")
    parser.add_argument("--sentiment-csv", required=True, help="감성 데이터 CSV 경로")
    parser.add_argument("--market-csv", required=True, help="시장 데이터 CSV 경로")
    parser.add_argument("--firm", choices=["samsung", "skhynix"], required=True, help="기업 선택")
    parser.add_argument("--start-date", help="분석 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="분석 종료일 (YYYY-MM-DD)")
    parser.add_argument("--split-year", type=int, help="연도 기준 분할 회귀 (예: 2023)")
    parser.add_argument("--output-dir", help="요약 저장 디렉터리")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    sentiment_path = _validate_paths(args.sentiment_csv)
    market_path = _validate_paths(args.market_csv)
    output_dir = ensure_output_dir(Path(args.output_dir).expanduser()) if args.output_dir else None

    if args.firm == "samsung":
        firm_flag = "is_samsung"
        label_prefix = "Samsung Electronics"
    else:
        firm_flag = "is_skhynix"
        label_prefix = "SK Hynix"

    results: Iterable[RegressionResult]
    if args.split_year:
        results = run_with_split_year(
            sentiment_path,
            market_path,
            firm_flag,
            label_prefix,
            args.split_year,
            output_dir,
        )
    else:
        result = run_single_regression(
            sentiment_path,
            market_path,
            firm_flag,
            label_prefix,
            args.start_date,
            args.end_date,
            output_dir,
        )
        results = [result]

    for res in results:
        print("=" * 80)
        print(f"[{res.label}] 표본 수: {res.n_samples}")
        if res.output_path:
            print(f"요약 저장: {res.output_path}")
        print(res.summary)


if __name__ == "__main__":
    main()

