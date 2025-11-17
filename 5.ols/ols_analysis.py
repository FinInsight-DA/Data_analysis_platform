from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

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
    diagnostics: Optional[pd.DataFrame] = None


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


def _select_entity_rows(
    df: pd.DataFrame,
    *,
    flag_column: Optional[str] = None,
    filter_column: Optional[str] = None,
    filter_values: Optional[Iterable],
) -> pd.DataFrame:
    if flag_column:
        if flag_column not in df.columns:
            raise KeyError(f"'{flag_column}' 컬럼을 찾을 수 없습니다.")
        mask = df[flag_column].astype(bool)
    elif filter_column:
        if filter_column not in df.columns:
            raise KeyError(f"'{filter_column}' 컬럼을 찾을 수 없습니다.")
        if not filter_values:
            raise ValueError("filter_column을 사용할 때는 filter_values를 지정해야 합니다.")
        mask = df[filter_column].isin(filter_values)
    else:
        mask = pd.Series(True, index=df.index)

    subset = df[mask].copy()
    if subset.empty:
        target_desc = flag_column or f"{filter_column} in {list(filter_values)}"
        raise ValueError(f"`{target_desc}` 조건을 만족하는 데이터가 없습니다.")
    return subset


def compute_sentiment_index(
    df: pd.DataFrame,
    *,
    flag_column: Optional[str] = None,
    filter_column: Optional[str] = None,
    filter_values: Optional[Iterable] = None,
) -> pd.DataFrame:
    sentiment_col = "sentiment" if "sentiment" in df.columns else "predicted_sentiment"
    required_cols = {"label", "aspect_category", sentiment_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"필요 컬럼이 없습니다: {missing_cols}")

    subset = _select_entity_rows(
        df,
        flag_column=flag_column,
        filter_column=filter_column,
        filter_values=filter_values,
    )

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


def extract_model_metrics(model: sm.regression.linear_model.RegressionResultsWrapper) -> pd.DataFrame:
    metrics = {
        "observations": float(getattr(model, "nobs", np.nan)),
        "rsquared": float(getattr(model, "rsquared", np.nan)),
        "adj_rsquared": float(getattr(model, "rsquared_adj", np.nan)),
        "f_value": float(getattr(model, "fvalue", np.nan)) if getattr(model, "fvalue", None) is not None else np.nan,
        "f_pvalue": float(getattr(model, "f_pvalue", np.nan)) if getattr(model, "f_pvalue", None) is not None else np.nan,
        "aic": float(getattr(model, "aic", np.nan)),
        "bic": float(getattr(model, "bic", np.nan)),
        "log_likelihood": float(getattr(model, "llf", np.nan)),
        "mse_model": float(getattr(model, "mse_model", np.nan)),
        "mse_resid": float(getattr(model, "mse_resid", np.nan)),
        "rmse": float(np.sqrt(model.mse_resid)) if getattr(model, "mse_resid", None) is not None else np.nan,
        "durbin_watson": float(durbin_watson(model.resid)),
        "condition_number": float(getattr(model, "condition_number", np.nan)),
    }
    return pd.DataFrame([metrics])


def add_lag_features(
    df: pd.DataFrame,
    *,
    target_columns: Optional[Iterable[str]],
    lag_periods: Iterable[int],
) -> tuple[pd.DataFrame, list[str]]:
    unique_periods = sorted({int(p) for p in lag_periods if int(p) > 0})
    if not unique_periods:
        return df, []

    if target_columns is None:
        columns_to_lag = list(df.columns)
    else:
        columns_to_lag = [col for col in target_columns if col in df.columns]

    lagged_column_names: list[str] = []
    df = df.copy()
    for col in columns_to_lag:
        for lag in unique_periods:
            lag_col_name = f"{col}_lag{lag}"
            df[lag_col_name] = df[col].shift(lag)
            lagged_column_names.append(lag_col_name)

    return df, lagged_column_names


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


def fit_ols_model(
    merged: pd.DataFrame,
    add_constant: bool = True,
    feature_subset: Optional[Iterable[str]] = None,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    feature_cols = [col for col in merged.columns if col not in {"date", "daily_return"}]

    if feature_subset:
        filtered_cols = [col for col in feature_cols if col in feature_subset]
        if not filtered_cols:
            raise ValueError(
                f"지정한 피처 {list(feature_subset)} 를 병합 데이터에서 찾을 수 없습니다."
            )
        feature_cols = filtered_cols

    X = merged[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0).astype(float)
    y = merged["daily_return"].astype(float)

    if X.empty or y.empty:
        raise ValueError("회귀에 사용할 데이터가 충분하지 않습니다.")

    if add_constant:
        X = sm.add_constant(X, has_constant="add")

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
    label: str,
    start: Optional[str],
    end: Optional[str],
    output_dir: Optional[Path],
    *,
    flag_column: Optional[str] = None,
    filter_column: Optional[str] = None,
    filter_values: Optional[Iterable] = None,
    export_sentiment_matrix: Optional[Path] = None,
    export_merged_data: Optional[Path] = None,
    feature_columns: Optional[Iterable[str]] = None,
    lag_periods: Optional[Iterable[int]] = None,
    lag_target_columns: Optional[Iterable[str]] = None,
) -> RegressionResult:
    sentiment_df = load_sentiment_data(sentiment_csv)
    sentiment_index = compute_sentiment_index(
        sentiment_df,
        flag_column=flag_column,
        filter_column=filter_column,
        filter_values=filter_values,
    )
    sentiment_matrix = build_sentiment_matrix(sentiment_index)
    lagged_cols: list[str] = []
    if lag_periods:
        sentiment_matrix, lagged_cols = add_lag_features(
            sentiment_matrix,
            target_columns=lag_target_columns,
            lag_periods=lag_periods,
        )
    market_df = load_market_data(market_csv)
    sentiment_for_merge = sentiment_matrix.copy()
    sentiment_for_merge.index = sentiment_for_merge.index.strftime("%Y-%m-%d")
    market_for_merge = market_df.copy()
    market_for_merge["date"] = market_for_merge["date"].dt.strftime("%Y-%m-%d")
    merged = merge_datasets(sentiment_for_merge, market_for_merge)
    merged["date"] = pd.to_datetime(merged["date"])
    merged = filter_period(merged, start, end)

    if merged.empty:
        raise ValueError("조건에 맞는 병합 데이터가 없습니다. 기간 또는 입력 데이터를 확인하세요.")

    if lagged_cols:
        existing_cols = [col for col in lagged_cols if col in merged.columns]
        if existing_cols:
            merged = merged.dropna(subset=existing_cols)
            if merged.empty:
                raise ValueError("시차 변수 생성 후 사용할 데이터가 충분하지 않습니다.")

    if export_sentiment_matrix:
        export_sentiment_matrix.parent.mkdir(parents=True, exist_ok=True)
        sentiment_matrix.reset_index().to_csv(export_sentiment_matrix, index=False, encoding="utf-8-sig")

    if export_merged_data:
        export_merged_data.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(export_merged_data, index=False, encoding="utf-8-sig")

    model = fit_ols_model(merged, feature_subset=feature_columns)
    summary_text = model.summary().as_text()
    coefficients = summarize_coefficients(model)

    output_path = None
    if output_dir:
        suffix = f"{label.replace(' ', '_').lower()}"
        if start or end:
            range_token = f"{start or 'start'}_{end or 'end'}".replace('-', '')
            suffix = f"{suffix}_{range_token}"
        output_path = output_dir / f"{suffix}_ols_summary.txt"

    diagnostics = extract_model_metrics(model)

    result = RegressionResult(
        label=label,
        n_samples=len(merged),
        summary=summary_text,
        output_path=output_path,
        coefficients=coefficients,
        diagnostics=diagnostics,
    )
    if output_path:
        export_summary(result)
    return result


def run_with_split_year(
    sentiment_csv: Path,
    market_csv: Path,
    label_prefix: str,
    split_year: int,
    output_dir: Optional[Path],
    *,
    flag_column: Optional[str] = None,
    filter_column: Optional[str] = None,
    filter_values: Optional[Iterable] = None,
    export_sentiment_matrix: Optional[Path] = None,
    export_merged_data: Optional[Path] = None,
    feature_columns: Optional[Iterable[str]] = None,
    lag_periods: Optional[Iterable[int]] = None,
    lag_target_columns: Optional[Iterable[str]] = None,
) -> Iterable[RegressionResult]:
    boundary = f"{split_year}-01-01"
    full = run_single_regression(
        sentiment_csv,
        market_csv,
        f"{label_prefix} (전체)",
        None,
        None,
        output_dir,
        flag_column=flag_column,
        filter_column=filter_column,
        filter_values=filter_values,
        export_sentiment_matrix=export_sentiment_matrix,
        export_merged_data=export_merged_data,
        feature_columns=feature_columns,
        lag_periods=lag_periods,
        lag_target_columns=lag_target_columns,
    )
    pre = run_single_regression(
        sentiment_csv,
        market_csv,
        f"{label_prefix} ({split_year-1}년 이전)",
        None,
        f"{split_year-1}-12-31",
        output_dir,
        flag_column=flag_column,
        filter_column=filter_column,
        filter_values=filter_values,
        export_sentiment_matrix=export_sentiment_matrix,
        export_merged_data=export_merged_data,
        feature_columns=feature_columns,
        lag_periods=lag_periods,
        lag_target_columns=lag_target_columns,
    )
    post = run_single_regression(
        sentiment_csv,
        market_csv,
        f"{label_prefix} ({split_year}년 이후)",
        boundary,
        None,
        output_dir,
        flag_column=flag_column,
        filter_column=filter_column,
        filter_values=filter_values,
        export_sentiment_matrix=export_sentiment_matrix,
        export_merged_data=export_merged_data,
        feature_columns=feature_columns,
        lag_periods=lag_periods,
        lag_target_columns=lag_target_columns,
    )
    return [full, pre, post]


def run_regression_from_frames(
    sentiment_df: pd.DataFrame,
    market_df: pd.DataFrame,
    label: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    frequency: Optional[str] = None,
    *,
    flag_column: Optional[str] = None,
    filter_column: Optional[str] = None,
    filter_values: Optional[Iterable] = None,
    feature_columns: Optional[Iterable[str]] = None,
    lag_periods: Optional[Iterable[int]] = None,
    lag_target_columns: Optional[Iterable[str]] = None,
) -> RegressionResult:
    sentiment_df = prepare_sentiment_dataframe(sentiment_df.copy())
    sentiment_index = compute_sentiment_index(
        sentiment_df.copy(),
        flag_column=flag_column,
        filter_column=filter_column,
        filter_values=filter_values,
    )
    sentiment_matrix = build_sentiment_matrix(sentiment_index)
    sentiment_matrix.index = pd.to_datetime(sentiment_matrix.index)

    market_prepared = prepare_market_dataframe(market_df.copy(), date_format=None)

    if frequency:
        sentiment_matrix, market_prepared = resample_sentiment_market(
            sentiment_matrix,
            market_prepared,
            frequency=frequency,
        )

    lagged_columns: list[str] = []
    if lag_periods:
        sentiment_matrix, lagged_columns = add_lag_features(
            sentiment_matrix,
            target_columns=lag_target_columns,
            lag_periods=lag_periods,
        )

    sentiment_for_merge = sentiment_matrix.copy()
    sentiment_for_merge.index = sentiment_for_merge.index.strftime("%Y-%m-%d")
    market_prepared["date"] = market_prepared["date"].dt.strftime("%Y-%m-%d")

    merged = merge_datasets(sentiment_for_merge, market_prepared)
    merged["date"] = pd.to_datetime(merged["date"])
    merged = filter_period(merged, start, end)

    if merged.empty:
        raise ValueError("조건에 맞는 병합 데이터가 없습니다. 기간 또는 입력 데이터를 확인하세요.")

    if lagged_columns:
        existing_lag_cols = [col for col in lagged_columns if col in merged.columns]
        if existing_lag_cols:
            merged = merged.dropna(subset=existing_lag_cols)
            if merged.empty:
                raise ValueError("시차 변수 생성 후 사용할 데이터가 충분하지 않습니다.")

    model = fit_ols_model(merged, feature_subset=feature_columns)
    summary_text = model.summary().as_text()
    coefficients = summarize_coefficients(model)

    return RegressionResult(
        label=label,
        n_samples=len(merged),
        summary=summary_text,
        coefficients=coefficients,
        diagnostics=extract_model_metrics(model),
    )


def run_regression_scenarios_from_frames(
    sentiment_df: pd.DataFrame,
    market_df: pd.DataFrame,
    label: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    split_year: Optional[int] = 2023,
    resample_frequency: Optional[str] = "W",
    *,
    flag_column: Optional[str] = None,
    filter_column: Optional[str] = None,
    filter_values: Optional[Iterable] = None,
    feature_columns: Optional[Iterable[str]] = None,
    lag_periods: Optional[Iterable[int]] = None,
    lag_target_columns: Optional[Iterable[str]] = None,
) -> Iterable[Tuple[str, RegressionResult]]:
    scenarios: list[Tuple[str, RegressionResult]] = []

    base_result = run_regression_from_frames(
        sentiment_df=sentiment_df,
        market_df=market_df,
        label=f"{label} (일별)",
        start=start,
        end=end,
        flag_column=flag_column,
        filter_column=filter_column,
        filter_values=filter_values,
        feature_columns=feature_columns,
        lag_periods=lag_periods,
        lag_target_columns=lag_target_columns,
    )
    scenarios.append(("일별 OLS", base_result))

    if split_year:
        pre_end = f"{split_year - 1}-12-31"
        post_start = f"{split_year}-01-01"

        try:
            pre_result = run_regression_from_frames(
                sentiment_df=sentiment_df,
                market_df=market_df,
                label=f"{label} ({split_year}년 이전)",
                start=start,
                end=pre_end,
                flag_column=flag_column,
                filter_column=filter_column,
                filter_values=filter_values,
                feature_columns=feature_columns,
                lag_periods=lag_periods,
                lag_target_columns=lag_target_columns,
            )
            scenarios.append((f"{split_year}년 이전 OLS", pre_result))
        except ValueError:
            pass

        try:
            post_result = run_regression_from_frames(
                sentiment_df=sentiment_df,
                market_df=market_df,
                label=f"{label} ({split_year}년 이후)",
                start=post_start,
                end=end,
                flag_column=flag_column,
                filter_column=filter_column,
                filter_values=filter_values,
                feature_columns=feature_columns,
                lag_periods=lag_periods,
                lag_target_columns=lag_target_columns,
            )
            scenarios.append((f"{split_year}년 이후 OLS", post_result))
        except ValueError:
            pass

    if resample_frequency:
        try:
            resampled_result = run_regression_from_frames(
                sentiment_df=sentiment_df,
                market_df=market_df,
                label=f"{label} ({resample_frequency} 재샘플링)",
                start=start,
                end=end,
                frequency=resample_frequency,
                flag_column=flag_column,
                filter_column=filter_column,
                filter_values=filter_values,
                feature_columns=feature_columns,
                lag_periods=lag_periods,
                lag_target_columns=lag_target_columns,
            )
            scenarios.append((f"{resample_frequency} 재샘플링 OLS", resampled_result))
        except ValueError:
            pass

    return scenarios


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="감성-주가 OLS 회귀 분석 스크립트")
    parser.add_argument("--sentiment-csv", required=True, help="감성 데이터 CSV 경로")
    parser.add_argument("--market-csv", required=True, help="시장 데이터 CSV 경로")
    parser.add_argument(
        "--firm",
        choices=["samsung", "skhynix"],
        help="기존 불리언 플래그 컬럼(is_samsung / is_skhynix)을 사용할 때 지정",
    )
    parser.add_argument("--flag-column", help="불리언 플래그 컬럼명 (예: is_company_a)")
    parser.add_argument("--filter-column", help="특정 값으로 필터링할 컬럼명 (예: company)")
    parser.add_argument(
        "--filter-values",
        nargs="+",
        help="filter-column에 대응하는 포함 값 (여러 값 가능)",
    )
    parser.add_argument("--start-date", help="분석 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="분석 종료일 (YYYY-MM-DD)")
    parser.add_argument("--split-year", type=int, help="연도 기준 분할 회귀 (예: 2023)")
    parser.add_argument("--output-dir", help="요약 저장 디렉터리")
    parser.add_argument("--export-sentiment-matrix", type=Path, help="감성 지수 매트릭스 저장 경로")
    parser.add_argument("--export-merged-data", type=Path, help="회귀 입력 병합 데이터 저장 경로")
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        help="회귀에 사용할 피처 이름 목록. 지정하지 않으면 모든 피처 사용",
    )
    parser.add_argument(
        "--lag-periods",
        nargs="+",
        type=int,
        help="생성할 시차 기간 (예: --lag-periods 1 2)",
    )
    parser.add_argument(
        "--lag-columns",
        nargs="+",
        help="시차 변수를 생성할 감성 컬럼 이름. 지정하지 않으면 모든 감성 컬럼에 적용",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    sentiment_path = _validate_paths(args.sentiment_csv)
    market_path = _validate_paths(args.market_csv)
    output_dir = ensure_output_dir(Path(args.output_dir).expanduser()) if args.output_dir else None

    label_prefix = "Entity"
    flag_column = args.flag_column
    filter_column = args.filter_column
    filter_values = args.filter_values
    lag_periods = args.lag_periods
    lag_columns = args.lag_columns

    if args.firm:
        if args.firm == "samsung":
            flag_column = "is_samsung"
            label_prefix = "Samsung Electronics"
        else:
            flag_column = "is_skhynix"
            label_prefix = "SK Hynix"

    if not flag_column and not filter_column:
        raise ValueError("분석 대상을 지정하기 위해 --flag-column 또는 --filter-column/--filter-values를 입력하세요.")

    if filter_column and not filter_values:
        raise ValueError("--filter-column 사용 시 --filter-values를 함께 지정해야 합니다.")

    results: Iterable[RegressionResult]
    if args.split_year:
        results = run_with_split_year(
            sentiment_path,
            market_path,
            label_prefix,
            args.split_year,
            output_dir,
            flag_column=flag_column,
            filter_column=filter_column,
            filter_values=filter_values,
            export_sentiment_matrix=args.export_sentiment_matrix,
            export_merged_data=args.export_merged_data,
            feature_columns=args.feature_columns,
            lag_periods=lag_periods,
            lag_target_columns=lag_columns,
        )
    else:
        result = run_single_regression(
            sentiment_path,
            market_path,
            label_prefix,
            args.start_date,
            args.end_date,
            output_dir,
            flag_column=flag_column,
            filter_column=filter_column,
            filter_values=filter_values,
            export_sentiment_matrix=args.export_sentiment_matrix,
            export_merged_data=args.export_merged_data,
            feature_columns=args.feature_columns,
            lag_periods=lag_periods,
            lag_target_columns=lag_columns,
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
