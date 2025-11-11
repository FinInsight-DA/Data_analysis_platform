from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import FinanceDataReader as fdr


def _normalize_date(date_str: Optional[str], fallback: str) -> str:
    if not date_str:
        return fallback
    return date_str


def resolve_stock_code(company: str) -> Tuple[str, str]:
    company = company.strip()
    listings = fdr.StockListing("KRX")

    if "Name" not in listings.columns:
        raise RuntimeError("상장 리스트에서 회사명(Name) 컬럼을 찾을 수 없습니다.")

    if "Symbol" in listings.columns:
        symbol_col = "Symbol"
    elif "Code" in listings.columns:
        symbol_col = "Code"
    else:
        raise RuntimeError("상장 리스트에 종목 코드를 나타내는 컬럼(Symbol/Code)이 없습니다.")

    if company.isdigit() and len(company) == 6:
        row = listings[listings[symbol_col] == company]
        if row.empty:
            raise ValueError(f"'{company}'에 해당하는 종목을 찾을 수 없습니다.")
        return row.iloc[0]["Name"], company

    row = listings[listings["Name"].str.contains(company, case=False, na=False)]
    if row.empty:
        raise ValueError(f"'{company}'에 해당하는 종목을 찾을 수 없습니다.")
    return row.iloc[0]["Name"], row.iloc[0][symbol_col]


def fetch_stock_data(stock_code: str, start: str, end: str) -> pd.DataFrame:
    df = fdr.DataReader(stock_code, start, end)
    if df.empty:
        raise RuntimeError("시세 데이터를 가져오지 못했습니다.")
    df = df.rename(
        columns={
            "Open": "시가",
            "High": "고가",
            "Low": "저가",
            "Close": "종가",
            "Volume": "거래량",
        }
    )
    df = df.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[["date", "시가", "고가", "저가", "종가", "거래량"]]
    return df


def fetch_exchange_rate(start: str, end: str) -> pd.DataFrame:
    df = fdr.DataReader('USD/KRW', start, end)
    if df.empty:
        raise RuntimeError("환율 데이터를 가져오지 못했습니다.")
    df = df[["Close"]].rename(columns={"Close": "환율"})
    df = df.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def merge_stock_and_fx(stock_df: pd.DataFrame, fx_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(stock_df, fx_df, on="date", how="left")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def crawl_market_data(
    company: str,
    start: Optional[str],
    end: Optional[str],
    *,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    company_name, stock_code = resolve_stock_code(company)
    print(f"[INFO] 대상 종목: {company_name} ({stock_code})")

    today_str = datetime.today().strftime("%Y-%m-%d")
    start_str = _normalize_date(start, "1990-01-01")
    end_str = _normalize_date(end, today_str)

    stock_df = fetch_stock_data(stock_code, start_str, end_str)
    fx_df = fetch_exchange_rate(start_str, end_str)
    merged = merge_stock_and_fx(stock_df, fx_df)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] 저장 완료: {save_path}")
    return merged


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinanceDataReader 기반 일별 시세 크롤러")
    parser.add_argument("--company", required=True, help="회사명 또는 6자리 종목코드")
    parser.add_argument("--start-date", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, help="저장할 CSV 경로 (선택)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    try:
        crawl_market_data(
            company=args.company,
            start=args.start_date,
            end=args.end_date,
            save_path=args.output,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

