import pandas as pd
import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
import numpy as np

# 같은 디렉토리의 모듈을 임포트하기 위해 경로 추가
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from crawl_market_data import crawl_market_data, resolve_stock_code
from ols_analysis import run_regression_scenarios_from_frames


def main():
    # --- 페이지 타이틀 ---
    st.markdown('<div class="main-header">감성·토픽 기반 OLS 회귀 분석</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                font-weight: bold;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
            }
            .sub-header {
                font-size: 1.5rem;
                font-weight: bold;
                color: #2c3e50;
                margin-top: 2rem;
                margin-bottom: 1rem;
            }
        </style>
        """, unsafe_allow_html=True)

    # --- 데이터 업로드 ---
    uploaded_sentiment = st.file_uploader(
        "감성 라벨이 포함된 CSV 업로드", type="csv"
    )

    sentiment_df: pd.DataFrame | None = None
    lookup_pressed = False
    submitted = False

    def create_company_column_from_flags(df: pd.DataFrame) -> pd.DataFrame:
        """
        is_samsung과 is_skhynix 컬럼을 이용해서 company 컬럼 생성
        - 둘 다 True: "both"
        - is_samsung만 True: "Samsung Electronics"
        - is_skhynix만 True: "SK Hynix"
        - 둘 다 False: None
        """
        if "company" in df.columns:
            return df
        
        has_samsung = "is_samsung" in df.columns
        has_skhynix = "is_skhynix" in df.columns
        
        if not (has_samsung or has_skhynix):
            return df
        
        # 불리언 값 정규화 (True/False, 1/0, "True"/"False" 등 처리)
        def normalize_bool(val):
            if pd.isna(val):
                return False
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            if isinstance(val, str):
                return val.lower() in ['true', '1', 'yes', 't']
            return False
        
        samsung_flags = df["is_samsung"].apply(normalize_bool) if has_samsung else pd.Series([False] * len(df))
        skhynix_flags = df["is_skhynix"].apply(normalize_bool) if has_skhynix else pd.Series([False] * len(df))
        
        # company 컬럼 생성
        def determine_company(samsung, skhynix):
            if samsung and skhynix:
                return "both"
            elif samsung:
                return "Samsung Electronics"
            elif skhynix:
                return "SK Hynix"
            else:
                return None
        
        df["company"] = [determine_company(s, h) for s, h in zip(samsung_flags, skhynix_flags)]
        return df

    if uploaded_sentiment is not None:
        try:
            sentiment_df = pd.read_csv(uploaded_sentiment)
            
            # inp_date를 date로 변환 (없으면 그대로 유지)
            if "inp_date" in sentiment_df.columns and "date" not in sentiment_df.columns:
                sentiment_df["date"] = sentiment_df["inp_date"]
            elif "inp_date" in sentiment_df.columns and "date" in sentiment_df.columns:
                # 둘 다 있으면 date 우선 사용 (또는 inp_date로 덮어쓰기)
                sentiment_df["date"] = sentiment_df["inp_date"]
            
            # company 컬럼이 없으면 is_samsung/is_skhynix로 생성
            sentiment_df = create_company_column_from_flags(sentiment_df)
                
        except Exception as exc:
            st.error(f"감성 CSV를 읽는 중 오류가 발생했습니다: {exc}")

    # ==============================
    # 1. 입력 폼 (기간, 옵션 등)
    # ==============================
    with st.form("analysis_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input("시작일 (옵션)", value=None)
        with col2:
            end_date = st.date_input("종료일 (옵션)", value=None)
        with col3:
            split_year_enabled = st.checkbox("연도 분할 회귀", value=False)
            split_year = st.number_input(
                "분할 연도", min_value=1900, max_value=2100, value=2025
            )

        resample_option = st.selectbox(
            "재샘플링 빈도",
            ["없음", "주간 (W)", "월간 (M)"],
            index=1,
            disabled=sentiment_df is None,
        )

        regression_mode = st.radio(
            "회귀 모드",
            ["다중 회귀", "단순 회귀"],
            index=0,
            horizontal=True,
            disabled=sentiment_df is None,
            key="regression_mode_radio",
        )

        # 아래 변수들은 폼 내에서 단계적으로 채워짐
        filter_column: str | None = None
        filter_values: list[str] = []
        filter_labels: dict[str, str] = {}
        market_name_inputs: dict[str, str] = {}
        candidate_features: list[str] = []
        lag_generated_feature_names: list[str] = []
        lag_periods_selected: list[int] = []
        lag_target_columns_selected: list[str] = []
        lag_enabled = False

        BOTH_ALIAS = "Both"

        # ==============================
        # 2. company 선택 및 감성 피처 후보 생성
        # ==============================
        if sentiment_df is not None:
            if "company" not in sentiment_df.columns:
                st.error("CSV에 `company` 컬럼이 없습니다. 업로드 파일을 확인하세요.")
            else:
                filter_column = "company"
                raw_values = (
                    sentiment_df[filter_column].dropna().astype(str).unique()
                )
                all_company_values = sorted(raw_values)
                unique_values = [
                    val for val in all_company_values if val != BOTH_ALIAS
                ][:500]

                if not unique_values:
                    st.error(
                        "CSV의 `company` 컬럼에서 사용할 수 있는 값이 없습니다."
                    )
                    st.stop()

                default_selection = (
                    unique_values[:2]
                    if len(unique_values) >= 2
                    else unique_values
                )
                filter_values = st.multiselect(
                    "분석할 회사 선택 (최대 500개)",
                    unique_values,
                    default=default_selection,
                )

                for val in filter_values:
                    filter_labels[val] = st.text_input(
                        f"{filter_column} = {val} 결과 라벨",
                        value=str(val),
                        key=f"filter_label_{filter_column}_{val}",
                    )
                    market_name_inputs[val] = st.text_input(
                        f"{val} 주가 조회용 상장사명 (콤마/줄바꿈 구분)",
                        value=str(val),
                        key=f"market_fetch_{filter_column}_{val}",
                        help="한국거래소 상장사 이름 그대로 입력하세요. 예: 삼성전자",
                    )

                # label + aspect_category 조합을 감성 피처 후보로 사용
                if {"label", "aspect_category"}.issubset(sentiment_df.columns):
                    label_aspect_series = (
                        sentiment_df["label"].astype(str)
                        + "_"
                        + sentiment_df["aspect_category"].astype(str)
                    )
                    candidate_features = sorted(
                        label_aspect_series.dropna().unique().tolist()
                    )

                # ==============================
                # 3. 시차(라그) 변수 옵션
                # ==============================
                if candidate_features:
                    lag_enabled = st.checkbox(
                        "시차(라그) 변수 추가",
                        value=False,
                        key="lag_toggle",
                        help="감성 지수 컬럼에 대한 과거 값(시차)을 추가 변수로 생성합니다.",
                    )
                    if lag_enabled:
                        lag_target_columns_selected = st.multiselect(
                            "시차를 적용할 감성 컬럼 선택",
                            options=candidate_features,
                            help="선택한 컬럼에 대해 지정한 시차만큼 과거 값을 함께 회귀 변수로 추가합니다. 비워두면 전체 감성 컬럼에 적용합니다.",
                            key="lag_target_columns",
                        )
                        lag_options = list(range(1, 13))
                        lag_periods_selected = st.multiselect(
                            "적용할 시차 기간 (예: 1은 1주/1개월 전)",
                            options=lag_options,
                            default=[1],
                            key="lag_periods",
                        )
                        columns_for_lag_names = (
                            lag_target_columns_selected
                            if lag_target_columns_selected
                            else candidate_features
                        )
                        for col in columns_for_lag_names:
                            for lag in lag_periods_selected:
                                lag_generated_feature_names.append(
                                    f"{col}_lag{lag}"
                                )
                        lag_generated_feature_names = sorted(
                            set(lag_generated_feature_names)
                        )

        # ==============================
        # 4. 회귀 모드별 피처 선택
        # ==============================
        selected_feature_names: list[str] = []
        single_feature_choice: str | None = None

        if regression_mode == "단순 회귀" and sentiment_df is not None:
            combined_options = sorted(
                set(candidate_features + lag_generated_feature_names)
            )
            single_feature_options = ["(선택 없음)"] + combined_options
            single_feature_choice = st.selectbox(
                "단순 회귀에 사용할 컬럼을 선택하세요",
                options=single_feature_options,
                index=0,
                key="single_reg_feature_select",
                help="감성 지수 매트릭스에 존재하는 컬럼 이름과 일치해야 합니다.",
            )
            if single_feature_choice and single_feature_choice != "(선택 없음)":
                selected_feature_names = [single_feature_choice]
        elif regression_mode == "다중 회귀" and sentiment_df is not None:
            selected_feature_names = st.multiselect(
                "다중 회귀에 사용할 컬럼 선택 (선택하지 않으면 전체 사용)",
                options=sorted(
                    set(candidate_features + lag_generated_feature_names)
                ),
                help="감성 지수 매트릭스에 존재하는 컬럼 이름과 일치해야 합니다.",
                key="multi_reg_feature_select",
            )

        # 폼 버튼
        lookup_pressed = st.form_submit_button(
            "선택한 회사 종목 코드 확인", type="secondary"
        )
        submitted = st.form_submit_button("회귀 분석 실행")

    # ==============================
    # 5. 회귀 분석 실행 버튼 로직
    # ==============================
    if submitted:
        if sentiment_df is None:
            st.error("감성 CSV를 업로드해 주세요.")
        else:
            start_str = start_date.strftime("%Y-%m-%d") if start_date else None
            end_str = end_date.strftime("%Y-%m-%d") if end_date else None

            # 재샘플링 옵션
            resample_freq = None
            if resample_option == "주간 (W)":
                resample_freq = "W"
            elif resample_option == "월간 (M)":
                resample_freq = "M"

            # 회귀 모드별 피처 결정
            if regression_mode == "단순 회귀":
                if not selected_feature_names:
                    st.error("단순 회귀 모드에서는 컬럼을 1개 선택해야 합니다.")
                    st.stop()
                selected_column = selected_feature_names[0]
                feature_columns = [selected_column]
                st.info(f"단순 회귀 컬럼: {selected_column}")
            else:
                feature_columns = (
                    selected_feature_names if selected_feature_names else None
                )

            # 시차 옵션 정리
            lag_periods_effective: list[int] | None = None
            lag_target_columns_effective: list[str] | None = None
            if lag_enabled and (lag_periods_selected or lag_target_columns_selected):
                lag_periods_effective = sorted(
                    {int(p) for p in lag_periods_selected if int(p) > 0}
                )
                lag_target_columns_effective = sorted(
                    set(lag_target_columns_selected)
                )

            # 선택한 피처 이름에 _lagX 가 들어있으면 자동으로 시차 대상/기간 보정
            selected_columns_iterable = (
                feature_columns if feature_columns is not None else []
            )
            for col in selected_columns_iterable:
                base, sep, lag_str = col.rpartition("_lag")
                if sep and lag_str.isdigit():
                    if lag_target_columns_effective is None:
                        lag_target_columns_effective = []
                    if base not in lag_target_columns_effective:
                        lag_target_columns_effective.append(base)
                    if lag_periods_effective is None:
                        lag_periods_effective = []
                    lag_value = int(lag_str)
                    if lag_value not in lag_periods_effective:
                        lag_periods_effective.append(lag_value)

            lag_target_display: str | None = None
            if lag_target_columns_effective:
                lag_target_columns_effective = sorted(
                    set(lag_target_columns_effective)
                )
                lag_target_display = ", ".join(lag_target_columns_effective)
            elif lag_periods_effective:
                lag_target_display = "전체 감성 컬럼"
                lag_target_columns_effective = None
            if lag_periods_effective:
                lag_periods_effective = sorted(set(lag_periods_effective))
            else:
                lag_periods_effective = None

            if lag_periods_effective and lag_target_display:
                lag_desc = ", ".join(str(p) for p in lag_periods_effective)
                st.info(
                    f"시차 변수 적용: {lag_target_display}에 대해 "
                    f"{lag_desc} 기간의 시차를 추가합니다."
                )

            # ==============================
            # 6. 타깃(회사)별 회귀 수행
            # ==============================
            targets: list[tuple[str, dict]] = []

            if filter_column and filter_values:
                available_companies = set(
                    sentiment_df[filter_column].dropna().astype(str).unique()
                )
                for val in filter_values:
                    raw_val = str(val)
                    label = filter_labels.get(val) or raw_val

                    if raw_val not in available_companies:
                        st.warning(
                            f"`{raw_val}` 값을 `company` 컬럼에서 찾을 수 없습니다. "
                            "CSV의 회사명이 한국거래소 상장명과 일치하는지 확인해주세요."
                        )
                        continue

                    targets.append(
                        (
                            label,
                            {
                                "flag_column": None,
                                "filter_column": filter_column,
                                "filter_values": [raw_val]
                                + (
                                    [BOTH_ALIAS]
                                    if BOTH_ALIAS in available_companies
                                    else []
                                ),
                                "company_key": raw_val,
                            },
                        )
                    )
            else:
                st.error("분석할 회사를 선택해 주세요.")

            # ==============================
            # 6-1. 여러 회사 주가 비교 시각화 (2개 이상 선택 시)
            # ==============================
            all_market_data = {}  # 회사별 시장 데이터 저장
            
            # --- 각 회사별 실행 ---
            for label, selector in targets:
                try:
                    # 시장 데이터 크롤링
                    with st.spinner(f"{label} 시장 데이터 수집 중..."):
                        market_key = selector.get("company_key")
                        market_entries = (
                            market_name_inputs.get(market_key, "")
                            if market_key
                            else ""
                        )
                        parsed_market_names = [
                            name.strip()
                            for name in market_entries.replace("\n", ",").split(",")
                            if name.strip()
                        ]

                        if not parsed_market_names:
                            raise ValueError(
                                f"{label}에 대해 주가 조회용 상장사명을 입력하지 않았습니다."
                            )

                        primary_market_name = parsed_market_names[0]
                        market_df = crawl_market_data(
                            company=primary_market_name,
                            start=start_str,
                            end=end_str,
                            save_path=None,
                        )
                        
                        # 주가 비교를 위해 저장
                        all_market_data[label] = market_df.copy()

                    # 회귀 시나리오 실행
                    scenario_results = run_regression_scenarios_from_frames(
                        sentiment_df=sentiment_df,
                        market_df=market_df,
                        label=label,
                        start=start_str,
                        end=end_str,
                        split_year=split_year if split_year_enabled else None,
                        resample_frequency=resample_freq,
                        flag_column=selector["flag_column"],
                        filter_column=selector["filter_column"],
                        filter_values=selector["filter_values"],
                        feature_columns=feature_columns,
                        lag_periods=lag_periods_effective,
                        lag_target_columns=lag_target_columns_effective,
                    )
                except Exception as exc:
                    st.error(f"{label} 처리 중 오류가 발생했습니다: {exc}")
                    continue

                # ==============================
                # 7. 결과 표시 (텍스트/테이블 먼저)
                # ==============================
                st.markdown(f"## {label}")

                # 시각화를 저장할 리스트
                visualization_figures = []
                visualization_titles = []

                # scenario_name, result 구조
                for scenario_name, result in scenario_results:
                    st.markdown(f"### {scenario_name}")
                    st.write(f"표본 수: {result.n_samples}")
                    safe_name = (
                        scenario_name.replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                    )

                    # 회귀 요약 텍스트
                    with st.expander("회귀 요약 보기", expanded=True):
                        st.markdown(f"```\n{result.summary}\n```")
                        st.download_button(
                            label=f"{scenario_name} 회귀 요약 TXT 다운로드",
                            data=result.summary,
                            file_name=f"summary_{label}_{safe_name}.txt",
                            mime="text/plain",
                        )

                # 마지막 시나리오의 result를 사용 (진단/계수 표시는 공통 구조라고 가정)
                if result.diagnostics is not None:
                    st.markdown("#### 회귀 요약 지표")
                    st.dataframe(
                        result.diagnostics.round(6), use_container_width=True
                    )
                    st.download_button(
                        label=f"{scenario_name} 요약 지표 CSV 다운로드",
                        data=result.diagnostics.to_csv(
                            index=False, encoding="utf-8-sig"
                        ),
                        file_name=f"summary_metrics_{label}_{safe_name}.csv",
                        mime="text/csv",
                    )

                    if result.coefficients is not None:
                        st.markdown("#### 회귀 계수")
                        st.dataframe(
                            result.coefficients.round(6),
                            use_container_width=True,
                        )
                        st.download_button(
                            label=f"{scenario_name} 계수 CSV 다운로드",
                            data=result.coefficients.to_csv(
                                index=False, encoding="utf-8-sig"
                            ),
                            file_name=f"coefficients_{label}_{safe_name}.csv",
                            mime="text/csv",
                        )

                        # 감성 변수 영향 요약 테이블
                        sentiment_columns_set = set(
                            candidate_features + lag_generated_feature_names
                        )
                        sentiment_effects = result.coefficients[
                            result.coefficients["variable"].isin(
                                sentiment_columns_set
                            )
                        ].copy()

                        if not sentiment_effects.empty:
                            sentiment_effects["영향 방향"] = sentiment_effects[
                                "coef"
                            ].apply(
                                lambda v: "양(+) 영향" if v > 0 else "음(-) 영향"
                            )

                            def _significance_label(p: float) -> str:
                                if p < 0.01:
                                    return "*** (p < 0.01)"
                                if p < 0.05:
                                    return "** (p < 0.05)"
                                if p < 0.1:
                                    return "* (p < 0.10)"
                                return "ns"

                            sentiment_effects["유의성"] = sentiment_effects[
                                "p_value"
                            ].apply(_significance_label)
                            sentiment_effects = sentiment_effects.sort_values(
                                "p_value"
                            )

                            display_cols = sentiment_effects[
                                [
                                    "variable",
                                    "coef",
                                    "p_value",
                                    "영향 방향",
                                    "유의성",
                                    "ci_lower",
                                    "ci_upper",
                                ]
                            ].rename(
                                columns={
                                    "variable": "컬럼",
                                    "coef": "계수",
                                    "p_value": "p-value",
                                    "ci_lower": "신뢰구간 하한",
                                    "ci_upper": "신뢰구간 상한",
                                }
                            )

                            st.markdown("#### 감성 지수가 종속변수에 미치는 영향 요약")
                            st.dataframe(
                                display_cols.round(6), use_container_width=True
                            )
                            st.caption(
                                "양(+) 영향은 종속변수를 증가시키는 방향, 음(-) 영향은 감소시키는 방향을 의미합니다. "
                                "유의성 표기는 통계적 유의수준을 의미합니다."
                            )

                        # -------------------------
                        # 계수 기반 시각화 (감성 변수만) - 나중에 표시하기 위해 저장
                        # -------------------------
                        coef_df = result.coefficients.copy()
                        sentiment_coefs = coef_df[
                            (coef_df["variable"] != "const")
                            & (coef_df["variable"].isin(sentiment_columns_set))
                        ].copy()

                        if not sentiment_coefs.empty:
                            sentiment_coefs = sentiment_coefs.sort_values(
                                "coef", ascending=True
                            )

                            # 양/음 계수 분리
                            fig_coef = go.Figure()

                            positive_coefs = sentiment_coefs[
                                sentiment_coefs["coef"] > 0
                            ]
                            if not positive_coefs.empty:
                                fig_coef.add_trace(
                                    go.Bar(
                                        y=positive_coefs["variable"],
                                        x=positive_coefs["coef"],
                                        name="양(+) 영향",
                                        marker_color="steelblue",
                                        orientation="h",
                                        error_x=dict(
                                            type="data",
                                            array=positive_coefs["coef"]
                                            - positive_coefs["ci_lower"],
                                            arrayminus=positive_coefs["ci_upper"]
                                            - positive_coefs["coef"],
                                            visible=True,
                                        ),
                                        text=[
                                            f"p={p:.3f}"
                                            for p in positive_coefs["p_value"]
                                        ],
                                        textposition="outside",
                                        hovertemplate="<b>%{y}</b><br>계수: %{x:.4f}<br>p-value: %{text}<extra></extra>",
                                    )
                                )

                            negative_coefs = sentiment_coefs[
                                sentiment_coefs["coef"] <= 0
                            ]
                            if not negative_coefs.empty:
                                fig_coef.add_trace(
                                    go.Bar(
                                        y=negative_coefs["variable"],
                                        x=negative_coefs["coef"],
                                        name="음(-) 영향",
                                        marker_color="crimson",
                                        orientation="h",
                                        error_x=dict(
                                            type="data",
                                            array=negative_coefs["coef"]
                                            - negative_coefs["ci_lower"],
                                            arrayminus=negative_coefs["ci_upper"]
                                            - negative_coefs["coef"],
                                            visible=True,
                                        ),
                                        text=[
                                            f"p={p:.3f}"
                                            for p in negative_coefs["p_value"]
                                        ],
                                        textposition="outside",
                                        hovertemplate="<b>%{y}</b><br>계수: %{x:.4f}<br>p-value: %{text}<extra></extra>",
                                    )
                                )

                            fig_coef.update_layout(
                                title=f"{scenario_name} - 회귀 계수 (신뢰구간 포함)",
                                xaxis_title="계수 값",
                                yaxis_title="감성 변수",
                                height=max(400, len(sentiment_coefs) * 30),
                                hovermode="closest",
                                showlegend=True,
                                xaxis=dict(
                                    zeroline=True,
                                    zerolinewidth=2,
                                    zerolinecolor="gray",
                                ),
                            )
                            visualization_figures.append(fig_coef)
                            visualization_titles.append(f"{scenario_name} - 회귀 계수")

                            # p-value 기반 영향도 히트맵(산점도)
                            if len(sentiment_coefs) > 1:
                                heatmap_data = sentiment_coefs[
                                    ["variable", "coef", "p_value"]
                                ].copy()
                                heatmap_data["유의성"] = heatmap_data[
                                    "p_value"
                                ].apply(
                                    lambda p: "***"
                                    if p < 0.01
                                    else (
                                        "**"
                                        if p < 0.05
                                        else ("*" if p < 0.1 else "ns")
                                    )
                                )

                                fig_heatmap = px.scatter(
                                    heatmap_data,
                                    x="coef",
                                    y="variable",
                                    size="p_value",
                                    color="coef",
                                    color_continuous_scale="RdBu",
                                    color_continuous_midpoint=0,
                                    hover_data=["p_value", "유의성"],
                                    title=f"{scenario_name} - 감성 변수 영향도 (크기: p-value, 색상: 계수)",
                                    labels={
                                        "coef": "회귀 계수",
                                        "variable": "감성 변수",
                                        "p_value": "p-value",
                                    },
                                )
                                fig_heatmap.update_layout(
                                    height=max(400, len(sentiment_coefs) * 30),
                                    xaxis=dict(
                                        zeroline=True,
                                        zerolinewidth=2,
                                        zerolinecolor="gray",
                                    ),
                                )
                                visualization_figures.append(fig_heatmap)
                                visualization_titles.append(f"{scenario_name} - 감성 변수 영향도")

                # ==============================
                # 8. 시장 데이터 시각화 준비 (나중에 표시하기 위해 저장)
                # ==============================
                if not market_df.empty and "date" in market_df.columns:
                    market_df_viz = market_df.copy()
                    market_df_viz["date"] = pd.to_datetime(
                        market_df_viz["date"]
                    )
                    market_df_viz = market_df_viz.sort_values("date")

                    # 주가 시계열
                    if "Close" in market_df_viz.columns or "종가" in market_df_viz.columns:
                        price_col = (
                            "Close"
                            if "Close" in market_df_viz.columns
                            else "종가"
                        )
                        fig_price = go.Figure()
                        fig_price.add_trace(
                            go.Scatter(
                                x=market_df_viz["date"],
                                y=market_df_viz[price_col],
                                mode="lines",
                                name="종가",
                                line=dict(color="steelblue", width=2),
                                hovertemplate="날짜: %{x}<br>종가: %{y:,.0f}원<extra></extra>",
                            )
                        )
                        fig_price.update_layout(
                            title=f"{label} 주가 시계열",
                            xaxis_title="날짜",
                            yaxis_title="종가 (원)",
                            hovermode="x unified",
                            height=400,
                        )
                        visualization_figures.append(fig_price)
                        visualization_titles.append(f"{label} 주가 시계열")

                    # 일일 수익률
                    if "daily_return" in market_df_viz.columns:
                        fig_return = go.Figure()
                        colors = [
                            "red"
                            if x < 0
                            else "green"
                            for x in market_df_viz["daily_return"]
                        ]
                        fig_return.add_trace(
                            go.Bar(
                                x=market_df_viz["date"],
                                y=market_df_viz["daily_return"],
                                name="일일 수익률",
                                marker_color=colors,
                                hovertemplate="날짜: %{x}<br>수익률: %{y:.4f}<extra></extra>",
                            )
                        )
                        fig_return.update_layout(
                            title=f"{label} 일일 수익률",
                            xaxis_title="날짜",
                            yaxis_title="일일 수익률",
                            hovermode="x unified",
                            height=400,
                            yaxis=dict(
                                zeroline=True,
                                zerolinewidth=2,
                                zerolinecolor="gray",
                            ),
                        )
                        visualization_figures.append(fig_return)
                        visualization_titles.append(f"{label} 일일 수익률")

                    # 환율 시계열
                    if "USDKRW" in market_df_viz.columns:
                        fig_fx = go.Figure()
                        fig_fx.add_trace(
                            go.Scatter(
                                x=market_df_viz["date"],
                                y=market_df_viz["USDKRW"],
                                mode="lines",
                                name="USD/KRW",
                                line=dict(color="orange", width=2),
                                hovertemplate="날짜: %{x}<br>환율: %{y:,.2f}원<extra></extra>",
                            )
                        )
                        fig_fx.update_layout(
                            title=f"{label} USD/KRW 환율",
                            xaxis_title="날짜",
                            yaxis_title="환율 (원)",
                            hovermode="x unified",
                            height=400,
                        )
                        visualization_figures.append(fig_fx)
                        visualization_titles.append(f"{label} USD/KRW 환율")

                    # 시장 데이터 미리보기
                    with st.expander(f"{label} 시장 데이터 미리보기"):
                        st.dataframe(
                            market_df.tail().sort_values("date"),
                            use_container_width=True,
                        )

                    st.download_button(
                        label=f"{label} 시세·환율 CSV 다운로드",
                        data=market_df.to_csv(
                            index=False, encoding="utf-8-sig"
                        ),
                        file_name=f"market_{label}.csv",
                        mime="text/csv",
                    )

                # ==============================
                # 9. 모든 시각화를 아래쪽에 모아서 표시
                # ==============================
                if visualization_figures:
                    st.markdown("---")
                    st.markdown(f"## {label} 시각화")
                    
                    for fig, title in zip(visualization_figures, visualization_titles):
                        st.markdown(f"### {title}")
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("---")
            
            # ==============================
            # 10. 추가 시각화 (전체 회사 비교)
            # ==============================
            if len(all_market_data) >= 2:
                st.markdown("---")
                st.markdown("## 회사별 주가 비교")
                
                # 주가 비교 차트
                fig_comparison = go.Figure()
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                
                for idx, (company_label, market_data) in enumerate(all_market_data.items()):
                    if not market_data.empty and "date" in market_data.columns:
                        market_data_viz = market_data.copy()
                        market_data_viz["date"] = pd.to_datetime(market_data_viz["date"])
                        market_data_viz = market_data_viz.sort_values("date")
                        
                        # 종가 컬럼 찾기
                        price_col = None
                        if "Close" in market_data_viz.columns:
                            price_col = "Close"
                        elif "종가" in market_data_viz.columns:
                            price_col = "종가"
                        
                        if price_col:
                            fig_comparison.add_trace(
                                go.Scatter(
                                    x=market_data_viz["date"],
                                    y=market_data_viz[price_col],
                                    mode="lines",
                                    name=company_label,
                                    line=dict(color=colors[idx % len(colors)], width=2),
                                    hovertemplate=f"<b>{company_label}</b><br>날짜: %{{x}}<br>종가: %{{y:,.0f}}원<extra></extra>",
                                )
                            )
                
                fig_comparison.update_layout(
                    title="회사별 주가 비교",
                    xaxis_title="날짜",
                    yaxis_title="종가 (원)",
                    hovermode="x unified",
                    height=500,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
            
            # ==============================
            # 11. 워드 클라우드 시각화
            # ==============================
            if sentiment_df is not None:
                st.markdown("---")
                st.markdown("## 워드 클라우드 시각화")
                
                # 키워드 데이터 추출 시도
                keyword_col = None
                if "extracted_keywords" in sentiment_df.columns:
                    keyword_col = "extracted_keywords"
                elif "aspect_term" in sentiment_df.columns:
                    keyword_col = "aspect_term"
                
                if keyword_col:
                    # 감성별 키워드 추출
                    sentiment_col = "sentiment" if "sentiment" in sentiment_df.columns else "predicted_sentiment"
                    
                    if sentiment_col in sentiment_df.columns:
                        # 긍정 키워드
                        positive_df = sentiment_df[
                            sentiment_df[sentiment_col].astype(str).str.lower().isin(["positive", "pos", "1", "1.0"])
                        ]
                        # 부정 키워드
                        negative_df = sentiment_df[
                            sentiment_df[sentiment_col].astype(str).str.lower().isin(["negative", "neg", "-1", "-1.0"])
                        ]
                        
                        def extract_keywords_freq(df, col_name):
                            """키워드 빈도 추출"""
                            all_keywords = []
                            for keywords_str in df[col_name].dropna():
                                if isinstance(keywords_str, str):
                                    # 리스트 형태 문자열 파싱
                                    try:
                                        import ast
                                        keywords = ast.literal_eval(keywords_str)
                                        if isinstance(keywords, list):
                                            all_keywords.extend(keywords)
                                    except:
                                        # 쉼표로 구분된 문자열
                                        keywords = [k.strip() for k in keywords_str.replace("[", "").replace("]", "").replace("'", "").split(",")]
                                        all_keywords.extend([k for k in keywords if k])
                            
                            # 빈도 계산
                            from collections import Counter
                            return dict(Counter(all_keywords))
                        
                        positive_freq = extract_keywords_freq(positive_df, keyword_col)
                        negative_freq = extract_keywords_freq(negative_df, keyword_col)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if positive_freq:
                                st.markdown("### 긍정 키워드 워드 클라우드")
                                try:
                                    wordcloud_pos = WordCloud(
                                        width=800, height=400,
                                        background_color='white',
                                        colormap='Greens',
                                        max_words=100
                                    ).generate_from_frequencies(positive_freq)
                                    
                                    fig_wc_pos, ax_pos = plt.subplots(figsize=(10, 5))
                                    ax_pos.imshow(wordcloud_pos, interpolation='bilinear')
                                    ax_pos.axis("off")
                                    ax_pos.set_title("Positive Keywords", fontsize=16, pad=20)
                                    st.pyplot(fig_wc_pos)
                                    plt.close(fig_wc_pos)
                                except Exception as e:
                                    st.warning(f"긍정 워드 클라우드 생성 실패: {e}")
                            else:
                                st.info("긍정 키워드 데이터가 없습니다.")
                        
                        with col2:
                            if negative_freq:
                                st.markdown("### 부정 키워드 워드 클라우드")
                                try:
                                    wordcloud_neg = WordCloud(
                                        width=800, height=400,
                                        background_color='white',
                                        colormap='Reds',
                                        max_words=100
                                    ).generate_from_frequencies(negative_freq)
                                    
                                    fig_wc_neg, ax_neg = plt.subplots(figsize=(10, 5))
                                    ax_neg.imshow(wordcloud_neg, interpolation='bilinear')
                                    ax_neg.axis("off")
                                    ax_neg.set_title("Negative Keywords", fontsize=16, pad=20)
                                    st.pyplot(fig_wc_neg)
                                    plt.close(fig_wc_neg)
                                except Exception as e:
                                    st.warning(f"부정 워드 클라우드 생성 실패: {e}")
                            else:
                                st.info("부정 키워드 데이터가 없습니다.")
                    else:
                        st.info("감성 컬럼을 찾을 수 없어 워드 클라우드를 생성할 수 없습니다.")
                else:
                    st.info("키워드 컬럼(`extracted_keywords` 또는 `aspect_term`)이 없어 워드 클라우드를 생성할 수 없습니다.")

    # ==============================
    # 12. 종목 코드 확인 버튼 로직
    # ==============================
    if lookup_pressed:
        if sentiment_df is None:
            st.error("감성 CSV를 업로드한 뒤 다시 시도하세요.")
        else:
            st.markdown("### 종목 코드 확인 결과")
            for company_key, entry in market_name_inputs.items():
                names = [
                    name.strip()
                    for name in entry.replace("\n", ",").split(",")
                    if name.strip()
                ]
                if not names:
                    st.warning(f"{company_key}: 상장사명을 입력하지 않았습니다.")
                    continue
                for name in names:
                    try:
                        resolved_name, code = resolve_stock_code(name)
                        st.success(f"{company_key}: {resolved_name} ({code})")
                    except Exception as exc:
                        st.error(
                            f"{company_key}: '{name}' 조회 실패 - {exc}"
                        )


if __name__ == "__main__":
    main()
