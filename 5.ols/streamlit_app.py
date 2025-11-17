import os
import sys
import pandas as pd
import streamlit as st

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from crawl_market_data import crawl_market_data, resolve_stock_code
from ols_analysis import run_regression_scenarios_from_frames


def main():
    st.title("감성·토픽 기반 OLS 회귀 분석")

    uploaded_sentiment = st.file_uploader("감성 라벨이 포함된 CSV 업로드", type="csv")

    sentiment_df: pd.DataFrame | None = None
    lookup_pressed = False
    submitted = False
    if uploaded_sentiment is not None:
        try:
            sentiment_df = pd.read_csv(uploaded_sentiment)
        except Exception as exc:
            st.error(f"감성 CSV를 읽는 중 오류가 발생했습니다: {exc}")

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

        if sentiment_df is not None:
            if "company" not in sentiment_df.columns:
                st.error("CSV에 `company` 컬럼이 없습니다. 업로드 파일을 확인하세요.")
            else:
                filter_column = "company"
                raw_values = sentiment_df[filter_column].dropna().astype(str).unique()
                all_company_values = sorted(raw_values)
                unique_values = [val for val in all_company_values if val != BOTH_ALIAS][
                    :500
                ]

                if not unique_values:
                    st.error("CSV의 `company` 컬럼에서 사용할 수 있는 값이 없습니다.")
                    st.stop()

                default_selection = (
                    unique_values[:2] if len(unique_values) >= 2 else unique_values
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

                # 감성/애스펙트 조합으로 피처 후보 만들기
                if {"label", "aspect_category"}.issubset(sentiment_df.columns):
                    label_aspect_series = (
                        sentiment_df["label"].astype(str)
                        + "_"
                        + sentiment_df["aspect_category"].astype(str)
                    )
                    candidate_features = sorted(
                        label_aspect_series.dropna().unique().tolist()
                    )

                # 라그 변수 설정
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
                                lag_generated_feature_names.append(f"{col}_lag{lag}")
                        lag_generated_feature_names = sorted(
                            set(lag_generated_feature_names)
                        )

        # -------------------- 회귀 변수 선택 --------------------
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
                options=sorted(set(candidate_features + lag_generated_feature_names)),
                help="감성 지수 매트릭스에 존재하는 컬럼 이름과 일치해야 합니다.",
                key="multi_reg_feature_select",
            )

        lookup_pressed = st.form_submit_button("선택한 회사 종목 코드 확인", type="secondary")
        submitted = st.form_submit_button("회귀 분석 실행")

    # ======================================================================
    # 폼 제출 이후 처리
    # ======================================================================
    if submitted:
        if sentiment_df is None:
            st.error("감성 CSV를 업로드해 주세요.")
        else:
            start_str = start_date.strftime("%Y-%m-%d") if start_date else None
            end_str = end_date.strftime("%Y-%m-%d") if end_date else None

            resample_freq = None
            if resample_option == "주간 (W)":
                resample_freq = "W"
            elif resample_option == "월간 (M)":
                resample_freq = "M"

            # 회귀 변수 구성
            if regression_mode == "단순 회귀":
                if not selected_feature_names:
                    st.error("단순 회귀 모드에서는 컬럼을 1개 선택해야 합니다.")
                    st.stop()
                selected_column = selected_feature_names[0]
                feature_columns = [selected_column]
                st.info(f"단순 회귀 컬럼: {selected_column}")
            else:
                feature_columns = selected_feature_names if selected_feature_names else None

            # 라그 설정
            lag_periods_effective: list[int] | None = None
            lag_target_columns_effective: list[str] | None = None
            if lag_enabled and (lag_periods_selected or lag_target_columns_selected):
                lag_periods_effective = sorted(
                    {int(p) for p in lag_periods_selected if int(p) > 0}
                )
                lag_target_columns_effective = sorted(set(lag_target_columns_selected))

            # 사용자가 직접 라그 컬럼을 선택한 경우, 거기서도 시차 정보 추론
            selected_columns_iterable = feature_columns if feature_columns is not None else []
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
                lag_target_columns_effective = sorted(set(lag_target_columns_effective))
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
                    f"시차 변수 적용: {lag_target_display}에 대해 {lag_desc} 기간의 시차를 추가합니다."
                )

            # -------------------- 타겟(회사) 설정 --------------------
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

            # ==================================================================
            # 회사별 회귀 실행
            # ==================================================================
            for label, selector in targets:
                try:
                    with st.spinner(f"{label} 시장 데이터 수집 중..."):
                        market_key = selector.get("company_key")
                        market_entries = (
                            market_name_inputs.get(market_key, "") if market_key else ""
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

                st.markdown(f"## {label}")

                last_result = None
                last_scenario_name = None

                for scenario_name, result in scenario_results:
                    last_result = result
                    last_scenario_name = scenario_name

                    st.markdown(f"### {scenario_name}")
                    st.write(f"표본 수: {result.n_samples}")
                    safe_name = (
                        scenario_name.replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                    )

                    with st.expander("회귀 요약 보기", expanded=True):
                        st.markdown(f"```\n{result.summary}\n```")
                        st.download_button(
                            label=f"{scenario_name} 회귀 요약 TXT 다운로드",
                            data=result.summary,
                            file_name=f"summary_{label}_{safe_name}.txt",
                            mime="text/plain",
                        )

                # 마지막 시나리오 기준으로 계수/진단표 보여주기
                if last_result is not None and last_result.diagnostics is not None:
                    result = last_result
                    scenario_name = last_scenario_name or "scenario"
                    safe_name = (
                        scenario_name.replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                    )

                    st.markdown("#### 회귀 요약 지표")
                    st.dataframe(result.diagnostics.round(6), use_container_width=True)
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
                            result.coefficients.round(6), use_container_width=True
                        )
                        st.download_button(
                            label=f"{scenario_name} 계수 CSV 다운로드",
                            data=result.coefficients.to_csv(
                                index=False, encoding="utf-8-sig"
                            ),
                            file_name=f"coefficients_{label}_{safe_name}.csv",
                            mime="text/csv",
                        )

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
                            sentiment_effects = sentiment_effects.sort_values("p_value")

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

                with st.expander(f"{label} 시장 데이터 미리보기"):
                    st.dataframe(
                        market_df.tail().sort_values("date"), use_container_width=True
                    )

                st.download_button(
                    label=f"{label} 시세·환율 CSV 다운로드",
                    data=market_df.to_csv(index=False, encoding="utf-8-sig"),
                    file_name=f"market_{label}.csv",
                    mime="text/csv",
                )

    # =================== 종목 코드 조회 버튼 ===================
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
                        st.error(f"{company_key}: '{name}' 조회 실패 - {exc}")


if __name__ == "__main__":
    main()
