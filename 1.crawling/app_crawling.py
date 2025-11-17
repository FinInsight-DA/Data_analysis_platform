# -*- coding: utf-8 -*-
"""
HBM 프로젝트 - 크롤링 페이지 Streamlit 앱 (성능 최적화 버전)
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os, sys
import time as time_module
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from crawling import InsightPageAPI

# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(
    page_title="데이터 수집",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS 스타일
# ============================================================================
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    /* 탭 색상 변경 */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom-color: #1f77b4 !important;
        color: #1f77b4 !important;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #1f77b4 !important;
    }
    /* 슬라이더 색상 변경 */
    .stSlider > div > div > div > div {
        background-color: #1f77b4 !important;
    }
    input[type="range"]::-webkit-slider-thumb {
        background-color: #1f77b4 !important;
    }
    input[type="range"]::-moz-range-thumb {
        background-color: #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 메인 앱
# ============================================================================

def main():
    st.markdown('<div class="main-header">데이터 수집</div>', unsafe_allow_html=True)
    
    # 탭 생성
    tab1, tab2 = st.tabs(["키워드", " "])
    
    with tab1:
        st.markdown('<div class="sub-header">📊 키워드 크롤링</div>', unsafe_allow_html=True)
        
        # API 설정
        api_key = st.text_input(
            "API 키",
            value=os.getenv("INSIGHT_API_KEY", ""),
            type="password",
            key="api_key_tab1",
        )
        
        # 검색 설정
        company_name = st.text_input(
            "수집 키워드",
            placeholder="예: 삼성전자, 하이닉스, 반도체"
        )
        
        # 크롤링 설정
        col_setting1, col_setting2, col_setting3 = st.columns(3)
        
        with col_setting1:
            page_num = st.number_input(
                "페이지 수",
                min_value=1,
                max_value=100,
                value=1,
                help="크롤링할 페이지 수 (페이지당 최대 10,000개)"
            )
        
        with col_setting2:
            crawl_size = st.number_input(
                "페이지당 문서 수",
                min_value=100,
                max_value=10000,
                value=10000,
                step=100,
                help="한 페이지당 가져올 문서 수 (최대 10,000개)"
            )
        
        with col_setting3:
            enable_checkpoint = st.checkbox(
                "중간 저장",
                value=True,
                help="페이지마다 중간 결과를 저장 (중단 시 재개 가능)"
            )
        
        # ⭐ 성능 최적화: 페이지 간 대기 시간 설정 추가
        col_delay1, col_delay2 = st.columns(2)
        
        with col_delay1:
            enable_delay = st.checkbox(
                "페이지 간 대기",
                value=True,
                help="Rate Limiting 방지를 위해 페이지 사이에 대기 (권장)"
            )
        
        with col_delay2:
            if enable_delay:
                delay_seconds = st.number_input(
                    "대기 시간 (초)",
                    min_value=1,
                    max_value=30,
                    value=5,
                    help="페이지 사이 대기 시간 (권장: 3~5초)"
                )
            else:
                delay_seconds = 0
        
        # 예상 정보 표시
        st.info(f"💡 총 문서 수: **{page_num * crawl_size:,}개**")
        
        st.markdown("---")
        
        # 날짜 선택
        col_date1, col_date2 = st.columns(2)
        
        default_start = datetime.now() - timedelta(days=365)
        default_end = datetime.now()
        
        with col_date1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                help="검색 시작 날짜"
            )
        
        with col_date2:
            end_date = st.date_input(
                "End Date",
                value=default_end,
                help="검색 종료 날짜"
            )
        
        # 크롤링 버튼
        if st.button("크롤링 버튼", use_container_width=True):
            if not api_key:
                st.error("❌ API 키를 입력하세요.")
            elif not company_name:
                st.error("❌ 키워드를 입력하세요.")
            elif start_date > end_date:
                st.error("❌ Start Date가 End Date보다 나중입니다. 날짜를 다시 확인하세요.")
            else:
                # 크롤링 시작
                st.markdown("### 크롤링 로그 및 상태 바")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_container = st.container()
                
                try:
                    # 키워드 및 동의어 설정
                    keywords = [k.strip() for k in company_name.split(',')]
                    main_keyword = keywords[0]
                    synonyms = keywords if len(keywords) > 1 else []
                    
                    # 중간 저장 파일명
                    checkpoint_file = f"crawl_checkpoint_{main_keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    status_text.text("🔍 크롤링 시작...")
                    progress_bar.progress(10)
                    
                    with log_container:
                        st.text(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 크롤링 시작")
                        st.text(f"  - 키워드: {main_keyword}")
                        st.text(f"  - 동의어: {', '.join(synonyms) if synonyms else '없음'}")
                        st.text(f"  - 기간: {start_date} ~ {end_date}")
                        st.text(f"  - 페이지 수: {page_num}")
                        st.text(f"  - 페이지당 문서 수: {crawl_size:,}")
                        if enable_checkpoint:
                            st.text(f"  - 중간 저장: 활성화 ({checkpoint_file})")
                        if enable_delay:
                            st.text(f"  - 페이지 간 대기: {delay_seconds}초 (Rate Limiting 방지)")
                    
                    progress_bar.progress(30)
                    
                    # API 클라이언트 생성
                    api = InsightPageAPI(token=api_key)
                    
                    # 크롤링 실행
                    all_documents = []
                    start_time = time_module.time()
                    
                    # ⭐ 성능 최적화: 재시도 설정
                    max_retries = 3  # 최대 재시도 횟수
                    
                    for page in range(page_num):
                        page_start = time_module.time()
                        
                        page_progress = 30 + (page / page_num * 60)
                        progress_bar.progress(int(page_progress))
                        
                        # 예상 남은 시간 계산
                        if page > 0:
                            elapsed = time_module.time() - start_time
                            avg_time_per_page = elapsed / page
                            remaining_pages = page_num - page
                            eta_seconds = avg_time_per_page * remaining_pages
                            eta_str = f"{int(eta_seconds // 60)}분 {int(eta_seconds % 60)}초"
                        else:
                            eta_str = "계산 중..."
                        
                        status_text.text(f"📥 페이지 {page + 1}/{page_num} 수집 중... (예상 남은 시간: {eta_str})")
                        
                        with log_container:
                            st.text(f"[{datetime.now().strftime('%H:%M:%S')}] 페이지 {page + 1}/{page_num} 크롤링 중...")
                        
                        # ⭐ 성능 최적화: 재시도 로직
                        documents = None
                        api_elapsed = 0
                        
                        for attempt in range(max_retries):
                            try:
                                api_start = time_module.time()
                                
                                result = api.get_documents(
                                    start_date=start_date.strftime("%Y-%m-%d"),
                                    end_date=end_date.strftime("%Y-%m-%d"),
                                    keyword=main_keyword,
                                    synonyms=synonyms,
                                    size=crawl_size,
                                    from_index=crawl_size * page + 1
                                )
                                
                                api_elapsed = time_module.time() - api_start
                                documents = result.get('documents', [])
                                
                                # 성공하면 재시도 루프 종료
                                break
                                
                            except Exception as e:
                                if attempt < max_retries - 1:
                                    wait_time = 2 ** attempt  # 지수 백오프: 1초, 2초, 4초
                                    with log_container:
                                        st.text(f"  ⚠️ API 오류 (재시도 {attempt + 1}/{max_retries}): {str(e)}")
                                        st.text(f"  ⏰ {wait_time}초 후 재시도...")
                                    time_module.sleep(wait_time)
                                else:
                                    # 최대 재시도 횟수 초과
                                    with log_container:
                                        st.text(f"  ❌ 최대 재시도 초과: {str(e)}")
                                    raise e
                        
                        if not documents:
                            with log_container:
                                st.text(f"[{datetime.now().strftime('%H:%M:%S')}] 페이지 {page + 1}: 데이터 없음 - 크롤링 종료")
                            break
                        
                        all_documents.extend(documents)
                        
                        # 페이지 처리 시간 계산
                        page_elapsed = time_module.time() - page_start
                        
                        # ⭐ 성능 진단: API 응답 시간 체크
                        if api_elapsed > 30:
                            with log_container:
                                st.text(f"  ⚠️ 경고: API 응답이 느립니다 ({api_elapsed:.1f}초)")
                                st.text(f"  💡 Rate Limiting 가능성 - 대기 시간을 늘리거나 크기를 줄이세요")
                        
                        with log_container:
                            st.text(f"[{datetime.now().strftime('%H:%M:%S')}] 페이지 {page + 1}: {len(documents):,}개 수집 완료 ({page_elapsed:.1f}초)")
                        
                        # 중간 저장
                        if enable_checkpoint and documents:
                            df_temp = pd.DataFrame(documents)
                            if page == 0:
                                df_temp.to_csv(checkpoint_file, index=False, encoding='utf-8-sig')
                            else:
                                df_temp.to_csv(checkpoint_file, mode='a', header=False, index=False, encoding='utf-8-sig')
                            
                            with log_container:
                                st.text(f"[{datetime.now().strftime('%H:%M:%S')}]   → 중간 저장 완료 (누적: {len(all_documents):,}개)")
                        
                        # ⭐ 성능 최적화: 페이지 간 대기 (Rate Limiting 방지)
                        if enable_delay and page < page_num - 1:  # 마지막 페이지가 아니면
                            with log_container:
                                st.text(f"[{datetime.now().strftime('%H:%M:%S')}]   ⏰ {delay_seconds}초 대기 중... (Rate Limiting 방지)")
                            time_module.sleep(delay_seconds)
                    
                    total_elapsed = time_module.time() - start_time
                    progress_bar.progress(100)
                    status_text.text("✅ 크롤링 완료!")
                    
                    with log_container:
                        st.text(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 크롤링 완료")
                        st.text(f"  - 총 문서 수: {len(all_documents):,}개")
                        st.text(f"  - 소요 시간: {int(total_elapsed // 60)}분 {int(total_elapsed % 60)}초")
                        st.text(f"  - 평균 속도: {len(all_documents) / total_elapsed:.0f}개/초")
                    
                    if len(all_documents) > 0:
                        st.markdown(f"""
                        <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                            ✅ <strong>크롤링 완료!</strong><br>
                            • 수집 문서: {len(all_documents):,}개<br>
                            • 소요 시간: {int(total_elapsed // 60)}분 {int(total_elapsed % 60)}초<br>
                            • 평균 속도: {len(all_documents) / total_elapsed:.0f}개/초
                            {f'<br>• 중간 저장: {checkpoint_file}' if enable_checkpoint else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 데이터프레임 생성
                        df = pd.DataFrame(all_documents)
                        
                        # 세션 스테이트에 저장
                        st.session_state['crawled_data'] = df
                        st.session_state['crawled_keyword'] = main_keyword
                        st.session_state['crawled_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # 데이터 정보
                        st.markdown("---")
                        st.markdown("### 📋 크롤링 결과")
                        
                        col_info1, col_info2, col_info3 = st.columns(3)
                        with col_info1:
                            st.metric("총 문서 수", f"{len(df):,}개")
                        with col_info2:
                            st.metric("컬럼 수", f"{len(df.columns)}개")
                        with col_info3:
                            st.metric("키워드", main_keyword)
                        
                        # 컬럼 정보
                        with st.expander("📊 데이터 컬럼 정보"):
                            cols = list(df.columns)
                            st.write(", ".join(cols))
                        
                        # 전체 데이터 미리보기
                        st.markdown("### 데이터 미리보기")
                        
                        # 표시할 행 수 선택
                        display_rows = st.slider(
                            "표시할 행 수",
                            min_value=10,
                            max_value=min(100, len(df)),
                            value=min(20, len(df)),
                            step=10
                        )
                        
                        st.dataframe(df.head(display_rows), use_container_width=True, height=400)
                        
                        # 다운로드 버튼
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{main_keyword}_{timestamp}.csv"
                        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
                        
                        st.download_button(
                            label="📥 CSV 파일 다운로드",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                    else:
                        st.warning("⚠️ 검색 결과가 없습니다.")
                        st.info("""
                        **검색 결과가 없는 이유:**
                        - 키워드가 뉴스에 없을 수 있습니다
                        - 날짜 범위에 해당 키워드 뉴스가 없을 수 있습니다
                        - 날짜 범위가 너무 짧을 수 있습니다
                        
                        **해결 방법:**
                        - 키워드를 다시 확인하세요
                        - 날짜 범위를 넓혀보세요 (예: 1년)
                        - 동의어를 추가해보세요 (예: `삼성전자, 삼성, Samsung`)
                        - 다른 키워드로 시도해보세요 (예: `반도체`, `AI`, `스타트업`)
                        """)
                    
                except Exception as e:
                    progress_bar.progress(0)
                    status_text.text("❌ 크롤링 실패")
                    st.error(f"❌ 오류 발생: {str(e)}")
                    with log_container:
                        st.text(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 오류 발생")
                        st.text(f"  - 에러: {str(e)}")
        
        # 이전 크롤링 결과가 있으면 표시
        if 'crawled_data' in st.session_state and st.session_state.get('crawled_data') is not None:
            st.markdown("---")
            st.markdown("### 💾 저장된 크롤링 데이터")
            
            df_saved = st.session_state['crawled_data']
            keyword_saved = st.session_state.get('crawled_keyword', '데이터')
            time_saved = st.session_state.get('crawled_time', '알 수 없음')
            
            col_saved1, col_saved2, col_saved3 = st.columns(3)
            with col_saved1:
                st.metric("저장된 문서", f"{len(df_saved):,}개")
            with col_saved2:
                st.metric("키워드", keyword_saved)
            with col_saved3:
                st.metric("크롤링 시간", time_saved)
            
            # 다시 다운로드 버튼
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{keyword_saved}_{timestamp}.csv"
            csv_data = df_saved.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📥 저장된 데이터 다운로드",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key="download_saved"
            )

if __name__ == "__main__":
    main()