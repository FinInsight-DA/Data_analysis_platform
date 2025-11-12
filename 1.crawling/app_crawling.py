import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os, sys
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))  # 루트/.env 로드

from crawling import InsightPageAPI

# 페이지 설정
st.set_page_config(
    page_title="크롤링 페이지",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">크롤링 페이지</div>', unsafe_allow_html=True)
    
    # 탭 생성
    tab1, tab2 = st.tabs(["크롤링API", "학습데이터"])
    
    with tab1:
        st.markdown("### 크롤링API (설정세션)")
        
        # API 설정
        api_key = st.text_input(
            "API 키",
            value=os.getenv("INSIGHT_API_KEY", ""),  # ← .env 값
            type="password",
            key="api_key_tab1",
        )
        
        # 검색 설정
        company_name = st.text_input(
            "분석 대상 기업",
            placeholder="예: 삼성전자, 삼성"
        )
        
        # 크롤링 설정
        col_setting1, col_setting2 = st.columns(2)
        
        with col_setting1:
            page_num = st.number_input(
                "페이지 수",
                min_value=1,
                max_value=50,
                value=1,
                help="크롤링할 페이지 수 (1페이지 = 지정한 개수만큼 문서)"
            )
        
        with col_setting2:
            crawl_size = st.number_input(
                "페이지당 문서 수",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="한 페이지당 가져올 문서 수 (최대 10,000개)"
            )
        
        st.info(f"📊 총 최대 수집 문서 수: **{page_num * crawl_size:,}개**")
        
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
                st.error("❌ 분석 대상 기업을 입력하세요.")
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
                    
                    status_text.text("🔍 크롤링 시작...")
                    progress_bar.progress(10)
                    
                    with log_container:
                        st.text(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 크롤링 시작")
                        st.text(f"  - 키워드: {main_keyword}")
                        st.text(f"  - 동의어: {', '.join(synonyms) if synonyms else '없음'}")
                        st.text(f"  - 기간: {start_date} ~ {end_date}")
                        st.text(f"  - 페이지 수: {page_num}")
                        st.text(f"  - 페이지당 문서 수: {crawl_size:,}")
                    
                    progress_bar.progress(30)
                    
                    # API 클라이언트 생성
                    api = InsightPageAPI(token=api_key)
                    
                    # 크롤링 실행
                    all_documents = []
                    
                    for page in range(page_num):
                        page_progress = 30 + (page / page_num * 60)
                        progress_bar.progress(int(page_progress))
                        
                        with log_container:
                            st.text(f"[{datetime.now().strftime('%H:%M:%S')}] 페이지 {page + 1}/{page_num} 크롤링 중...")
                        
                        result = api.get_documents(
                            start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d"),
                            keyword=main_keyword,
                            synonyms=synonyms,
                            size=crawl_size,
                            from_index=crawl_size * page + 1
                        )
                        
                        documents = result.get('documents', [])
                        
                        if not documents:
                            with log_container:
                                st.text(f"[{datetime.now().strftime('%H:%M:%S')}] 페이지 {page + 1}: 데이터 없음 - 크롤링 종료")
                            break
                        
                        all_documents.extend(documents)
                        
                        with log_container:
                            st.text(f"[{datetime.now().strftime('%H:%M:%S')}] 페이지 {page + 1}: {len(documents):,}개 문서 수집")
                    
                    progress_bar.progress(100)
                    status_text.text("✅ 크롤링 완료!")
                    
                    with log_container:
                        st.text(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 크롤링 완료")
                        st.text(f"  - 총 문서 수: {len(all_documents):,}개")
                    
                    if len(all_documents) > 0:
                        st.success(f"✅ 크롤링 완료! {len(all_documents):,}개 문서를 수집했습니다.")
                        
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
    
    with tab2:
        st.markdown("### 🧹 학습 데이터 준비 및 검토")
        
        # 1. 데이터 로드 확인 및 업로드 기능
        if 'crawled_data' not in st.session_state or st.session_state['crawled_data'] is None:
            st.warning("⚠️ 먼저 '크롤링API' 탭에서 데이터를 수집하거나, CSV 파일을 업로드하여 데이터를 로드하세요.")
            
            # 파일 업로드 옵션
            uploaded_file = st.file_uploader("로컬에서 전처리된 학습 데이터 CSV 업로드", type=['csv'], key='train_upload')
            
            if uploaded_file is not None:
                # 업로드된 데이터를 임시로 session_state에 저장하여 사용
                try:
                    df_loaded = pd.read_csv(uploaded_file)
                    st.session_state['processed_data'] = df_loaded
                    st.success(f"✅ 파일 로드 완료. 총 {len(df_loaded):,}개 문서.")
                except Exception as e:
                    st.error(f"파일 로드 중 오류 발생: {e}")
            
            if st.session_state.get('processed_data') is None and st.session_state.get('crawled_data') is None:
                return # 데이터가 없으면 탭 진행 중단
        
        # 크롤링된 데이터 또는 업로드된 데이터 사용
        df = st.session_state.get('processed_data') if 'processed_data' in st.session_state else st.session_state.get('crawled_data')
        
        if df is None:
            return

        # 2. KPI Metrics (현재 데이터 상태)
        total_rows = len(df)
        # 'sentiment' 컬럼 존재 여부로 라벨링 완료 상태 추정 (1_감성라벨부착.ipynb 결과)
        has_sentiment = 'sentiment' in df.columns
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        with col_kpi1:
            st.metric("총 데이터 행 수", f"{total_rows:,} 개")
        with col_kpi2:
            st.metric("감성 라벨 존재 여부", "✅ 있음" if has_sentiment else "❌ 없음")
        with col_kpi3:
            st.metric("다음 단계 준비 상태", "✅ 학습 준비 완료" if has_sentiment else "⚠️ 라벨링 단계 필요")

        st.markdown("---")
        
        # 3. 데이터 클리닝/전처리 설정 (1_감성라벨부착.ipynb 및 전처리 단계 반영)
        st.markdown("### ⚙️ 데이터 클리닝 및 전처리 설정")
        with st.expander("전처리 옵션 설정 (실제 적용 로직은 백엔드에서 구현 필요)", expanded=False):
            
            st.subheader("1. 중복/노이즈 제거")
            col_clean1, col_clean2 = st.columns(2)
            with col_clean1:
                dedup_option = st.checkbox("문서 중복 제거", value=True, help="제목/본문이 완전히 동일한 문서를 제거합니다.")
                short_filter = st.slider("최소 길이 필터 (단어)", min_value=5, max_value=50, value=10, help="이 길이 미만의 문장을 제거합니다.", key='min_len_filter')
            with col_clean2:
                # 불용어 처리 설정
                st.text_area("추가 불용어 목록", value="기자, 관련, 이날, 현재, 것으로", height=100, key='stopwords_list')
                
            st.subheader("2. 텍스트 정규화")
            normalize_text = st.checkbox("문자 정규화 (이모지, 특수기호)", value=True, key='normalize_check')

        st.markdown("---")
        
        # 4. 데이터 미리보기
        st.markdown("### 📋 데이터 미리보기")
        st.dataframe(df.head(10), use_container_width=True)
        
        # 5. 최종 작업 버튼
        st.markdown("---")
        
        # 데이터프레임을 CSV로 변환 (다운로드를 위해)
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_data_{timestamp}.csv"
        
        st.download_button(
            label="💾 전처리된 학습 데이터 다운로드 (CSV)", 
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
            type='secondary'
        )
        st.caption("이 파일을 다운로드하여 `1_감성라벨부착.ipynb` 등의 학습 단계에 사용하세요.")


if __name__ == "__main__":
    main()
