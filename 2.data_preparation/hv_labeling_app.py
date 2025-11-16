# -*- coding: utf-8 -*-
"""
HBM 프로젝트 - H/V 라벨링 자동화 Streamlit 앱 (로컬 환경용)
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple
from io import BytesIO
from datetime import datetime
import os

# ============================================================================
# 페이지 설정
# ============================================================================
# st.set_page_config(
#     page_title="H/V 라벨링 자동화",
#     page_icon="🏷️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 상수 정의
# ============================================================================
LABEL_DESCRIPTIONS = {
    'H': '수평적 통합 (Horizontal)',
    'V': '수직적 통합 (Vertical)'
}

LABEL_TYPE_MAPPING = {
    'H': 'horizontal',
    'V': 'vertical'
}

# ============================================================================
# 함수 정의
# ============================================================================

@st.cache_data
def load_term_db_from_json(file_content):
    """JSON 형식의 TERM_DB 로드"""
    try:
        return json.loads(file_content)
    except Exception as e:
        st.error(f"TERM_DB 로드 실패: {e}")
        return {}

def detect_label_in_text(text: str, TERM_DB: dict, label_priority: list,
                         default_label: str, min_matches: int = 1,
                         case_sensitive: bool = False) -> Tuple[str, str, str, int]:
    """라벨 자동 감지"""
    if pd.isna(text) or not text:
        return (default_label, '공통', 'Unknown', 0)

    text = str(text).strip()
    if not case_sensitive:
        text = text.lower()
    
    label_matches = {label: [] for label in TERM_DB.keys()}

    for label_type, categories in TERM_DB.items():
        for category, terms in categories.items():
            for term in terms:
                search_term = term if case_sensitive else term.lower()
                if search_term in text:
                    label_matches[label_type].append((label_type, category, term))

    for priority_label in label_priority:
        if priority_label in label_matches:
            match_count = len(label_matches[priority_label])
            if match_count >= min_matches:
                return (
                    label_matches[priority_label][0][0],
                    label_matches[priority_label][0][1],
                    label_matches[priority_label][0][2],
                    match_count
                )

    return (default_label, '공통', 'Unknown', 0)

def process_labeling(df, TERM_DB, config):
    """라벨링 처리"""
    # sentence 생성
    df['sentence'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    
    # 라벨링 실행
    results = df['sentence'].apply(
        lambda x: detect_label_in_text(
            x, TERM_DB,
            config['label_priority'],
            config['default_label'],
            config['min_matches'],
            config['case_sensitive']
        )
    )
    
    df['label'] = results.apply(lambda x: x[0])
    df['aspect_category'] = results.apply(lambda x: x[1])
    df['aspect_term'] = results.apply(lambda x: x[2])
    df['match_count'] = results.apply(lambda x: x[3])
    df['HV_type'] = df['label'].map(LABEL_TYPE_MAPPING)
    
    return df

def create_distribution_chart(df):
    """라벨 분포 차트 생성"""
    label_counts = df['label'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=[LABEL_DESCRIPTIONS.get(label, label) for label in label_counts.index],
            y=label_counts.values,
            text=label_counts.values,
            textposition='auto',
            marker_color=['#ff7f0e', '#1f77b4']
        )
    ])
    
    fig.update_layout(
        title='라벨 분포',
        xaxis_title='라벨 타입',
        yaxis_title='문서 수',
        height=400
    )
    
    return fig

def create_company_distribution(df):
    """회사별 분포 차트"""
    if 'company' not in df.columns:
        return None
    
    company_dist = pd.crosstab(df['company'], df['label'])
    
    fig = go.Figure(data=[
        go.Bar(name=LABEL_DESCRIPTIONS.get(label, label),
               x=company_dist.index,
               y=company_dist[label],
               text=company_dist[label],
               textposition='auto')
        for label in company_dist.columns
    ])
    
    fig.update_layout(
        title='회사별 라벨 분포',
        xaxis_title='회사',
        yaxis_title='문서 수',
        barmode='group',
        height=400
    )
    
    return fig

def create_category_distribution(df):
    """카테고리별 분포 차트"""
    category_counts = df['aspect_category'].value_counts().head(10)
    
    fig = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        labels={'x': '카테고리', 'y': '문서 수'},
        title='Top 10 Aspect 카테고리',
        text_auto=True
    )
    
    fig.update_layout(height=400)
    
    return fig

# ============================================================================
# 메인 앱
# ============================================================================

def main():
    # 헤더
    st.markdown('<div class="main-header">🏷️ H/V 라벨링 자동화</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # ============================================================================
    # 1. 파일 업로드 섹션
    # ============================================================================
    st.markdown('<div class="sub-header">📁 1. 파일 업로드</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_csv = st.file_uploader(
            "데이터 CSV 파일 (title, content 컬럼 필수)",
            type=['csv'],
            key='csv_uploader'
        )
    
    with col2:
        uploaded_term_db = st.file_uploader(
            "Term DB JSON 파일",
            type=['json'],
            key='json_uploader'
        )
    
    # JSON 편집기
    if uploaded_term_db is not None:
        with st.expander("📝 JSON 파일 수정 및 저장"):
            term_db_content = uploaded_term_db.read().decode('utf-8')
            uploaded_term_db.seek(0)  # 파일 포인터 리셋
            
            edited_json = st.text_area(
                "JSON 내용 편집",
                value=term_db_content,
                height=300,
                key='json_editor'
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("💾 수정된 JSON 저장", use_container_width=True):
                    try:
                        # JSON 유효성 검사
                        json.loads(edited_json)
                        st.download_button(
                            label="📥 수정된 JSON 다운로드",
                            data=edited_json,
                            file_name=f"term_db_edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    except json.JSONDecodeError as e:
                        st.error(f"❌ JSON 형식 오류: {e}")
    
    if uploaded_csv is None or uploaded_term_db is None:
        st.info("⬆️ CSV 파일과 Term DB JSON 파일을 업로드해주세요.")
        return
    
    # 데이터 로드
    try:
        df = pd.read_csv(uploaded_csv)
        st.success(f"✅ 데이터 로드 완료: {len(df):,}개 문서")
    except Exception as e:
        st.error(f"❌ CSV 파일 로드 실패: {e}")
        return
    
    try:
        # JSON 편집기에서 수정된 내용 사용
        if 'json_editor' in st.session_state and st.session_state.json_editor:
            term_db_content = st.session_state.json_editor
        else:
            term_db_content = uploaded_term_db.read().decode('utf-8')
            uploaded_term_db.seek(0)
        
        TERM_DB = load_term_db_from_json(term_db_content)
        st.success(f"✅ Term DB 로드 완료: {len(TERM_DB)}개 라벨")
    except Exception as e:
        st.error(f"❌ Term DB 파일 로드 실패: {e}")
        return
    
    st.markdown("---")
    
    # ============================================================================
    # 2. 파라미터 설정 (분석가용)
    # ============================================================================
    st.markdown('<div class="sub-header">⚙️ 2. 파라미터 설정</div>', unsafe_allow_html=True)
    
    # 기본값 자동 설정
    auto_label_priority = ['V', 'H'] if 'V' in TERM_DB and 'H' in TERM_DB else list(TERM_DB.keys())
    auto_default_label = auto_label_priority[-1] if auto_label_priority else 'H'
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_matches = st.slider(
            "최소 매칭 키워드 수",
            min_value=1,
            max_value=5,
            value=1,
            help="문장에 최소 몇 개의 키워드가 매칭되어야 라벨을 부여할지 설정"
        )
    
    with col2:
        label_priority_option = st.selectbox(
            "라벨 우선순위",
            options=['V 우선 (V→H)', 'H 우선 (H→V)'],
            index=0 if auto_label_priority[0] == 'V' else 1,
            help="V와 H 둘 다 매칭될 때 우선 적용할 라벨"
        )
        label_priority = ['V', 'H'] if 'V 우선' in label_priority_option else ['H', 'V']
    
    with col3:
        default_label = st.radio(
            "기본 라벨",
            options=['H', 'V'],
            index=0 if auto_default_label == 'H' else 1,
            horizontal=True,
            help="키워드가 매칭되지 않을 때 적용할 기본 라벨"
        )
    
    with col4:
        case_sensitive = st.checkbox(
            "대소문자 구분",
            value=False,
            help="키워드 매칭 시 대소문자를 구분할지 여부"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        remove_unknown = st.checkbox(
            "Unknown 제외",
            value=True,
            help="키워드가 매칭되지 않은 문서를 결과에서 제외"
        )
    
    with col2:
        min_match_filter = st.slider(
            "최소 match_count 필터",
            min_value=0,
            max_value=10,
            value=1,
            help="결과에 포함할 최소 매칭 수 (라벨링 후 필터링)"
        )
    
    # 현재 설정 요약
    with st.expander("📋 현재 설정 요약"):
        st.write(f"""
        - **라벨링 조건**: 최소 {min_matches}개 키워드 매칭 시 라벨 부여
        - **우선순위**: {' → '.join(label_priority)}
        - **기본 라벨**: {default_label} (매칭 실패 시)
        - **대소문자**: {'구분함' if case_sensitive else '구분 안 함'}
        - **Unknown 제외**: {'예' if remove_unknown else '아니오'}
        - **결과 필터**: match_count >= {min_match_filter}
        """)
    
    config = {
        'min_matches': min_matches,
        'label_priority': label_priority,
        'default_label': default_label,
        'case_sensitive': case_sensitive,
        'remove_unknown': remove_unknown,
        'min_match_filter': min_match_filter
    }
    
    st.markdown("---")
    
    # 라벨링 실행 버튼
    if st.button("🚀 라벨링 실행", type="primary", use_container_width=True):
        with st.spinner("라벨링 진행 중..."):
            # 라벨링 처리
            df_labeled = process_labeling(df.copy(), TERM_DB, config)
            
            # 필터링 적용
            df_original_len = len(df_labeled)
            
            if config['remove_unknown']:
                df_labeled = df_labeled[df_labeled['aspect_term'] != 'Unknown'].copy()
            
            df_labeled = df_labeled[df_labeled['match_count'] >= config['min_match_filter']].copy()
            
            # 세션 상태에 저장
            st.session_state['df_labeled'] = df_labeled
            st.session_state['df_original_len'] = df_original_len
            st.session_state['config'] = config
            
            st.success("✅ 라벨링 완료!")
    
    # 결과 표시
    if 'df_labeled' in st.session_state:
        df_labeled = st.session_state['df_labeled']
        df_original_len = st.session_state['df_original_len']
        
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 3. 라벨링 결과</div>', unsafe_allow_html=True)
        
        # 주요 통계
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("전체 문서", f"{df_original_len:,}")
        
        with col2:
            st.metric("처리된 문서", f"{len(df_labeled):,}")
        
        with col3:
            h_count = (df_labeled['label'] == 'H').sum()
            st.metric("H (수평적)", f"{h_count:,} ({h_count/len(df_labeled)*100:.1f}%)")
        
        with col4:
            v_count = (df_labeled['label'] == 'V').sum()
            st.metric("V (수직적)", f"{v_count:,} ({v_count/len(df_labeled)*100:.1f}%)")
        
        # 차트 표시
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_distribution_chart(df_labeled)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            if 'company' in df_labeled.columns:
                fig2 = create_company_distribution(df_labeled)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
        
        fig3 = create_category_distribution(df_labeled)
        st.plotly_chart(fig3, use_container_width=True)
        
        # 데이터 미리보기
        st.markdown("---")
        
        with st.expander("🔍 데이터 미리보기 (처음 100개)", expanded=False):
            display_cols = ['title', 'company', 'label', 'HV_type', 'aspect_category',
                          'aspect_term', 'match_count']
            display_cols = [col for col in display_cols if col in df_labeled.columns]
            st.dataframe(df_labeled[display_cols].head(100), use_container_width=True)
        
        # 다운로드
        st.markdown("---")
        st.markdown('<div class="sub-header">💾 4. 결과 다운로드</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_utf8sig = df_labeled.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV 다운로드 (UTF-8-SIG, Excel용)",
                data=csv_utf8sig,
                file_name=f"hv_labeled_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            csv_utf8 = df_labeled.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="📥 CSV 다운로드 (UTF-8)",
                data=csv_utf8,
                file_name=f"hv_labeled_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_labeled.to_excel(writer, index=False, sheet_name='라벨링결과')
            
            st.download_button(
                label="📥 Excel 다운로드",
                data=buffer.getvalue(),
                file_name=f"hv_labeled_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
