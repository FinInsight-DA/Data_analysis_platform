# -*- coding: utf-8 -*-
"""
HBM 프로젝트 - 데이터 라벨링 자동화 Streamlit 앱 (로컬 환경용)
"""

import streamlit as st
import pandas as pd
import numpy as np
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
st.set_page_config(
    page_title="데이터 라벨링",
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

def check_company_mentions(sent: str, company_config: dict) -> dict:
    """
    문장에서 설정된 회사명 언급 여부 확인
    
    Args:
        sent: 입력 문장 문자열
        company_config: 회사별 키워드 딕셔너리
            예: {
                'Samsung Electronics': ['삼성전자', '삼성', 'samsung'],
                'SK Hynix': ['하이닉스', 'sk하이닉스', 'sk hynix']
            }
        
    Returns:
        회사명을 키로 하는 불리언 딕셔너리
        예: {'Samsung Electronics': True, 'SK Hynix': False, ...}
    """
    if pd.isna(sent) or not sent:
        return {company: False for company in company_config.keys()}
    
    sent_lower = str(sent).lower()
    result = {}
    
    for company_name, keywords in company_config.items():
        result[company_name] = any(keyword.lower() in sent_lower for keyword in keywords)
    
    return result

def process_labeling(df, TERM_DB, config, company_config: dict = None):
    """라벨링 처리"""
    # sentence 생성
    df['sentence'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    
    # company 컬럼 추가
    if company_config is None:
        # 기본 설정: 삼성전자, SK하이닉스
        company_config = {
            'Samsung Electronics': ['삼성전자', '삼성', 'samsung'],
            'SK Hynix': ['하이닉스', 'sk하이닉스', 'sk hynix']
        }
    
    # 회사 언급 확인
    company_checks = df['sentence'].apply(lambda x: check_company_mentions(x, company_config))
    
    # company 컬럼 생성 (두 회사 모두 언급된 경우 "both"로 설정)
    def determine_company(checks_dict):
        mentioned_companies = [company for company, is_mentioned in checks_dict.items() if is_mentioned]
        if len(mentioned_companies) == 0:
            return None
        elif len(mentioned_companies) == 1:
            return mentioned_companies[0]
        else:
            # 두 회사 이상 언급된 경우
            return "both"
    
    df['company'] = company_checks.apply(determine_company)
    
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
    
    # 파란 계열 그라데이션 (진한 파랑 → 연한 파랑)
    colors = ['#1a5490', '#2874b5', '#4a90c5', '#73a9d6']
    bar_colors = [colors[i % len(colors)] for i in range(len(label_counts))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[LABEL_DESCRIPTIONS.get(label, label) for label in label_counts.index],
            y=label_counts.values,
            text=label_counts.values,
            textposition='outside',
            textfont=dict(size=14, color='#2c3e50', family='Arial'),
            marker=dict(
                color=bar_colors,
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{x}</b><br>문서 수: %{y:,}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='라벨 분포',
            font=dict(size=18, color='#2c3e50', family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='',
            tickfont=dict(size=13, color='#2c3e50'),
            showgrid=False,
            showline=False
        ),
        yaxis=dict(
            title='문서 수',
            title_font=dict(size=13, color='#7f8c8d'),
            tickfont=dict(size=12, color='#7f8c8d'),
            showgrid=True,
            gridwidth=1,
            gridcolor='#ecf0f1',
            showline=False,
            range=[0, label_counts.values.max() * 1.15]
        ),
        height=480,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=100, b=60, l=80, r=40),
        showlegend=False
    )
    
    return fig

def create_company_distribution(df):
    """회사별 분포 차트"""
    if 'company' not in df.columns:
        return None
    
    company_dist = pd.crosstab(df['company'], df['label'])
    
    # 파란 계열 그라데이션
    colors = ['#1a5490', '#2874b5', '#4a90c5', '#73a9d6']
    
    fig = go.Figure(data=[
        go.Bar(
            name=LABEL_DESCRIPTIONS.get(label, label),
            x=company_dist.index,
            y=company_dist[label],
            text=company_dist[label],
            textposition='outside',
            textfont=dict(size=13, color='#2c3e50'),
            marker=dict(
                color=colors[i % len(colors)],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{x}</b><br>%{fullData.name}: %{y:,}<extra></extra>'
        )
        for i, label in enumerate(company_dist.columns)
    ])
    
    fig.update_layout(
        title=dict(
            text='회사별 라벨 분포',
            font=dict(size=18, color='#2c3e50', family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='',
            tickfont=dict(size=12, color='#2c3e50'),
            showgrid=False,
            showline=False
        ),
        yaxis=dict(
            title='문서 수',
            title_font=dict(size=13, color='#7f8c8d'),
            tickfont=dict(size=12, color='#7f8c8d'),
            showgrid=True,
            gridwidth=1,
            gridcolor='#ecf0f1',
            showline=False,
            range=[0, company_dist.max().max() * 1.15]
        ),
        barmode='group',
        height=480,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=100, b=100, l=80, r=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor='white',
            bordercolor='#ecf0f1',
            borderwidth=1,
            font=dict(size=12, color='#2c3e50')
        )
    )
    
    return fig

def create_category_distribution(df):
    """카테고리별 분포 차트"""
    category_counts = df['aspect_category'].value_counts().head(10)
    
    # 파란 계열 그라데이션 (진한 파랑 → 연한 파랑)
    n = len(category_counts)
    colors = []
    for i in range(n):
        # 진한 파랑(#1a5490)에서 연한 파랑(#b3d9ff)으로 그라데이션
        ratio = i / max(n - 1, 1)
        r = int(26 + (179 - 26) * ratio)
        g = int(84 + (217 - 84) * ratio)
        b = int(144 + (255 - 144) * ratio)
        colors.append(f'rgb({r},{g},{b})')
    
    fig = go.Figure(data=[
        go.Bar(
            x=category_counts.index,
            y=category_counts.values,
            text=category_counts.values,
            textposition='outside',
            textfont=dict(size=12, color='#2c3e50'),
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{x}</b><br>문서 수: %{y:,}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='Top 10 키워드 카테고리',
            font=dict(size=18, color='#2c3e50', family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='',
            tickfont=dict(size=11, color='#2c3e50'),
            tickangle=0,
            tickmode='array',
            tickvals=list(range(len(category_counts))),
            ticktext=list(category_counts.index),
            showgrid=False,
            showline=False
        ),
        yaxis=dict(
            title='문서 수',
            title_font=dict(size=13, color='#7f8c8d'),
            tickfont=dict(size=12, color='#7f8c8d'),
            showgrid=True,
            gridwidth=1,
            gridcolor='#ecf0f1',
            showline=False,
            range=[0, category_counts.values.max() * 1.15]
        ),
        height=520,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=100, b=120, l=80, r=40),
        showlegend=False
    )
    
    return fig

# ============================================================================
# 메인 앱
# ============================================================================

def main():
    # 헤더
    st.markdown('<div class="main-header">데이터 라벨링</div>', unsafe_allow_html=True)
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
    
    # ============================================================================
    # 파일 변경 감지 및 세션 상태 초기화 (추가된 부분)
    # ============================================================================
    current_csv_name = uploaded_csv.name if uploaded_csv else None
    current_json_name = uploaded_term_db.name if uploaded_term_db else None

    # 이전 파일명과 비교
    if 'prev_csv_name' not in st.session_state:
        st.session_state['prev_csv_name'] = None
    if 'prev_json_name' not in st.session_state:
        st.session_state['prev_json_name'] = None

    # 파일이 바뀌면 결과 초기화
    if (current_csv_name != st.session_state['prev_csv_name'] or
        current_json_name != st.session_state['prev_json_name']):
        
        # 세션 상태 초기화
        if 'df_labeled' in st.session_state:
            del st.session_state['df_labeled']
        if 'df_original_len' in st.session_state:
            del st.session_state['df_original_len']
        if 'config' in st.session_state:
            del st.session_state['config']
        
        # 현재 파일명 저장
        st.session_state['prev_csv_name'] = current_csv_name
        st.session_state['prev_json_name'] = current_json_name
    
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
                if st.button("💾 수정된 JSON 저장", use_container_width=True, key="save_term_db"):
                    try:
                        # JSON 유효성 검사
                        json.loads(edited_json)
                        st.download_button(
                            label="📥 수정된 JSON 다운로드",
                            data=edited_json,
                            file_name=f"term_db_edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True,
                            key="download_term_db"
                        )
                    except json.JSONDecodeError as e:
                        st.error(f"❌ JSON 형식 오류: {e}")
            
            with col_b:
                if st.button("🔄 원본으로 되돌리기", use_container_width=True, key="reset_term_db"):
                    st.session_state.json_editor = term_db_content
                    st.rerun()
    
    if uploaded_csv is None or uploaded_term_db is None:
        st.info("⬆️ CSV 파일과 Term DB JSON 파일을 업로드해주세요.")
        return
    
    # 파일 업로드 완료 메시지
    st.success("✅ 파일 업로드 완료! 데이터를 로드하는 중...")
    
    # 데이터 로드
    try:
        df = pd.read_csv(uploaded_csv)
        
        # 필수 컬럼 확인
        required_cols = ['title', 'content']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ 필수 컬럼이 없습니다: {', '.join(missing_cols)}")
            st.info("💡 CSV 파일에 'title'과 'content' 컬럼이 있는지 확인해주세요.")
            return
        
        # 성공 메시지
        st.markdown(f"""
        <div style="background-color: #d4edda; border: 1px solid #c3e6cb; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            ✅ <strong>데이터 로드 완료:</strong> {len(df):,}개 문서
        </div>
        """, unsafe_allow_html=True)
        
        # 데이터 미리보기
        with st.expander("📊 데이터 미리보기 (처음 5개 행)", expanded=False):
            st.dataframe(df[required_cols].head(5), use_container_width=True)
            st.caption(f"전체 데이터: {len(df):,}개 행, {len(df.columns)}개 컬럼")
            
    except pd.errors.EmptyDataError:
        st.error("❌ CSV 파일이 비어있습니다.")
        return
    except pd.errors.ParserError as e:
        st.error(f"❌ CSV 파일 파싱 오류: {e}")
        st.info("💡 CSV 파일 형식이 올바른지 확인해주세요.")
        return
    except Exception as e:
        st.error(f"❌ CSV 파일 로드 실패: {e}")
        st.exception(e)
        return
    
    try:
        # JSON 편집기에서 수정된 내용 사용
        if 'json_editor' in st.session_state and st.session_state.json_editor:
            term_db_content = st.session_state.json_editor
        else:
            term_db_content = uploaded_term_db.read().decode('utf-8')
            uploaded_term_db.seek(0)
        
        TERM_DB = load_term_db_from_json(term_db_content)
        
        if not TERM_DB:
            st.warning("⚠️ Term DB가 비어있습니다. JSON 파일 형식을 확인해주세요.")
            return
        
        # 성공 메시지
        st.markdown(f"""
        <div style="background-color: #d4edda; border: 1px solid #c3e6cb; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            ✅ <strong>Term DB 로드 완료:</strong> {len(TERM_DB)}개 라벨
        </div>
        """, unsafe_allow_html=True)
        
        # Term DB 미리보기
        with st.expander("📋 Term DB 미리보기", expanded=False):
            for label, terms in TERM_DB.items():
                st.write(f"**{label}**: {len(terms)}개 키워드")
                st.caption(f"키워드 예시: {', '.join(list(terms)[:5])}...")
        
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON 파일 형식 오류: {e}")
        st.info("💡 JSON 파일 형식이 올바른지 확인해주세요.")
        return
    except Exception as e:
        st.error(f"❌ Term DB 파일 로드 실패: {e}")
        st.exception(e)
        return
    
    st.markdown("---")
    st.markdown("### 다음 단계: 파라미터 설정 및 라벨링 실행")
    
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
    
    # ============================================================================
    # 회사 분류 설정
    # ============================================================================
    st.markdown("---")
    st.markdown('<div class="sub-header">🏢 회사 분류 설정</div>', unsafe_allow_html=True)
    
    # 기본 회사 설정
    default_company_config = {
        'Samsung Electronics': ['삼성전자', '삼성', 'samsung'],
        'SK Hynix': ['하이닉스', 'sk하이닉스', 'sk hynix']
    }
    
    # 세션 상태에 회사 설정 저장
    if 'company_config' not in st.session_state:
        st.session_state['company_config'] = default_company_config.copy()
    
    with st.expander("📝 회사명 및 키워드 설정", expanded=False):
        st.info("💡 기본 설정: 삼성전자, SK하이닉스. 필요시 회사 추가/수정 가능합니다.")
        
        # 회사 추가/수정 UI
        company_config_editor = {}
        
        # 기존 회사들 표시 및 수정
        for idx, (company_name, keywords) in enumerate(st.session_state['company_config'].items()):
            st.markdown(f"**회사 {idx + 1}**")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                new_company_name = st.text_input(
                    "회사명",
                    value=company_name,
                    key=f"company_name_{idx}",
                    help="회사명을 입력하세요 (예: Samsung Electronics)"
                )
            
            with col2:
                if st.button("🗑️ 삭제", key=f"delete_company_{idx}", use_container_width=True):
                    # 삭제 처리
                    temp_config = st.session_state['company_config'].copy()
                    del temp_config[company_name]
                    st.session_state['company_config'] = temp_config
                    st.rerun()
            
            keywords_str = st.text_input(
                "키워드 (쉼표로 구분)",
                value=", ".join(keywords),
                key=f"keywords_{idx}",
                help="이 회사를 식별할 키워드를 쉼표로 구분하여 입력하세요"
            )
            
            # 키워드 파싱
            keywords_list = [k.strip() for k in keywords_str.split(',') if k.strip()]
            if new_company_name and keywords_list:
                company_config_editor[new_company_name] = keywords_list
        
        # 새 회사 추가
        st.markdown("---")
        st.markdown("**➕ 새 회사 추가**")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            new_company_name_input = st.text_input(
                "새 회사명",
                value="",
                key="new_company_name",
                placeholder="예: Apple, TSMC 등"
            )
        
        with col2:
            new_company_keywords_input = st.text_input(
                "키워드 (쉼표로 구분)",
                value="",
                key="new_company_keywords",
                placeholder="예: 애플, apple, iphone"
            )
        
        if st.button("➕ 회사 추가", key="add_company", use_container_width=True):
            if new_company_name_input and new_company_keywords_input:
                keywords_list = [k.strip() for k in new_company_keywords_input.split(',') if k.strip()]
                if keywords_list:
                    st.session_state['company_config'][new_company_name_input] = keywords_list
                    st.rerun()
            else:
                st.warning("회사명과 키워드를 모두 입력해주세요.")
        
        # 기본값으로 초기화 버튼
        if st.button("🔄 기본값으로 초기화", key="reset_company_config", use_container_width=True):
            st.session_state['company_config'] = default_company_config.copy()
            st.rerun()
        
        # 최종 설정 표시
        if company_config_editor:
            st.session_state['company_config'] = company_config_editor
    
    # 회사 설정 미리보기
    st.markdown("**현재 회사 설정:**")
    for company_name, keywords in st.session_state['company_config'].items():
        st.caption(f"• **{company_name}**: {', '.join(keywords)}")
    
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
            # 라벨링 처리 (회사 설정 포함)
            df_labeled = process_labeling(df.copy(), TERM_DB, config, st.session_state['company_config'])
            
            # 필터링 적용
            df_original_len = len(df_labeled)
            
            if config['remove_unknown']:
                df_labeled = df_labeled[df_labeled['aspect_term'] != 'Unknown'].copy()
            
            df_labeled = df_labeled[df_labeled['match_count'] >= config['min_match_filter']].copy()
            
            # 세션 상태에 저장
            st.session_state['df_labeled'] = df_labeled
            st.session_state['df_original_len'] = df_original_len
            st.session_state['config'] = config
            
            # 깔끔한 회색 배경
            st.markdown("""
            <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                ✅ <strong>라벨링 완료!</strong>
            </div>
            """, unsafe_allow_html=True)
    
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
            else:
                # company 컬럼이 없으면 match_count 분포 표시
                match_dist = df_labeled['match_count'].value_counts().sort_index()
                
                # 파란 계열 그라데이션
                n = len(match_dist)
                colors = []
                for i in range(n):
                    ratio = i / max(n - 1, 1)
                    r = int(26 + (115 - 26) * ratio)
                    g = int(84 + (169 - 84) * ratio)
                    b = int(144 + (214 - 144) * ratio)
                    colors.append(f'rgb({r},{g},{b})')
                
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=match_dist.index,
                        y=match_dist.values,
                        text=match_dist.values,
                        textposition='outside',
                        textfont=dict(size=13, color='#2c3e50'),
                        marker=dict(
                            color=colors,
                            line=dict(color='white', width=2)
                        ),
                        hovertemplate='<b>매칭 수: %{x}</b><br>문서 수: %{y:,}<extra></extra>'
                    )
                ])
                
                fig2.update_layout(
                    title=dict(
                        text='매칭 키워드 수 분포',
                        font=dict(size=18, color='#2c3e50', family='Arial'),
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis=dict(
                        title='매칭 키워드 수',
                        title_font=dict(size=13, color='#7f8c8d'),
                        tickfont=dict(size=12, color='#2c3e50'),
                        showgrid=False,
                        showline=False
                    ),
                    yaxis=dict(
                        title='문서 수',
                        title_font=dict(size=13, color='#7f8c8d'),
                        tickfont=dict(size=12, color='#7f8c8d'),
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='#ecf0f1',
                        showline=False,
                        range=[0, match_dist.values.max() * 1.15]
                    ),
                    height=480,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(t=100, b=60, l=80, r=40)
                )
                
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
            # Excel 파일 생성
            buffer = BytesIO()
            try:
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_labeled.to_excel(writer, index=False, sheet_name='라벨링결과')
                
                excel_data = buffer.getvalue()
                
                st.download_button(
                    label="📥 Excel 다운로드",
                    data=excel_data,
                    file_name=f"hv_labeled_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except ImportError:
                st.warning("⚠️ openpyxl이 설치되지 않아 Excel 다운로드를 사용할 수 없습니다.")
                st.info("대신 CSV 다운로드를 사용해주세요.")

if __name__ == "__main__":
    main()