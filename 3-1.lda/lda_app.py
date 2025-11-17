# -*- coding: utf-8 -*-
"""
LDA 토픽 모델링 자동화 Streamlit 앱 (분석가용)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pickle
import hashlib
import json
import re
from io import BytesIO
from datetime import datetime
from pathlib import Path

# KoNLPy & Gensim
from konlpy.tag import Okt
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from tqdm import tqdm

# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(
    page_title="LDA 토픽 모델링",
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 기본 불용어
# ============================================================================
DEFAULT_STOP_WORDS = {
    '은', '는', '이', '가', '을', '를', '의', '와', '과', '도',
    '에', '로', '에서', '부터', '까지',
    '하다', '있다', '되다', '같다', '없다',
    '것', '수', '등', '개', '명', '년', '월', '일',
    '업계', '기업', '회사', '업체', '관계자',
    '올해', '내년', '작년', '이번', '지난해', '최근'
}

# ============================================================================
# LDA 클래스
# ============================================================================
class LDATopicModeling:
    """LDA 토픽 모델링"""
    
    def __init__(self, df, stop_words, min_noun_length=2):
        self.df = df
        self.stop_words = stop_words
        self.min_noun_length = min_noun_length
        self.processed_sentences = None
        self.valid_indices = None  # 유효한 문장의 원본 인덱스 저장
        self.dictionary = None
        self.corpus = None
        self.models = {}
        self.topics_dict = {}
        self.coherence_scores = {}
        self.perplexity_scores = {}  # Perplexity 추가
    
    def preprocess(self, use_cache=True):
        """형태소 분석"""
        if use_cache and 'preprocessed_data' in st.session_state:
            self.processed_sentences = st.session_state['preprocessed_data']
            self.valid_indices = st.session_state['valid_indices']
            return
        
        okt = Okt()
        
        def clean_text(text):
            if pd.isna(text):
                return []
            text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', str(text))
            try:
                nouns = okt.nouns(text)
                return [
                    n for n in nouns
                    if len(n) >= self.min_noun_length
                    and not n.isdigit()
                    and n not in self.stop_words
                ]
            except:
                return []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_processed = []
        valid_indices = []
        
        for i, text in enumerate(self.df['sentence']):
            cleaned = clean_text(text)
            all_processed.append(cleaned)
            if len(cleaned) > 0:
                valid_indices.append(i)
            
            if i % 100 == 0:
                progress_bar.progress((i + 1) / len(self.df))
                status_text.text(f"형태소 분석 중... {i+1}/{len(self.df)}")
        
        progress_bar.progress(1.0)
        status_text.text(f"형태소 분석 완료: {len(self.df):,}개")
        
        # 유효한 문장만 저장
        self.processed_sentences = [all_processed[i] for i in valid_indices]
        self.valid_indices = valid_indices
        
        # 캐시 저장
        st.session_state['preprocessed_data'] = self.processed_sentences
        st.session_state['valid_indices'] = self.valid_indices
    
    def create_dict_corpus(self, no_below, no_above, keep_n):
        """Dictionary & Corpus 생성"""
        self.dictionary = corpora.Dictionary(self.processed_sentences)
        original_size = len(self.dictionary)
        
        self.dictionary.filter_extremes(
            no_below=no_below,
            no_above=no_above,
            keep_n=keep_n
        )
        
        self.corpus = [self.dictionary.doc2bow(text) for text in self.processed_sentences]
        
        return original_size, len(self.dictionary)
    
    def train_lda(self, n_topics, passes, iterations, alpha, eta):
        """LDA 학습"""
        model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=n_topics,
            passes=passes,
            iterations=iterations,
            random_state=42,
            per_word_topics=False,
            alpha=alpha,
            eta=eta
        )
        
        self.models[n_topics] = model
        
        # 토픽 할당
        doc_topics = []
        for bow in self.corpus:
            topic_dist = model.get_document_topics(bow)
            if topic_dist:
                dominant = max(topic_dist, key=lambda x: x[1])[0]
                doc_topics.append(dominant)
            else:
                doc_topics.append(-1)
        
        self.topics_dict[n_topics] = np.array(doc_topics)
        
        # Coherence 계산
        coherence_model = CoherenceModel(
            model=model,
            texts=self.processed_sentences,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        coherence = coherence_model.get_coherence()
        self.coherence_scores[n_topics] = coherence
        
        # Perplexity 계산
        perplexity = model.log_perplexity(self.corpus)
        self.perplexity_scores[n_topics] = perplexity
        
        return coherence, perplexity
    
    def get_result_df(self, n_topics):
        """결과 데이터프레임 생성"""
        topics = self.topics_dict[n_topics]
        
        # 유효한 인덱스만 사용하여 데이터프레임 생성
        result_df = self.df.iloc[self.valid_indices].copy()
        result_df['lda_topic'] = topics
        
        # 토픽이 할당되지 않은 문서 제거
        result_df = result_df[result_df['lda_topic'] != -1]
        
        return result_df

# ============================================================================
# 엘보우 포인트 계산 함수
# ============================================================================
def calculate_elbow_point(scores_dict, maximize=True):
    """
    엘보우 포인트 계산 
    
    Parameters:
    - scores_dict: {n_topics: score} 딕셔너리
    - maximize: True면 높을수록 좋음 (Coherence), False면 낮을수록 좋음 (Perplexity)
    
    Returns:
    - elbow_point: 최적 토픽 개수
    """
    if len(scores_dict) < 3:
        return None
    
    topics = np.array(sorted(scores_dict.keys()))
    scores = np.array([scores_dict[k] for k in topics])
    
    if not maximize:
        scores = -scores  # Perplexity는 낮을수록 좋으므로 부호 반전
    
    # 정규화 (0-1 범위로)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    topics_norm = (topics - topics.min()) / (topics.max() - topics.min() + 1e-10)
    
    # 시작점과 끝점을 잇는 직선까지의 거리 계산
    # 직선: y = mx + b
    m = (scores_norm[-1] - scores_norm[0]) / (topics_norm[-1] - topics_norm[0] + 1e-10)
    b = scores_norm[0]
    
    # 각 점에서 직선까지의 수직 거리
    distances = np.abs(scores_norm - (m * topics_norm + b)) / np.sqrt(m**2 + 1)
    
    # 가장 먼 점이 엘보우 포인트
    elbow_idx = np.argmax(distances)
    elbow_point = topics[elbow_idx]
    
    return int(elbow_point)

# ============================================================================
# 시각화 함수
# ============================================================================
def create_metrics_comparison_chart(coherence_scores, perplexity_scores):
    """Coherence & Perplexity 비교 차트 - 파란색 계열로 통일"""
    from plotly.subplots import make_subplots
    
    # 엘보우 포인트 계산 (효율성 균형점)
    coherence_elbow = calculate_elbow_point(coherence_scores, maximize=True)
    perplexity_elbow = calculate_elbow_point(perplexity_scores, maximize=False)
    
    # 최고 성능 값 계산
    best_coherence_topic = max(coherence_scores.keys(), key=lambda k: coherence_scores[k])
    best_perplexity_topic = min(perplexity_scores.keys(), key=lambda k: perplexity_scores[k])
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Coherence (높을수록 좋음) - 🔷효율: {coherence_elbow}개, 🔵최고: {best_coherence_topic}개',
            f'Perplexity (낮을수록 좋음) - 🔷효율: {perplexity_elbow}개, 🔵최고: {best_perplexity_topic}개'
        ),
        horizontal_spacing=0.15
    )
    
    # Coherence 바 차트 - 파란색 계열로 통일
    colors_coherence = []
    for k in coherence_scores.keys():
        if k == best_coherence_topic and k == coherence_elbow:
            colors_coherence.append('#0D47A1')  # 둘 다 해당 - 가장 진한 파랑
        elif k == coherence_elbow:
            colors_coherence.append('#64B5F6')  # 효율성 - 밝은 파랑 (골드 대신)
        elif k == best_coherence_topic:
            colors_coherence.append('#1565C0')  # 최고 성능 - 진한 파랑
        else:
            colors_coherence.append('#90CAF9')  # 일반 - 연한 파랑
    
    fig.add_trace(
        go.Bar(
            x=list(coherence_scores.keys()),
            y=list(coherence_scores.values()),
            text=[f"{v:.4f}" for v in coherence_scores.values()],
            textposition='auto',
            textfont=dict(color='white', size=11, family='Arial'),
            marker_color=colors_coherence,
            marker_line=dict(width=1.5, color='white'),
            name='Coherence'
        ),
        row=1, col=1
    )
    
    # Perplexity 바 차트 - 파란색 계열로 통일
    colors_perplexity = []
    for k in perplexity_scores.keys():
        if k == best_perplexity_topic and k == perplexity_elbow:
            colors_perplexity.append('#0D47A1')  # 둘 다 해당 - 가장 진한 파랑
        elif k == perplexity_elbow:
            colors_perplexity.append('#64B5F6')  # 효율성 - 밝은 파랑
        elif k == best_perplexity_topic:
            colors_perplexity.append('#1565C0')  # 최고 성능 - 진한 파랑
        else:
            colors_perplexity.append('#90CAF9')  # 일반 - 연한 파랑
    
    fig.add_trace(
        go.Bar(
            x=list(perplexity_scores.keys()),
            y=list(perplexity_scores.values()),
            text=[f"{v:.2f}" for v in perplexity_scores.values()],
            textposition='auto',
            textfont=dict(color='white', size=11, family='Arial'),
            marker_color=colors_perplexity,
            marker_line=dict(width=1.5, color='white'),
            name='Perplexity'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="토픽 개수", row=1, col=1)
    fig.update_xaxes(title_text="토픽 개수", row=1, col=2)
    fig.update_yaxes(title_text="Coherence", row=1, col=1)
    fig.update_yaxes(title_text="Perplexity", row=1, col=2)
    
    fig.update_layout(
        height=450,
        showlegend=False,
        title_text="토픽 개수별 평가 지표 비교 (🔷=효율성 균형점, 🔵=최고 성능)",
        title_font=dict(size=16, color='#1565C0', family='Arial'),
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white',
        font=dict(family='Arial', color='#37474F')
    )
    
    # 그리드 라인 스타일
    fig.update_xaxes(
        gridcolor='#E0E0E0',
        gridwidth=0.5,
        showline=True,
        linewidth=1,
        linecolor='#BDBDBD'
    )
    fig.update_yaxes(
        gridcolor='#E0E0E0',
        gridwidth=0.5,
        showline=True,
        linewidth=1,
        linecolor='#BDBDBD'
    )
    
    return fig

def create_coherence_chart(coherence_scores):
    """Coherence 점수 비교 차트 (호환성 유지)"""
    # 파란색 계열 그라데이션 생성
    n = len(coherence_scores)
    colors = []
    for i in range(n):
        ratio = i / max(n - 1, 1)
        r = int(26 + (179 - 26) * ratio)   # 26(#1a) → 179(#b3)
        g = int(84 + (217 - 84) * ratio)   # 84(#54) → 217(#d9)
        b = int(144 + (255 - 144) * ratio) # 144(#90) → 255(#ff)
        colors.append(f'rgb({r},{g},{b})')
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(coherence_scores.keys()),
            y=list(coherence_scores.values()),
            text=[f"{v:.4f}" for v in coherence_scores.values()],
            textposition='auto',
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            )
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='토픽 개수별 Coherence 점수',
            font=dict(size=18, color='#2c3e50', family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='토픽 개수',
            tickfont=dict(size=12, color='#2c3e50'),
            showgrid=False,
            showline=False
        ),
        yaxis=dict(
            title='Coherence 점수',
            title_font=dict(size=13, color='#7f8c8d'),
            tickfont=dict(size=12, color='#7f8c8d'),
            showgrid=True,
            gridwidth=1,
            gridcolor='#ecf0f1',
            showline=False
        ),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_topic_distribution_chart(result_df):
    """토픽별 문서 수 분포 - 파란색 계열 그라데이션"""
    topic_counts = result_df['lda_topic'].value_counts().sort_index()
    
    # 파란색 계열 그라데이션 생성
    n = len(topic_counts)
    colors = []
    for i in range(n):
        ratio = i / max(n - 1, 1)
        r = int(26 + (179 - 26) * ratio)   # 26(#1a) → 179(#b3)
        g = int(84 + (217 - 84) * ratio)   # 84(#54) → 217(#d9)
        b = int(144 + (255 - 144) * ratio) # 144(#90) → 255(#ff)
        colors.append(f'rgb({r},{g},{b})')
    
    fig = go.Figure(data=[
        go.Bar(
            x=topic_counts.index,
            y=topic_counts.values,
            text=topic_counts.values,
            textposition='auto',
            textfont=dict(color='white', size=11, family='Arial'),
            marker=dict(
                color=colors,
                line=dict(width=1.5, color='white')
            )
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='토픽별 문서 수',
            font=dict(size=18, color='#2c3e50', family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='토픽 번호',
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
            showline=False
        ),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_topic_keywords_table(model, n_topics, top_n=10):
    """토픽별 키워드 테이블"""
    data = []
    for topic_id in range(n_topics):
        words = model.show_topic(topic_id, topn=top_n)
        keywords = ', '.join([f"{word}({prob:.3f})" for word, prob in words[:5]])
        data.append({
            '토픽': f"Topic {topic_id}",
            '주요 키워드': keywords
        })
    
    return pd.DataFrame(data)

# ============================================================================
# 메인 앱
# ============================================================================
def main():
    # 헤더
    st.markdown('<div class="main-header">LDA 토픽 모델링</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # ============================================================================
    # 1. 파일 업로드
    # ============================================================================
    st.markdown('<div class="sub-header">📁 1. 데이터 업로드</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "CSV 파일 업로드 (sentence 컬럼 필수)",
        type=['csv'],
        help="LDA 토픽 모델링을 수행할 CSV 파일"
    )
    
    # ============================================================================
    # 파일 변경 감지 및 캐시 초기화
    # ============================================================================
    current_file_name = uploaded_file.name if uploaded_file else None
    
    if 'prev_file_name' not in st.session_state:
        st.session_state['prev_file_name'] = None
    
    # 파일이 바뀌면 캐시 초기화
    if current_file_name != st.session_state['prev_file_name']:
        if 'preprocessed_data' in st.session_state:
            del st.session_state['preprocessed_data']
        if 'valid_indices' in st.session_state:
            del st.session_state['valid_indices']
        if 'lda' in st.session_state:
            del st.session_state['lda']
        if 'results' in st.session_state:
            del st.session_state['results']
        
        st.session_state['prev_file_name'] = current_file_name
    
    if uploaded_file is None:
        st.info("⬆️ CSV 파일을 업로드해주세요.")
        return
    
    # 데이터 로드
    try:
        df = pd.read_csv(uploaded_file)
        st.markdown(f"""
        <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            ✅ <strong>데이터 로드 완료:</strong> {len(df):,}개 문서
        </div>
        """, unsafe_allow_html=True)
        
        if 'sentence' not in df.columns:
            st.error("❌ 'sentence' 컬럼이 없습니다.")
            return
        
    except Exception as e:
        st.error(f"❌ 파일 로드 실패: {e}")
        return
    
    st.markdown("---")
    
    # ============================================================================
    # 2. 파라미터 설정
    # ============================================================================
    st.markdown('<div class="sub-header">⚙️ 2. 파라미터 설정</div>', unsafe_allow_html=True)
    
    # 토픽 개수
    st.markdown("**토픽 개수 설정**")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        topic_numbers_input = st.text_input(
            "학습할 토픽 개수 (쉼표로 구분)",
            value="5, 10, 15, 20",
            help="예: 5, 10, 15, 20"
        )
    
    with col2:
        try:
            topic_numbers = [int(x.strip()) for x in topic_numbers_input.split(',')]
            st.info(f"{len(topic_numbers)}개 설정")
        except:
            st.error("숫자와 쉼표만 입력")
            return
    
    st.markdown("---")
    
    # LDA 하이퍼파라미터
    st.markdown("**LDA 하이퍼파라미터**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        passes = st.slider("Passes", 1, 50, 10, help="전체 코퍼스를 몇 번 반복할지")
    
    with col2:
        iterations = st.slider("Iterations", 50, 500, 100, help="각 문서를 몇 번 업데이트할지")
    
    with col3:
        alpha_mode = st.radio(
            "Alpha 설정",
            options=['auto', 'symmetric', 'asymmetric', 'manual'],
            horizontal=True,
            help="모델 학습 방식 선택"
        )
        
        if alpha_mode == 'auto':
            alpha = 'auto'
            st.caption("✅ 데이터로부터 자동 최적화")
        elif alpha_mode == 'symmetric':
            alpha = 'symmetric'
            st.caption("✅ 모든 토픽 동일 가중치 (1/K)")
        elif alpha_mode == 'asymmetric':
            alpha = 'asymmetric'
            st.caption("✅ 토픽별 다른 가중치 (자동)")
        else:  # manual
            alpha = st.number_input(
                "Alpha 값",
                min_value=0.001,
                max_value=10.0,
                value=0.1,
                step=0.01,
                format="%.3f",
                help="문서-토픽 분포. 낮을수록 문서가 적은 토픽에 집중"
            )
            st.caption(f"현재값: {alpha}")
    
    with col4:
        eta_mode = st.radio(
            "Eta (Beta) 설정",
            options=['auto', 'symmetric', 'manual'],
            horizontal=True,
            help="모델 학습 방식 선택"
        )
        
        if eta_mode == 'auto':
            eta = 'auto'
            st.caption("✅ 데이터로부터 자동 최적화")
        elif eta_mode == 'symmetric':
            eta = 'symmetric'
            st.caption("✅ 모든 단어 동일 가중치 (1/V)")
        else:  # manual
            eta = st.number_input(
                "Eta 값",
                min_value=0.001,
                max_value=10.0,
                value=0.01,
                step=0.01,
                format="%.3f",
                help="토픽-단어 분포. 낮을수록 토픽이 적은 단어에 집중"
            )
            st.caption(f"현재값: {eta}")
    
    st.markdown("---")
    
    # Dictionary 필터링
    st.markdown("**Dictionary 필터링**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        no_below = st.number_input(
            "no_below",
            min_value=1,
            max_value=100,
            value=5,
            help="최소 문서 출현 빈도"
        )
    
    with col2:
        no_above = st.slider(
            "no_above",
            0.0, 1.0, 0.5,
            help="최대 문서 출현 비율"
        )
    
    with col3:
        keep_n = st.number_input(
            "keep_n",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
            help="유지할 최대 단어 수"
        )
    
    st.markdown("---")
    
    # 불용어 관리
    st.markdown("**불용어 관리**")
    
    with st.expander("📝 불용어 편집 (선택사항)"):
        st.write("**현재 기본 불용어:**")
        stop_words_text = st.text_area(
            "불용어 목록 (쉼표로 구분)",
            value=', '.join(sorted(DEFAULT_STOP_WORDS)),
            height=150,
            help="불용어를 쉼표로 구분하여 입력하세요"
        )
        
        stop_words = set([w.strip() for w in stop_words_text.split(',') if w.strip()])
        st.info(f"✅ 총 {len(stop_words)}개 불용어 설정")
    
    # 형태소 분석 옵션
    col1, col2 = st.columns(2)
    
    with col1:
        min_noun_length = st.slider(
            "최소 명사 길이",
            1, 5, 2,
            help="이 길이보다 짧은 명사는 제외"
        )
    
    with col2:
        use_cache = st.checkbox(
            "캐시 사용",
            value=True,
            help="이전 형태소 분석 결과 재사용"
        )
    
    # 파라미터 요약
    with st.expander("📋 설정 요약"):
        st.write(f"""
        **토픽 개수:** {', '.join(map(str, topic_numbers))}
        
        **LDA 파라미터:**
        - Passes: {passes}
        - Iterations: {iterations}
        - Alpha: {alpha}
        - Eta: {eta}
        
        **Dictionary 필터:**
        - no_below: {no_below}
        - no_above: {no_above}
        - keep_n: {keep_n:,}
        
        **전처리:**
        - 최소 명사 길이: {min_noun_length}
        - 불용어: {len(stop_words)}개
        """)
    
    st.markdown("---")
    
    # ============================================================================
    # 3. 학습 실행
    # ============================================================================
    st.markdown('<div class="sub-header">🚀 3. 학습 실행</div>', unsafe_allow_html=True)
    
    if st.button("학습 시작", type="primary", use_container_width=True):
        start_time = time.time()
        
        # 초기화
        lda = LDATopicModeling(df, stop_words, min_noun_length)
        
        # 전처리
        with st.spinner("형태소 분석 중..."):
            lda.preprocess(use_cache=use_cache)
        
        st.markdown(f"""
        <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            ✅ <strong>전처리 완료:</strong> {len(lda.processed_sentences):,}개 문장
        </div>
        """, unsafe_allow_html=True)
        
        # Dictionary & Corpus
        with st.spinner("Dictionary & Corpus 생성 중..."):
            original_size, filtered_size = lda.create_dict_corpus(no_below, no_above, keep_n)
        
        st.markdown(f"""
        <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            ✅ <strong>Dictionary 생성 완료</strong><br>
            원본: {original_size:,}개 → 필터링 후: {filtered_size:,}개
        </div>
        """, unsafe_allow_html=True)
        
        # 학습
        st.markdown("**LDA 학습 진행**")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, n_topics in enumerate(topic_numbers):
            status_text.text(f"학습 중: {n_topics}개 토픽...")
            
            with st.spinner(f"{n_topics}개 토픽 학습 중..."):
                coherence, perplexity = lda.train_lda(n_topics, passes, iterations, alpha, eta)
            
            results.append({
                '토픽 개수': n_topics,
                'Coherence': f"{coherence:.4f}",
                'Perplexity': f"{perplexity:.2f}",
                '문서 수': f"{(lda.topics_dict[n_topics] != -1).sum():,}"
            })
            
            progress_bar.progress((i + 1) / len(topic_numbers))
        
        status_text.text("✅ 모든 학습 완료!")
        
        # 결과 저장
        st.session_state['lda'] = lda
        st.session_state['results'] = results
        
        # 실행 시간
        elapsed = time.time() - start_time
        st.markdown(f"""
        <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            🎉 <strong>학습 완료!</strong> (총 소요 시간: {elapsed/60:.1f}분)
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # ============================================================================
    # 4. 결과 출력
    # ============================================================================
    if 'lda' in st.session_state:
        lda = st.session_state['lda']
        results = st.session_state['results']
        
        st.markdown('<div class="sub-header">📊 3. 학습 결과</div>', unsafe_allow_html=True)
        
        # 전체 요약
        st.markdown("**학습 요약**")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        
        # Coherence & Perplexity 비교
        st.markdown("---")
        st.markdown("**평가 지표 비교**")
        
        # 엘보우 포인트 계산 (효율성 균형점)
        coherence_elbow = calculate_elbow_point(lda.coherence_scores, maximize=True)
        perplexity_elbow = calculate_elbow_point(lda.perplexity_scores, maximize=False)
        
        # 최고 성능 계산
        best_coherence_topic = max(lda.coherence_scores.keys(), key=lambda k: lda.coherence_scores[k])
        best_perplexity_topic = min(lda.perplexity_scores.keys(), key=lambda k: lda.perplexity_scores[k])
        
        # 추천 메시지 - 파란색 계열로 통일
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="
                background-color: #E3F2FD;
                padding: 1.5rem;
                border-radius: 8px;
            ">
                <h4 style="color: #1976D2; margin: 0 0 1rem 0; font-size: 1.1rem;">🔷 효율성 균형점</h4>
                <div style="color: #1565C0; font-size: 0.95rem; line-height: 1.6;">
                    <strong>Coherence:</strong> {coherence_elbow}개 토픽<br>
                    <strong>Perplexity:</strong> {perplexity_elbow}개 토픽
                </div>
                <p style="color: #1976D2; font-size: 0.85rem; margin-top: 0.8rem; margin-bottom: 0;">
                    💡 성능 대비 효율성이 가장 좋은 지점
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="
                background-color: #E8F4F8;
                padding: 1.5rem;
                border-radius: 8px;
            ">
                <h4 style="color: #0D47A1; margin: 0 0 1rem 0; font-size: 1.1rem;">🔵 최고 성능</h4>
                <div style="color: #1565C0; font-size: 0.95rem; line-height: 1.6;">
                    <strong>Coherence:</strong> {best_coherence_topic}개 ({lda.coherence_scores[best_coherence_topic]:.4f})<br>
                    <strong>Perplexity:</strong> {best_perplexity_topic}개 ({lda.perplexity_scores[best_perplexity_topic]:.2f})
                </div>
                <p style="color: #1976D2; font-size: 0.85rem; margin-top: 0.8rem; margin-bottom: 0;">
                    💡 각 지표에서 최고 성능을 보이는 값
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background-color: #FAFAFA;
            padding: 0.8rem 1.2rem;
            border-radius: 8px;
            margin-top: 1rem;
        ">
            <p style="color: #546E7A; font-size: 0.9rem; margin: 0;">
                📌 <strong>선택 가이드:</strong> 해석 용이성과 속도를 원하면 효율성 균형점(🔷밝은 파랑), 최고 정확도를 원하면 최고 성능(🔵진한 파랑) 선택
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        fig_metrics = create_metrics_comparison_chart(lda.coherence_scores, lda.perplexity_scores)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        
        st.markdown("---")
        
        # 토픽별 상세 결과
        st.markdown("**토픽별 상세 결과**")
        
        selected_n_topics = st.selectbox(
            "토픽 개수 선택",
            options=sorted(lda.models.keys()),
            index=len(lda.models.keys())-1
        )
        
        model = lda.models[selected_n_topics]
        result_df = lda.get_result_df(selected_n_topics)
        
        # 통계
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("전체 문서", f"{len(result_df):,}")
        with col2:
            st.metric("토픽 수", selected_n_topics)
        with col3:
            st.metric("Coherence", f"{lda.coherence_scores[selected_n_topics]:.4f}")
        with col4:
            st.metric("Perplexity", f"{lda.perplexity_scores[selected_n_topics]:.2f}")
        
        # 토픽 분포
        fig_dist = create_topic_distribution_chart(result_df)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # 토픽별 키워드
        st.markdown("**토픽별 주요 키워드**")
        keywords_df = create_topic_keywords_table(model, selected_n_topics)
        st.dataframe(keywords_df, use_container_width=True)
        
        st.markdown("---")
        
        # ============================================================================
        # 토픽 선택 및 필터링
        # ============================================================================
        st.markdown('<div class="sub-header">🎯 토픽 선택 및 필터링</div>', unsafe_allow_html=True)
        
        st.write("**분석할 토픽을 선택하세요** (감성분석/회귀분석 등 후속 분석용)")
        
        # 토픽 ID 리스트 생성
        unique_topics = list(range(selected_n_topics))
        
        # 토픽별 정보를 데이터프레임으로 만들기
        topic_info = []
        for topic_id in unique_topics:
            count = (result_df['lda_topic'] == topic_id).sum()
            words = model.show_topic(topic_id, topn=5)
            keywords = ', '.join([word for word, prob in words])
            topic_info.append({
                'Topic ID': f"Topic {topic_id}",
                '문서 수': count,
                '주요 키워드': keywords
            })
        
        topic_info_df = pd.DataFrame(topic_info)
        
        # 토픽 정보 표시
        st.dataframe(topic_info_df, use_container_width=True, height=300)
        
        # 토픽 선택 UI
        col1, col2 = st.columns([3, 1])
        
        # session_state 초기화 (버튼 앞에)
        if 'selected_topics_list' not in st.session_state:
            st.session_state['selected_topics_list'] = unique_topics[:min(3, len(unique_topics))]
        
        with col2:
            if st.button("🔄 전체 선택", key="select_all", use_container_width=True):
                st.session_state['selected_topics_list'] = unique_topics
            
            if st.button("❌ 전체 해제", key="clear_all", use_container_width=True):
                st.session_state['selected_topics_list'] = []
        
        with col1:
            selected_topics = st.multiselect(
                "분석할 토픽 선택",
                options=unique_topics,
                default=st.session_state['selected_topics_list'],
                help="여러 개 선택 가능합니다. 선택한 토픽만 필터링하여 저장됩니다.",
                format_func=lambda x: f"Topic {x}"
            )
        
        # multiselect 값 변경 시 즉시 session_state 업데이트
        if selected_topics != st.session_state['selected_topics_list']:
            st.session_state['selected_topics_list'] = selected_topics
        
        # 선택 결과 표시
        if selected_topics:
            filtered_df = result_df[result_df['lda_topic'].isin(selected_topics)].copy()
            
            st.markdown(f"""
            <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                ✅ <strong>{len(selected_topics)}개 토픽 선택됨</strong> (총 {len(filtered_df):,}개 문서)
            </div>
            """, unsafe_allow_html=True)
            
            # 선택한 토픽 요약
            with st.expander("📊 선택한 토픽 요약"):
                for topic_id in selected_topics:
                    count = (filtered_df['lda_topic'] == topic_id).sum()
                    pct = count / len(filtered_df) * 100
                    words = model.show_topic(topic_id, topn=5)
                    keywords = ', '.join([f"{word}({prob:.3f})" for word, prob in words])
                    st.write(f"**Topic {topic_id}** ({count:,}개, {pct:.1f}%): {keywords}")
            
            # 세션에 저장 (다른 분석에서 사용 가능)
            st.session_state['filtered_df'] = filtered_df
            st.session_state['selected_topics'] = selected_topics
            
        else:
            st.warning("⚠️ 최소 1개 이상의 토픽을 선택해주세요.")
            filtered_df = result_df
        
        # 상세 토픽 정보
        with st.expander("🔍 토픽별 상세 정보"):
            for topic_id in unique_topics:
                count = (result_df['lda_topic'] == topic_id).sum()
                words = model.show_topic(topic_id, topn=10)
                keywords = ', '.join([f"{word}({prob:.3f})" for word, prob in words])
                
                # 선택된 토픽 강조
                if topic_id in selected_topics:
                    st.markdown(f"**✅ Topic {topic_id}** ({count:,}개 문서) - **선택됨**")
                else:
                    st.markdown(f"**Topic {topic_id}** ({count:,}개 문서)")
                st.text(keywords)
                st.markdown("---")
        
        # 데이터 미리보기
        with st.expander("📄 데이터 미리보기 (처음 100개)"):
            display_cols = ['sentence', 'lda_topic']
            if 'company' in filtered_df.columns:
                display_cols.insert(1, 'company')
            if 'label' in filtered_df.columns:
                display_cols.insert(2, 'label')
            
            display_cols = [col for col in display_cols if col in filtered_df.columns]
            st.dataframe(filtered_df[display_cols].head(100), use_container_width=True)
        
        st.markdown("---")
        
                # =====================================================================
        # 5. 다운로드
        # =====================================================================
        st.markdown('<div class="sub-header">💾 4. 결과 다운로드</div>', unsafe_allow_html=True)

        st.info(f"💡 **선택한 토픽 ({len(selected_topics)}개)의 데이터만 저장됩니다** ({len(filtered_df):,}개 문서)")

        col1, col2, col3 = st.columns(3)

        # -----------------------------
        # 5-1. CSV 다운로드
        # -----------------------------
        with col1:
            st.write("**📥 CSV 다운로드**")

            csv_utf8sig = filtered_df.to_csv(index=False, encoding="utf-8-sig")
            file_name_csv = f"lda_{selected_n_topics}_topics_selected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            st.download_button(
                label="CSV 다운로드 (UTF-8-SIG, Excel용)",
                data=csv_utf8sig,
                file_name=file_name_csv,
                mime="text/csv",
                use_container_width=True,
                key="lda_download_csv",
            )

            st.caption(f"💡 선택한 토픽: {len(selected_topics)}개 / 문서: {len(filtered_df):,}개")

        # -----------------------------
        # 5-2. Excel 다운로드
        # -----------------------------
        with col2:
            st.write("**📥 Excel 다운로드**")

            buffer = BytesIO()
            try:
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    # 시트 1: 선택한 토픽의 문서
                    filtered_df.to_excel(writer, index=False, sheet_name="선택한토픽")
                    # 시트 2: 전체 토픽 키워드
                    keywords_df.to_excel(writer, index=False, sheet_name="전체토픽키워드")
                    # 시트 3: 선택한 토픽 정보만
                    selected_info_df = topic_info_df[
                        topic_info_df["Topic ID"].isin([f"Topic {x}" for x in selected_topics])
                    ]
                    selected_info_df.to_excel(writer, index=False, sheet_name="선택한토픽정보")

                excel_data = buffer.getvalue()
                file_name_xlsx = f"lda_{selected_n_topics}_topics_selected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

                st.download_button(
                    label="Excel 다운로드",
                    data=excel_data,
                    file_name=file_name_xlsx,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="lda_download_excel",
                )

            except ImportError:
                st.warning("⚠️ openpyxl이 설치되지 않아 Excel 다운로드를 사용할 수 없습니다.")
                st.info("`pip install openpyxl` 후 다시 시도해주세요.")

            st.caption("💡 3개 시트 포함 (선택한토픽, 전체토픽키워드, 선택한토픽정보)")

        # -----------------------------
        # 5-3. 메타데이터(JSON) 다운로드
        # -----------------------------
        with col3:
            st.write("**📥 메타데이터(JSON) 다운로드**")

            metadata = {
                "n_topics": selected_n_topics,
                "total_documents": len(result_df),
                "selected_topics": selected_topics,
                "filtered_documents": len(filtered_df),
                "coherence_score": float(lda.coherence_scores[selected_n_topics]),
                "perplexity_score": float(lda.perplexity_scores[selected_n_topics]),
                "parameters": {
                    "passes": passes,
                    "iterations": iterations,
                    "alpha": str(alpha),  # auto / symmetric 등 문자열일 수 있어서 str
                    "eta": str(eta),
                    "no_below": no_below,
                    "no_above": no_above,
                    "keep_n": keep_n,
                },
                "timestamp": datetime.now().isoformat(),
            }

            json_str = json.dumps(metadata, ensure_ascii=False, indent=2)
            file_name_json = f"lda_{selected_n_topics}_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            st.download_button(
                label="메타데이터 JSON 다운로드",
                data=json_str.encode("utf-8"),
                file_name=file_name_json,
                mime="application/json",
                use_container_width=True,
                key="lda_download_json",
            )

            st.caption("💡 학습 파라미터 및 선택 토픽 정보 포함")


if __name__ == "__main__":
    main()