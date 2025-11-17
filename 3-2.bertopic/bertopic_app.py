# -*- coding: utf-8 -*-
"""
BERTopic 토픽 모델링 Streamlit 앱 (분석가용)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import re
import os
from pathlib import Path
from io import BytesIO
from datetime import datetime
import json

# BERTopic & Related
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(
    page_title="BERTopic 토픽 모델링",
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 한국어 임베딩 모델 목록
# ============================================================================
EMBEDDING_MODELS = {
    'jhgan/ko-sroberta-multitask': 'KoSRoBERTa (추천)',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': 'Multilingual MiniLM',
    'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens': 'XLM-R 100langs'
}

# ============================================================================
# 헬퍼 함수
# ============================================================================
def scale_parameters(data_size, base_size=309513):
    """데이터 크기에 따른 파라미터 스케일링"""
    ratio = data_size / base_size
    return {
        'min_cluster_size': max(30, int(50 * ratio)),
        'min_samples': max(5, int(10 * ratio)),
        'n_neighbors': max(15, min(30, int(25 * np.sqrt(ratio))))
    }

def smart_tokenizer(text):
    """스마트 토크나이저"""
    pattern = r'\b[가-힣]{2,}\b|\b[A-Z]{2,}\b|\b[a-z]{3,}\b'
    tokens = re.findall(pattern, text.lower())
    filtered = []
    for token in tokens:
        if any(char.isdigit() for char in token):
            continue
        if len(token) < 2:
            continue
        filtered.append(token)
    return filtered

# ============================================================================
# 시각화 함수
# ============================================================================
def create_topic_distribution_chart(topics):
    """토픽 분포 차트"""
    topic_counts = pd.Series(topics).value_counts().sort_index()
    topic_counts = topic_counts[topic_counts.index != -1]  # Outlier 제외
    
    fig = go.Figure(data=[
        go.Bar(
            x=topic_counts.index,
            y=topic_counts.values,
            text=topic_counts.values,
            textposition='auto',
            marker_color='#1565C0'
        )
    ])
    
    fig.update_layout(
        title='토픽별 문서 수 (Outlier 제외)',
        xaxis_title='토픽 번호',
        yaxis_title='문서 수',
        height=400
    )
    
    return fig

def create_outlier_chart(topics):
    """Outlier vs 토픽 할당 비율"""
    outlier_count = (topics == -1).sum()
    topic_count = (topics != -1).sum()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['토픽 할당', 'Outlier'],
            values=[topic_count, outlier_count],
            marker_colors=['#0D47A1','#1565C0'],
            hole=0.4
        )
    ])
    
    fig.update_layout(
        title='토픽 할당 vs Outlier',
        height=400
    )
    
    return fig

def create_keywords_table(topic_model, topics):
    """토픽별 키워드 테이블"""
    unique_topics = sorted([t for t in set(topics) if t != -1])
    
    data = []
    for topic_id in unique_topics:
        count = (topics == topic_id).sum()
        pct = count / len(topics) * 100
        words = topic_model.get_topic(topic_id)
        
        if words:
            keywords = ', '.join([f"{w[0]}({w[1]:.3f})" for w in words[:5]])
            data.append({
                '토픽': f"Topic {topic_id}",
                '문서 수': f"{count:,}",
                '비율': f"{pct:.1f}%",
                '주요 키워드': keywords
            })
    
    return pd.DataFrame(data)

# ============================================================================
# 메인 앱
# ============================================================================
def main():
    # 헤더
    st.markdown('<div class="main-header">BERTopic 토픽 모델링</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # ============================================================================
    # 1. 파일 업로드
    # ============================================================================
    st.markdown('<div class="sub-header">📁 1. 데이터 업로드</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "CSV 파일 업로드 (sentence 컬럼 필수)",
        type=['csv'],
        help="BERTopic 토픽 모델링을 수행할 CSV 파일"
    )
    
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
        
        texts = df['sentence'].tolist()
        
    except Exception as e:
        st.error(f"❌ 파일 로드 실패: {e}")
        return
    
    st.markdown("---")
    
    # ============================================================================
    # 2. 파라미터 설정
    # ============================================================================
    st.markdown('<div class="sub-header">⚙️ 2. 파라미터 설정</div>', unsafe_allow_html=True)
    
    # 임베딩 모델
    st.markdown("**임베딩 모델**")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        embedding_model_key = st.selectbox(
            "한국어 임베딩 모델 선택",
            options=list(EMBEDDING_MODELS.keys()),
            format_func=lambda x: EMBEDDING_MODELS[x],
            index=0
        )
    
    with col2:
        use_embedding_cache = st.checkbox(
            "임베딩 캐시 사용",
            value=True,
            help="이전 생성된 임베딩 재사용 (같은 세션 내)"
        )
    
    st.markdown("---")
    
    # UMAP 파라미터
    st.markdown("**UMAP 파라미터**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_components = st.slider(
            "차원 수 (n_components)",
            min_value=2,
            max_value=10,
            value=5,
            help="차원 축소 후 차원 수"
        )
    
    with col2:
        n_neighbors = st.slider(
            "이웃 수 (n_neighbors)",
            min_value=5,
            max_value=50,
            value=15,
            help="로컬 구조 학습에 사용할 이웃 수"
        )
    
    with col3:
        min_dist = st.slider(
            "최소 거리 (min_dist)",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.05,
            help="임베딩 공간에서 점들 간 최소 거리"
        )
    
    st.markdown("---")
    
    # HDBSCAN 파라미터
    st.markdown("**HDBSCAN 파라미터**")
    col1, col2 = st.columns(2)
    
    with col1:
        min_cluster_size = st.slider(
            "최소 클러스터 크기",
            min_value=10,
            max_value=200,
            value=50,
            help="클러스터로 인정되기 위한 최소 문서 수"
        )
    
    with col2:
        min_samples = st.slider(
            "최소 샘플 수",
            min_value=1,
            max_value=50,
            value=10,
            help="코어 포인트가 되기 위한 최소 이웃 수"
        )
    
    st.markdown("---")
    
    # 토픽 개수 설정
    st.markdown("**토픽 개수 설정**")
    col1, col2 = st.columns(2)
    
    with col1:
        topic_mode = st.radio(
            "토픽 개수 결정 방식",
            options=['자동', '수동'],
            index=0,
            horizontal=True,
            help="자동: 자동으로 최적 개수 결정, 수동: 직접 개수 지정"
        )
    
    with col2:
        if topic_mode == '수동':
            nr_topics = st.number_input(
                "토픽 개수",
                min_value=5,
                max_value=100,
                value=20,
                step=5
            )
        else:
            nr_topics = 'auto'
            st.info("자동으로 최적 토픽 개수를 결정합니다")
    
    st.markdown("---")
    
    # Vectorizer 파라미터
    st.markdown("**Vectorizer 파라미터**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_features = st.slider(
            "최대 단어 수",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="토픽 표현에 사용할 최대 단어 개수"
        )
    
    with col2:
        max_df = st.slider(
            "최대 문서 빈도",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="너무 자주 나타나는 단어 제외"
        )
    
    with col3:
        ngram_max = st.selectbox(
            "N-gram 최대값",
            options=[1, 2, 3],
            index=0,
            help="1: 단일 단어만, 2: 2단어 조합 포함"
        )
    
    st.markdown("---")
    
    # 샘플링 옵션
    st.markdown("**샘플링 옵션 (대용량 데이터용)**")
    col1, col2 = st.columns(2)
    
    with col1:
        use_sampling = st.checkbox(
            "샘플링 사용",
            value=len(df) > 50000,
            help="데이터가 많을 때 샘플로 학습 후 전체 예측"
        )
    
    with col2:
        if use_sampling:
            sample_size = st.number_input(
                "샘플 크기",
                min_value=1000,
                max_value=min(100000, len(df)),
                value=min(50000, len(df)),
                step=5000
            )
        else:
            sample_size = len(df)
    
    # 현재 설정 요약
    with st.expander("📋 현재 설정 요약"):
        st.write(f"""
        **임베딩**
        - 모델: {EMBEDDING_MODELS[embedding_model_key]}
        - 캐시 사용: {'예' if use_embedding_cache else '아니오'}
        
        **UMAP**
        - n_components: {n_components}
        - n_neighbors: {n_neighbors}
        - min_dist: {min_dist}
        
        **HDBSCAN**
        - min_cluster_size: {min_cluster_size}
        - min_samples: {min_samples}
        
        **토픽**
        - 결정 방식: {topic_mode}
        - 토픽 개수: {nr_topics if topic_mode == '수동' else '자동'}
        
        **Vectorizer**
        - max_features: {max_features}
        - max_df: {max_df}
        - ngram: (1, {ngram_max})
        
        **샘플링**
        - 사용: {'예' if use_sampling else '아니오'}
        - 크기: {sample_size:,}개 / {len(df):,}개
        """)
    
    st.markdown("---")
    
    # ============================================================================
    # 3. 학습 실행
    # ============================================================================
    if st.button("🚀 BERTopic 학습 시작", type="primary", use_container_width=True):
        start_time = time.time()
        
        try:
            # 1. 임베딩 생성
            st.markdown("### 1️⃣ 임베딩 생성")
            
            cache_key = f"embeddings_{embedding_model_key}_{len(texts)}"
            
            if use_embedding_cache and cache_key in st.session_state:
                st.info("✅ 캐시된 임베딩 사용")
                embeddings = st.session_state[cache_key]
            else:
                # Progress bar 추가
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(f"임베딩 모델 로딩 중... ({EMBEDDING_MODELS[embedding_model_key]})")
                progress_bar.progress(10)
                
                model = SentenceTransformer(embedding_model_key)
                
                progress_bar.progress(30)
                status_text.text(f"임베딩 생성 중... ({len(texts):,}개 문서)")
                
                # 배치 단위로 임베딩 생성 + 진행률 표시
                batch_size = 32
                embeddings_list = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    batch_embeddings = model.encode(
                        batch,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                    embeddings_list.append(batch_embeddings)
                    
                    # 진행률 업데이트 (30% ~ 90%)
                    progress = 30 + int((i / len(texts)) * 60)
                    progress_bar.progress(min(progress, 90))
                    status_text.text(f"임베딩 생성 중... {i+len(batch):,}/{len(texts):,} ({progress}%)")
                
                embeddings = np.vstack(embeddings_list)
                st.session_state[cache_key] = embeddings
                
                progress_bar.progress(100)
                status_text.text(f"✅ 임베딩 생성 완료!")
                
                st.markdown(f"""
                <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    ✅ <strong>임베딩 생성 완료:</strong> {embeddings.shape}
                </div>
                """, unsafe_allow_html=True)
                
                # progress bar 정리
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
            
            # 2. 샘플링 (옵션)
            st.markdown("### 2️⃣ 데이터 준비")
            
            if use_sampling and sample_size < len(texts):
                np.random.seed(42)
                sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
                train_embeddings = embeddings[sample_indices]
                train_texts = [texts[i] for i in sample_indices]
                st.info(f"샘플 사용: {sample_size:,}개로 학습")
            else:
                train_embeddings = embeddings
                train_texts = texts
                st.info(f"전체 데이터 사용: {len(texts):,}개")
            
            # 3. BERTopic 모델 학습
            st.markdown("### 3️⃣ BERTopic 모델 학습")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("UMAP 차원 축소 준비 중...")
            progress_bar.progress(20)
            
            # UMAP
            umap_model = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric='cosine',
                random_state=42
            )
            
            status_text.text("HDBSCAN 클러스터링 준비 중...")
            progress_bar.progress(40)
            
            # HDBSCAN
            hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method='eom',
                metric='euclidean',
                prediction_data=False
            )
            
            status_text.text("Vectorizer 준비 중...")
            progress_bar.progress(50)
            
            # Vectorizer
            vectorizer_model = CountVectorizer(
                tokenizer=smart_tokenizer,
                max_features=max_features,
                max_df=max_df,
                ngram_range=(1, ngram_max)
            )
            
            status_text.text("BERTopic 모델 생성 중...")
            progress_bar.progress(60)
            
            # BERTopic
            topic_model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                nr_topics=nr_topics if topic_mode == '수동' else 'auto',
                min_topic_size=max(10, int(len(texts) * 0.001)),
                calculate_probabilities=False,
                verbose=False
            )
            
            status_text.text("토픽 학습 중... (UMAP + HDBSCAN + c-TF-IDF)")
            progress_bar.progress(70)
            
            topics, probs = topic_model.fit_transform(train_texts, train_embeddings)
            topics = np.array(topics)
            
            progress_bar.progress(100)
            status_text.text("✅ 학습 완료!")
            
            st.markdown("""
            <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                ✅ <strong>학습 완료!</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # progress bar 정리
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # 4. 전체 데이터 예측 (샘플링 사용 시)
            if use_sampling and sample_size < len(texts):
                st.markdown("### 4️⃣ 전체 데이터 예측")
                with st.spinner("전체 데이터에 토픽 할당 중..."):
                    topics, _ = topic_model.transform(texts, embeddings)
                    topics = np.array(topics)
                st.markdown("""
                <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    ✅ <strong>예측 완료!</strong>
                </div>
                """, unsafe_allow_html=True)
            
            # 결과 저장
            st.session_state['topic_model'] = topic_model
            st.session_state['topics'] = topics
            st.session_state['df_result'] = df.copy()
            st.session_state['df_result']['bertopic_topic'] = topics
            st.session_state['df_result']['outlier'] = (topics == -1).astype(int)
            
            elapsed = time.time() - start_time
            st.markdown(f"""
            <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                🎉 <strong>전체 완료!</strong> (총 소요 시간: {elapsed/60:.1f}분)
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
            import traceback
            st.text(traceback.format_exc())
            return
        
        st.markdown("---")
    
    # ============================================================================
    # 4. 결과 출력
    # ============================================================================
    if 'topics' in st.session_state:
        topics = st.session_state['topics']
        topic_model = st.session_state['topic_model']
        df_result = st.session_state['df_result']
        
        st.markdown('<div class="sub-header">📊 3. 학습 결과</div>', unsafe_allow_html=True)
        
        # 주요 통계
        outlier_count = (topics == -1).sum()
        outlier_pct = outlier_count / len(topics) * 100
        unique_topics = sorted([t for t in set(topics) if t != -1])
        n_topics = len(unique_topics)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("전체 문서", f"{len(topics):,}")
        
        with col2:
            st.metric("토픽 수", n_topics)
        
        with col3:
            st.metric("Outlier", f"{outlier_count:,} ({outlier_pct:.1f}%)")
        
        with col4:
            # 평가
            if outlier_pct < 25:
                status = "우수"
            elif outlier_pct < 35:
                status = "양호"
            elif outlier_pct < 45:
                status = "보통"
            else:
                status = "개선필요"
            st.metric("평가", status)
        
        st.markdown("---")
        
        # 차트
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_topic_distribution_chart(topics)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = create_outlier_chart(topics)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        # 토픽별 키워드
        st.markdown("**토픽별 주요 키워드**")
        keywords_df = create_keywords_table(topic_model, topics)
        st.dataframe(keywords_df, use_container_width=True)
        
        st.markdown("---")
        
        # ============================================================================
        # 토픽 선택 및 필터링
        # ============================================================================
        st.markdown('<div class="sub-header">🎯 토픽 선택 및 필터링</div>', unsafe_allow_html=True)
        
        st.write("**분석할 토픽을 선택하세요** (감성분석/회귀분석 등 후속 분석용)")
        
        # 토픽별 정보를 데이터프레임으로 만들기
        topic_info = []
        for topic_id in unique_topics:
            count = (topics == topic_id).sum()
            pct = count / len(topics) * 100
            words = topic_model.get_topic(topic_id)
            if words:
                keywords = ', '.join([w[0] for w in words[:5]])
                topic_info.append({
                    'Topic ID': topic_id,
                    '문서 수': count,
                    '비율 (%)': f"{pct:.1f}",
                    '주요 키워드': keywords
                })
        
        topic_info_df = pd.DataFrame(topic_info)
        
        # 토픽 정보 표시
        st.dataframe(topic_info_df, use_container_width=True, height=300)
        
        # Outlier 포함 여부
        include_outlier = st.checkbox(
            "Outlier (-1) 포함",
            value=False,
            help="체크하면 Outlier 토픽도 결과에 포함됩니다"
        )
        
        # 토픽 선택 UI
        col1, col2 = st.columns([3, 1])
        
        # 선택 가능한 토픽 목록
        available_topics = unique_topics.copy()
        if include_outlier:
            available_topics = [-1] + available_topics
        
        # session_state 초기화
        if 'selected_bertopic_list' not in st.session_state:
            st.session_state['selected_bertopic_list'] = unique_topics[:min(3, len(unique_topics))]
        
        with col2:
            if st.button("🔄 전체 선택", key="select_all", use_container_width=True):
                st.session_state['selected_bertopic_list'] = available_topics
                st.rerun()
            
            if st.button("❌ 전체 해제", key="clear_all", use_container_width=True):
                st.session_state['selected_bertopic_list'] = []
                st.rerun()
        
        with col1:
            selected_topics = st.multiselect(
                "분석할 토픽 선택",
                options=available_topics,
                default=st.session_state['selected_bertopic_list'],
                help="여러 개 선택 가능합니다. 선택한 토픽만 필터링하여 저장됩니다.",
                format_func=lambda x: f"Topic {x}" if x != -1 else "Outlier (-1)"
            )
            
            # multiselect 값이 변경되면 session_state 업데이트
            st.session_state['selected_bertopic_list'] = selected_topics
        
        # 선택 결과 표시
        if selected_topics:
            filtered_df = df_result[df_result['bertopic_topic'].isin(selected_topics)].copy()
            
            st.markdown(f"""
            <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                ✅ <strong>{len(selected_topics)}개 토픽 선택됨</strong> (총 {len(filtered_df):,}개 문서)
            </div>
            """, unsafe_allow_html=True)
            
            # 선택한 토픽 요약
            with st.expander("📊 선택한 토픽 요약"):
                for topic_id in selected_topics:
                    count = (filtered_df['bertopic_topic'] == topic_id).sum()
                    pct = count / len(filtered_df) * 100
                    
                    if topic_id == -1:
                        st.write(f"**Outlier (-1)** ({count:,}개, {pct:.1f}%): 미분류 문서")
                    else:
                        words = topic_model.get_topic(topic_id)
                        if words:
                            keywords = ', '.join([f"{w[0]}({w[1]:.3f})" for w in words[:5]])
                            st.write(f"**Topic {topic_id}** ({count:,}개, {pct:.1f}%): {keywords}")
            
            # 토픽별 상세 정보 (LDA 스타일)
            with st.expander("🔍 토픽별 상세 정보"):
                for topic_id in unique_topics[:20]:  # 상위 20개만
                    count = (topics == topic_id).sum()
                    pct = count / len(topics) * 100
                    words = topic_model.get_topic(topic_id)
                    
                    if words:
                        keywords = ', '.join([f"{w[0]}({w[1]:.3f})" for w in words[:10]])
                        
                        # 선택된 토픽 강조
                        if topic_id in selected_topics:
                            st.markdown(f"**✅ Topic {topic_id}** ({count:,}개 문서, {pct:.1f}%) - **선택됨**")
                        else:
                            st.markdown(f"**Topic {topic_id}** ({count:,}개 문서, {pct:.1f}%)")
                        
                        st.text(keywords)
                        st.markdown("---")
            
            # 세션에 저장 (다른 분석에서 사용 가능)
            st.session_state['filtered_df'] = filtered_df
            st.session_state['selected_topics'] = selected_topics
            
            # 데이터 미리보기 (선택한 토픽만)
            with st.expander("📄 데이터 미리보기 (처음 100개)"):
                display_cols = ['sentence', 'bertopic_topic', 'outlier']
                if 'company' in filtered_df.columns:
                    display_cols.insert(1, 'company')
                if 'label' in filtered_df.columns:
                    display_cols.insert(2, 'label')
                
                display_cols = [col for col in display_cols if col in filtered_df.columns]
                st.dataframe(filtered_df[display_cols].head(100), use_container_width=True)
            
        else:
            st.warning("⚠️ 최소 1개 이상의 토픽을 선택해주세요.")
            filtered_df = df_result
        
        st.markdown("---")
        
        # ============================================================================
        # 5. 결과 저장
        # ============================================================================
        st.markdown('<div class="sub-header">💾 4. 결과 저장</div>', unsafe_allow_html=True)
        
        st.info(f"💡 **선택한 토픽 ({len(selected_topics)}개)의 데이터만 저장됩니다** ({len(filtered_df):,}개 문서)")
        
        col1, col2, col3 = st.columns(3)
        
        # CSV 저장
        with col1:
            st.write("**💾 CSV 저장**")
            
            default_path = str(Path.home() / "Desktop" / f"bertopic_result_selected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            
            save_path = st.text_input(
                "저장 경로",
                value=default_path,
                help="파일을 저장할 경로를 입력하세요",
                key="csv_path"
            )
            
            if st.button("💾 파일로 저장", key="save_csv", use_container_width=True):
                try:
                    filtered_df.to_csv(save_path, index=False, encoding='utf-8-sig')
                    st.markdown(f"""
                    <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                        ✅ <strong>저장 완료!</strong><br>{save_path}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 파일 크기 표시
                    import os
                    file_size = os.path.getsize(save_path) / 1024
                    st.info(f"📊 파일 크기: {file_size:.2f} KB")
                    
                except Exception as e:
                    st.error(f"❌ 저장 실패: {str(e)}")
            
            st.caption(f"💡 선택한 토픽: {len(selected_topics)}개\n문서: {len(filtered_df):,}개")
        
        # Excel 저장
        with col2:
            st.write("**💾 Excel 저장**")
            
            default_path_excel = str(Path.home() / "Desktop" / f"bertopic_result_selected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            
            save_path_excel = st.text_input(
                "저장 경로 (Excel)",
                value=default_path_excel,
                help="Excel 파일을 저장할 경로를 입력하세요",
                key="excel_path"
            )
            
            if st.button("💾 Excel로 저장", key="save_excel", use_container_width=True):
                try:
                    with pd.ExcelWriter(save_path_excel, engine='openpyxl') as writer:
                        filtered_df.to_excel(writer, index=False, sheet_name='선택한토픽')
                        keywords_df.to_excel(writer, index=False, sheet_name='전체토픽키워드')
                        
                        # 선택한 토픽 정보 시트 추가
                        selected_info = topic_info_df[topic_info_df['Topic ID'].isin(selected_topics)]
                        selected_info.to_excel(writer, index=False, sheet_name='선택한토픽정보')
                    
                    st.markdown(f"""
                    <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                        ✅ <strong>저장 완료!</strong><br>{save_path_excel}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    import os
                    file_size = os.path.getsize(save_path_excel) / 1024
                    st.info(f"📊 파일 크기: {file_size:.2f} KB")
                    
                except Exception as e:
                    st.error(f"❌ 저장 실패: {str(e)}")
            
            st.caption("💡 3개 시트 포함\n(선택한토픽, 전체토픽키워드, 선택한토픽정보)")
        
        # 메타데이터 저장
        with col3:
            st.write("**💾 메타데이터 저장**")
            
            default_path_json = str(Path.home() / "Desktop" / f"bertopic_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            save_path_json = st.text_input(
                "저장 경로 (JSON)",
                value=default_path_json,
                help="메타데이터 JSON 파일을 저장할 경로를 입력하세요",
                key="json_path"
            )
            
            if st.button("💾 JSON으로 저장", key="save_json", use_container_width=True):
                try:
                    metadata = {
                        'n_topics': n_topics,
                        'selected_topics': [int(t) for t in selected_topics],
                        'filtered_documents': len(filtered_df),
                        'outlier_count': int(outlier_count),
                        'outlier_percentage': float(outlier_pct),
                        'total_documents': len(topics),
                        'parameters': {
                            'embedding_model': embedding_model_key,
                            'n_components': n_components,
                            'n_neighbors': n_neighbors,
                            'min_dist': min_dist,
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'topic_mode': topic_mode,
                            'nr_topics': nr_topics if topic_mode == '수동' else 'auto',
                            'max_features': max_features,
                            'max_df': max_df,
                            'ngram_range': f"(1, {ngram_max})"
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    with open(save_path_json, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
                    st.markdown(f"""
                    <div style="background-color: #F0F2F6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                        ✅ <strong>저장 완료!</strong><br>{save_path_json}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    import os
                    file_size = os.path.getsize(save_path_json) / 1024
                    st.info(f"📊 파일 크기: {file_size:.2f} KB")
                    
                except Exception as e:
                    st.error(f"❌ 저장 실패: {str(e)}")

if __name__ == "__main__":
    main()