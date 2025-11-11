# -*- coding: utf-8 -*-
"""
BERTopic 토픽 모델링 자동화 (로컬 버전)
"""

import time
import os
import pickle
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np
import re
from tqdm import tqdm

# BERTopic & Related
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# ============================================================================
# 설정
# ============================================================================
INPUT_CSV = '/Users/song/Desktop/workspace/fin/hv_labeled.csv'  # 입력 CSV 파일 경로
OUTPUT_DIR = './BERTopic_results'         # 결과 저장 디렉토리
CACHE_DIR = './BERTopic_cache'            # 캐시 디렉토리

# 임베딩 모델
EMBEDDING_MODEL = 'jhgan/ko-sroberta-multitask'  # 'jhgan/ko-sroberta-multitask', 'paraphrase-multilingual-MiniLM-L12-v2'

# UMAP 파라미터
N_COMPONENTS = 5
N_NEIGHBORS = 15
MIN_DIST = 0.0

# HDBSCAN 파라미터
MIN_CLUSTER_SIZE = 50
MIN_SAMPLES = 10

# 토픽 개수 설정
TOPIC_MODE = 'auto'  # 'auto' 또는 숫자 (예: 20)

# Vectorizer 파라미터
MAX_FEATURES = 200
MAX_DF = 0.8
NGRAM_MAX = 1

# 샘플링 설정
USE_SAMPLING = False  # 대용량 데이터일 경우 True
SAMPLE_SIZE = 50000   # 샘플링 크기

# 디렉토리 생성
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 헬퍼 함수
# ============================================================================
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
# BERTopic 클래스
# ============================================================================
class BERTopicModeling:
    """BERTopic 토픽 모델링 자동화"""
    
    def __init__(self, df, verbose=True):
        self.df = df
        self.verbose = verbose
        self.embeddings = None
        self.topic_model = None
        self.topics = None
        self.df_result = None
    
    def create_embeddings(self, use_cache=True):
        """임베딩 생성 (캐시 활용)"""
        texts = self.df['sentence'].tolist()
        
        # 캐시 파일명 생성
        data_hash = hashlib.md5(
            (self.df['sentence'].str.cat() + EMBEDDING_MODEL).encode()
        ).hexdigest()[:8]
        cache_file = f"{CACHE_DIR}/embeddings_{data_hash}.pkl"
        
        # 캐시 로드
        if use_cache and os.path.exists(cache_file):
            if self.verbose:
                print("📦 캐시된 임베딩 로드 중...")
            with open(cache_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            if self.verbose:
                print(f"✅ 임베딩 로드 완료 (캐시): {self.embeddings.shape}")
            return texts
        
        # 임베딩 생성
        if self.verbose:
            print("\n🔤 임베딩 생성 시작...")
            print(f"   모델: {EMBEDDING_MODEL}")
        
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        # 배치 단위로 임베딩 생성
        batch_size = 32
        embeddings_list = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="임베딩 생성", disable=not self.verbose):
            batch = texts[i:i+batch_size]
            batch_embeddings = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings_list.append(batch_embeddings)
        
        self.embeddings = np.vstack(embeddings_list)
        
        # 캐시 저장
        with open(cache_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        if self.verbose:
            print(f"✅ 임베딩 생성 완료: {self.embeddings.shape}")
        
        return texts
    
    def train_bertopic(self, texts, sample_texts=None, sample_embeddings=None):
        """BERTopic 학습"""
        if self.verbose:
            print("\n🚀 BERTopic 모델 학습")
            print(f"   - UMAP: n_components={N_COMPONENTS}, n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST}")
            print(f"   - HDBSCAN: min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}")
            print(f"   - 토픽 개수: {TOPIC_MODE}")
        
        start_time = time.time()
        
        # UMAP
        umap_model = UMAP(
            n_components=N_COMPONENTS,
            n_neighbors=N_NEIGHBORS,
            min_dist=MIN_DIST,
            metric='cosine',
            random_state=42
        )
        
        # HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            cluster_selection_method='eom',
            metric='euclidean',
            prediction_data=False
        )
        
        # Vectorizer
        vectorizer_model = CountVectorizer(
            tokenizer=smart_tokenizer,
            max_features=MAX_FEATURES,
            max_df=MAX_DF,
            ngram_range=(1, NGRAM_MAX)
        )
        
        # BERTopic
        self.topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=TOPIC_MODE if TOPIC_MODE == 'auto' else int(TOPIC_MODE),
            min_topic_size=max(10, int(len(texts) * 0.001)),
            calculate_probabilities=False,
            verbose=False
        )
        
        # 학습
        if sample_texts is not None:
            if self.verbose:
                print(f"   📊 샘플로 학습 중... ({len(sample_texts):,}개)")
            self.topics, _ = self.topic_model.fit_transform(sample_texts, sample_embeddings)
        else:
            if self.verbose:
                print(f"   📊 전체 데이터로 학습 중... ({len(texts):,}개)")
            self.topics, _ = self.topic_model.fit_transform(texts, self.embeddings)
        
        self.topics = np.array(self.topics)
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"✅ 학습 완료! (소요 시간: {elapsed:.1f}초)")
    
    def predict_all(self, texts):
        """전체 데이터 예측 (샘플링 사용 시)"""
        if self.verbose:
            print("\n📊 전체 데이터 예측 중...")
        
        self.topics, _ = self.topic_model.transform(texts, self.embeddings)
        self.topics = np.array(self.topics)
        
        if self.verbose:
            print("✅ 예측 완료!")
    
    def create_result_df(self):
        """결과 데이터프레임 생성"""
        self.df_result = self.df.copy()
        self.df_result['bertopic_topic'] = self.topics
        self.df_result['outlier'] = (self.topics == -1).astype(int)
        
        # 통계
        outlier_count = (self.topics == -1).sum()
        outlier_pct = outlier_count / len(self.topics) * 100
        unique_topics = sorted([t for t in set(self.topics) if t != -1])
        n_topics = len(unique_topics)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("📊 학습 결과")
            print(f"{'='*80}")
            print(f"   - 전체 문서: {len(self.topics):,}개")
            print(f"   - 토픽 수: {n_topics}개")
            print(f"   - Outlier: {outlier_count:,}개 ({outlier_pct:.1f}%)")
            
            if outlier_pct < 25:
                status = "우수"
            elif outlier_pct < 35:
                status = "양호"
            elif outlier_pct < 45:
                status = "보통"
            else:
                status = "개선필요"
            print(f"   - 평가: {status}")
            print(f"{'='*80}")
        
        return unique_topics
    
    def print_topics(self, unique_topics, top_n=10):
        """토픽별 키워드 출력"""
        print(f"\n{'='*80}")
        print(f"📋 토픽별 주요 키워드 (Top {top_n})")
        print(f"{'='*80}")
        
        for topic_id in unique_topics[:20]:  # 상위 20개만
            count = (self.topics == topic_id).sum()
            pct = count / len(self.topics) * 100
            words = self.topic_model.get_topic(topic_id)
            
            if words:
                keywords = ', '.join([f"{w[0]}({w[1]:.3f})" for w in words[:top_n]])
                print(f"\n[Topic {topic_id}] ({count:,}개 문서, {pct:.1f}%)")
                print(f"  {keywords}")
        
        print(f"\n{'='*80}")
    
    def save_results(self, unique_topics, selected_topics=None):
        """결과 저장"""
        if self.verbose:
            print(f"\n💾 결과 저장 중...")
        
        # 선택한 토픽만 필터링
        if selected_topics is not None:
            result_df = self.df_result[self.df_result['bertopic_topic'].isin(selected_topics)].copy()
            suffix = f"_selected_{len(selected_topics)}topics"
        else:
            result_df = self.df_result.copy()
            suffix = ""
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV 저장
        output_csv = f"{OUTPUT_DIR}/bertopic_result{suffix}_{timestamp}.csv"
        result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        # Excel 저장 (키워드 포함)
        output_excel = f"{OUTPUT_DIR}/bertopic_result{suffix}_{timestamp}.xlsx"
        
        # 토픽별 키워드 테이블 생성
        keywords_data = []
        for topic_id in unique_topics:
            count = (self.topics == topic_id).sum()
            pct = count / len(self.topics) * 100
            words = self.topic_model.get_topic(topic_id)
            
            if words:
                keywords = ', '.join([f"{w[0]}({w[1]:.3f})" for w in words[:5]])
                keywords_data.append({
                    '토픽': f"Topic {topic_id}",
                    '문서 수': f"{count:,}",
                    '비율': f"{pct:.1f}%",
                    '주요 키워드': keywords
                })
        
        keywords_df = pd.DataFrame(keywords_data)
        
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='선택한토픽' if selected_topics else '전체토픽')
            keywords_df.to_excel(writer, index=False, sheet_name='토픽키워드')
        
        # 모델 저장
        model_path = f"{OUTPUT_DIR}/bertopic_model.pkl"
        self.topic_model.save(model_path, serialization='pickle')
        
        # 메타데이터 저장
        outlier_count = (self.topics == -1).sum()
        outlier_pct = outlier_count / len(self.topics) * 100
        
        metadata = {
            'n_topics': int(len(unique_topics)),
            'total_documents': int(len(result_df)),
            'selected_topics': [int(t) for t in (selected_topics if selected_topics else unique_topics)],
            'outlier_count': int(outlier_count),
            'outlier_percentage': float(outlier_pct),
            'parameters': {
                'embedding_model': EMBEDDING_MODEL,
                'n_components': int(N_COMPONENTS),
                'n_neighbors': int(N_NEIGHBORS),
                'min_dist': float(MIN_DIST),
                'min_cluster_size': int(MIN_CLUSTER_SIZE),
                'min_samples': int(MIN_SAMPLES),
                'topic_mode': str(TOPIC_MODE),
                'max_features': int(MAX_FEATURES),
                'max_df': float(MAX_DF),
                'ngram_range': f"(1, {NGRAM_MAX})"
            },
            'timestamp': timestamp
        }
        
        meta_path = f"{OUTPUT_DIR}/bertopic_metadata{suffix}_{timestamp}.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"✅ 저장 완료!")
            print(f"   - CSV: {output_csv}")
            print(f"   - Excel: {output_excel}")
            print(f"   - 모델: {model_path}")
            print(f"   - 메타데이터: {meta_path}")

# ============================================================================
# 메인 함수
# ============================================================================
def main():
    """메인 실행 함수"""
    start_time = time.time()
    
    print("="*80)
    print("🎯 BERTopic 토픽 모델링 시작")
    print("="*80)
    
    # ========================================
    # 1. 데이터 로드
    # ========================================
    print("\n📁 1. 데이터 로드")
    print(f"   입력 파일: {INPUT_CSV}")
    
    if not os.path.exists(INPUT_CSV):
        print(f"\n❌ 파일을 찾을 수 없습니다: {INPUT_CSV}")
        print("힌트: INPUT_CSV 경로를 수정하세요.")
        return
    
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"✅ 데이터 로드 완료: {len(df):,}개 문서")
        print(f"   컬럼: {list(df.columns)}")
        
        if 'sentence' not in df.columns:
            print("\n❌ 'sentence' 컬럼이 없습니다.")
            return
            
    except Exception as e:
        print(f"\n❌ 파일 로드 실패: {e}")
        return
    
    # ========================================
    # 2. 파라미터 확인
    # ========================================
    print("\n⚙️ 2. 파라미터 설정")
    print(f"   임베딩 모델: {EMBEDDING_MODEL}")
    print(f"   UMAP: n_components={N_COMPONENTS}, n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST}")
    print(f"   HDBSCAN: min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}")
    print(f"   토픽 개수: {TOPIC_MODE}")
    print(f"   Vectorizer: max_features={MAX_FEATURES}, max_df={MAX_DF}, ngram=(1,{NGRAM_MAX})")
    print(f"   샘플링: {'사용' if USE_SAMPLING else '미사용'}" + (f" ({SAMPLE_SIZE:,}개)" if USE_SAMPLING else ""))
    
    # ========================================
    # 3. BERTopic 실행
    # ========================================
    print("\n🚀 3. BERTopic 토픽 모델링 실행")
    
    bertopic = BERTopicModeling(df, verbose=True)
    
    # 임베딩 생성
    texts = bertopic.create_embeddings(use_cache=True)
    
    # 샘플링 (옵션)
    if USE_SAMPLING and SAMPLE_SIZE < len(texts):
        print(f"\n📊 샘플링 사용: {SAMPLE_SIZE:,}개로 학습")
        np.random.seed(42)
        sample_indices = np.random.choice(len(bertopic.embeddings), SAMPLE_SIZE, replace=False)
        sample_embeddings = bertopic.embeddings[sample_indices]
        sample_texts = [texts[i] for i in sample_indices]
        
        # 학습
        bertopic.train_bertopic(texts, sample_texts, sample_embeddings)
        
        # 전체 데이터 예측
        bertopic.predict_all(texts)
    else:
        print(f"\n📊 전체 데이터 사용: {len(texts):,}개")
        bertopic.train_bertopic(texts)
    
    # ========================================
    # 4. 결과 생성
    # ========================================
    unique_topics = bertopic.create_result_df()
    
    # 토픽별 키워드 출력
    bertopic.print_topics(unique_topics, top_n=10)
    
    # ========================================
    # 5. 토픽 선택 및 저장
    # ========================================
    print(f"\n{'='*80}")
    print("🎯 저장할 토픽 선택")
    print(f"{'='*80}")
    
    # Outlier 포함 여부
    print(f"\n💡 Outlier (-1) 토픽을 포함하시겠습니까?")
    include_outlier = input("   포함 (y/n) [기본: n]: ").strip().lower()
    
    available_topics = unique_topics.copy()
    if include_outlier == 'y':
        available_topics = [-1] + available_topics
        print("✅ Outlier 포함")
    else:
        print("✅ Outlier 제외")
    
    # 토픽별 정보 출력
    print(f"\n토픽별 문서 수:")
    for topic_id in available_topics[:30]:  # 최대 30개만 표시
        if topic_id == -1:
            count = (bertopic.topics == -1).sum()
            pct = count / len(bertopic.topics) * 100
            print(f"  Topic {topic_id} (Outlier): {count:,}개 ({pct:.1f}%)")
        else:
            count = (bertopic.topics == topic_id).sum()
            pct = count / len(bertopic.topics) * 100
            words = bertopic.topic_model.get_topic(topic_id)
            if words:
                keywords = ', '.join([w[0] for w in words[:3]])
                print(f"  Topic {topic_id}: {count:,}개 ({pct:.1f}%) - {keywords}")
    
    if len(available_topics) > 30:
        print(f"  ... (총 {len(available_topics)}개 토픽)")
    
    # 사용자 입력
    print(f"\n💡 저장할 토픽을 선택하세요:")
    print("   1. 전체 토픽 저장 (Enter 또는 'all' 입력)")
    print("   2. 특정 토픽만 저장 (예: 0,2,5 또는 0-5 또는 0-5,9,11)")
    
    user_input = input("\n선택: ").strip()
    
    selected_topics = None
    
    if user_input == '' or user_input.lower() == 'all':
        # 전체 저장
        print(f"✅ 전체 {len(available_topics)}개 토픽 저장")
        selected_topics = None
    else:
        # 특정 토픽 파싱
        try:
            selected_topics = []
            
            # 쉼표로 분리
            parts = user_input.split(',')
            
            for part in parts:
                part = part.strip()
                
                # 범위 입력 처리 (예: 0-5)
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected_topics.extend(range(start, end + 1))
                else:
                    # 단일 숫자
                    selected_topics.append(int(part))
            
            # 중복 제거 및 정렬
            selected_topics = sorted(list(set(selected_topics)))
            
            # 유효성 검사
            selected_topics = [t for t in selected_topics if t in available_topics]
            
            if not selected_topics:
                print("⚠️ 유효한 토픽이 없습니다. 전체 토픽을 저장합니다.")
                selected_topics = None
            else:
                print(f"✅ {len(selected_topics)}개 토픽 선택: {selected_topics}")
                
                # 선택한 토픽 정보 출력
                selected_count = bertopic.df_result[bertopic.df_result['bertopic_topic'].isin(selected_topics)].shape[0]
                selected_pct = selected_count / len(bertopic.df_result) * 100
                print(f"   - 선택한 토픽의 문서 수: {selected_count:,}개 ({selected_pct:.1f}%)")
                
        except Exception as e:
            print(f"⚠️ 입력 형식 오류: {e}")
            print("   전체 토픽을 저장합니다.")
            selected_topics = None
    
    # 결과 저장
    bertopic.save_results(unique_topics, selected_topics=selected_topics)
    
    # ========================================
    # 6. 완료
    # ========================================
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("✅ BERTopic 토픽 모델링 완료!")
    print(f"{'='*80}")
    print(f"총 실행 시간: {total_time/60:.1f}분")
    print(f"결과 저장 위치: {OUTPUT_DIR}/")
    print(f"{'='*80}\n")

# ============================================================================
# 실행
# ============================================================================
if __name__ == "__main__":
    main()
