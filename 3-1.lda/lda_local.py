# -*- coding: utf-8 -*-
"""
LDA 토픽 모델링 자동화 (로컬 버전)
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
from konlpy.tag import Okt
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

# ============================================================================
# 설정
# ============================================================================
INPUT_CSV = '/Users/song/Desktop/workspace/fin/hv_labeled.csv'  # 입력 CSV 파일 경로
OUTPUT_DIR = './LDA_results'         # 결과 저장 디렉토리
CACHE_DIR = './LDA_cache'            # 캐시 디렉토리

# 학습할 토픽 개수 리스트
TOPIC_NUMBERS = [5, 10, 15, 20]

# LDA 하이퍼파라미터
PASSES = 5
ITERATIONS = 50
ALPHA = 'auto'  # 'auto', 'symmetric', 'asymmetric' 또는 숫자값
ETA = 'auto'    # 'auto', 'symmetric' 또는 숫자값

# Dictionary 필터링 파라미터
NO_BELOW = 5      # 최소 문서 빈도
NO_ABOVE = 0.5    # 최대 문서 비율
KEEP_N = 1000     # 최대 단어 수

# 전처리 파라미터
MIN_NOUN_LENGTH = 2  # 최소 명사 길이

# 불용어
STOP_WORDS = {
    '은', '는', '이', '가', '을', '를', '의', '와', '과', '도',
    '에', '로', '에서', '부터', '까지',
    '하다', '있다', '되다', '같다', '없다',
    '것', '수', '등', '개', '명', '년', '월', '일',
    '업계', '기업', '회사', '업체', '관계자',
    '올해', '내년', '작년', '이번', '지난해', '최근'
}

# 디렉토리 생성
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LDA 클래스
# ============================================================================
class LDATopicModeling:
    """LDA 토픽 모델링 자동화"""
    
    def __init__(self, df, stop_words, min_noun_length=2, verbose=True):
        self.df = df
        self.stop_words = stop_words
        self.min_noun_length = min_noun_length
        self.verbose = verbose
        self.processed_sentences = None
        self.dictionary = None
        self.corpus = None
        self.models = {}
        self.topics_dict = {}
        self.coherence_scores = {}
        self.perplexity_scores = {}
    
    def preprocess(self, use_cache=True):
        """형태소 분석 (캐시 활용)"""
        # 캐시 파일명 생성
        data_hash = hashlib.md5(
            (self.df['sentence'].str.cat() + str(self.min_noun_length)).encode()
        ).hexdigest()[:8]
        cache_file = f"{CACHE_DIR}/processed_{data_hash}.pkl"
        
        # 캐시 로드
        if use_cache and os.path.exists(cache_file):
            if self.verbose:
                print("📦 캐시된 전처리 결과 로드 중...")
            with open(cache_file, 'rb') as f:
                self.processed_sentences = pickle.load(f)
            if self.verbose:
                print(f"✅ 형태소 분석 완료 (캐시): {len(self.processed_sentences):,}개 문장")
            return
        
        # 형태소 분석
        if self.verbose:
            print("\n📝 형태소 분석 시작...")
        
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
        
        self.processed_sentences = []
        for text in tqdm(self.df['sentence'], desc="형태소 분석", disable=not self.verbose):
            self.processed_sentences.append(clean_text(text))
        
        self.processed_sentences = [s for s in self.processed_sentences if len(s) > 0]
        
        # 캐시 저장
        with open(cache_file, 'wb') as f:
            pickle.dump(self.processed_sentences, f)
        
        if self.verbose:
            print(f"✅ 형태소 분석 완료: {len(self.processed_sentences):,}개 문장")
    
    def create_dict_corpus(self, no_below, no_above, keep_n):
        """Dictionary & Corpus 생성"""
        if self.verbose:
            print("\n📚 Dictionary & Corpus 생성 중...")
        
        self.dictionary = corpora.Dictionary(self.processed_sentences)
        original_size = len(self.dictionary)
        
        self.dictionary.filter_extremes(
            no_below=no_below,
            no_above=no_above,
            keep_n=keep_n
        )
        
        self.corpus = [self.dictionary.doc2bow(text) for text in self.processed_sentences]
        
        if self.verbose:
            print(f"✅ Dictionary & Corpus 생성 완료")
            print(f"   - 원본 단어 수: {original_size:,}")
            print(f"   - 필터링 후: {len(self.dictionary):,}")
            print(f"   - Corpus 크기: {len(self.corpus):,}")
        
        return original_size, len(self.dictionary)
    
    def train_lda(self, n_topics, passes, iterations, alpha, eta):
        """LDA 학습"""
        if self.verbose:
            print(f"\n🚀 LDA 학습 시작 ({n_topics}개 토픽)...")
            print(f"   - Passes: {passes}")
            print(f"   - Iterations: {iterations}")
            print(f"   - Alpha: {alpha}")
            print(f"   - Eta: {eta}")
        
        start_time = time.time()
        
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
        if self.verbose:
            print("   📊 Coherence 계산 중...")
        
        coherence_model = CoherenceModel(
            model=model,
            texts=self.processed_sentences,
            dictionary=self.dictionary,
            coherence='c_v',
            processes=1  # macOS multiprocessing 에러 방지
        )
        coherence = coherence_model.get_coherence()
        self.coherence_scores[n_topics] = coherence
        
        # Perplexity 계산
        perplexity = model.log_perplexity(self.corpus)
        self.perplexity_scores[n_topics] = perplexity
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"✅ 학습 완료! (소요 시간: {elapsed:.1f}초)")
            print(f"   - Coherence: {coherence:.4f}")
            print(f"   - Perplexity: {perplexity:.2f}")
            print(f"   - 할당된 문서: {(self.topics_dict[n_topics] != -1).sum():,}개")
        
        return coherence, perplexity
    
    def get_result_df(self, n_topics):
        """결과 데이터프레임 생성"""
        topics = self.topics_dict[n_topics]
        result_df = self.df.iloc[:len(topics)].copy()
        result_df['lda_topic'] = topics
        result_df = result_df[result_df['lda_topic'] != -1]
        return result_df
    
    def save_results(self, n_topics, selected_topics=None):
        """결과 저장"""
        if self.verbose:
            print(f"\n💾 결과 저장 중 ({n_topics}개 토픽)...")
        
        model = self.models[n_topics]
        result_df = self.get_result_df(n_topics)
        
        # 선택한 토픽만 필터링
        if selected_topics is not None:
            result_df = result_df[result_df['lda_topic'].isin(selected_topics)].copy()
            suffix = f"_selected_{len(selected_topics)}topics"
        else:
            suffix = ""
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV 저장
        output_csv = f"{OUTPUT_DIR}/lda_{n_topics}_topics{suffix}_{timestamp}.csv"
        result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        # Excel 저장 (키워드 포함)
        output_excel = f"{OUTPUT_DIR}/lda_{n_topics}_topics{suffix}_{timestamp}.xlsx"
        
        # 토픽별 키워드 테이블 생성
        keywords_data = []
        for topic_id in range(n_topics):
            words = model.show_topic(topic_id, topn=10)
            keywords = ', '.join([f"{word}({prob:.3f})" for word, prob in words[:5]])
            keywords_data.append({
                '토픽': f"Topic {topic_id}",
                '주요 키워드': keywords
            })
        keywords_df = pd.DataFrame(keywords_data)
        
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='선택한토픽' if selected_topics else '전체토픽')
            keywords_df.to_excel(writer, index=False, sheet_name='토픽키워드')
        
        # 모델 저장
        model_path = f"{OUTPUT_DIR}/lda_model_{n_topics}_topics.model"
        model.save(model_path)
        
        # 메타데이터 저장
        metadata = {
            'n_topics': n_topics,
            'total_documents': len(result_df),
            'selected_topics': selected_topics if selected_topics else list(range(n_topics)),
            'coherence_score': float(self.coherence_scores[n_topics]),
            'perplexity_score': float(self.perplexity_scores[n_topics]),
            'parameters': {
                'passes': PASSES,
                'iterations': ITERATIONS,
                'alpha': str(ALPHA),
                'eta': str(ETA),
                'no_below': NO_BELOW,
                'no_above': NO_ABOVE,
                'keep_n': KEEP_N
            },
            'timestamp': timestamp
        }
        
        meta_path = f"{OUTPUT_DIR}/lda_{n_topics}_topics{suffix}_metadata_{timestamp}.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"✅ 저장 완료!")
            print(f"   - CSV: {output_csv}")
            print(f"   - Excel: {output_excel}")
            print(f"   - 모델: {model_path}")
            print(f"   - 메타데이터: {meta_path}")
        
        return result_df
    
    def print_topics(self, n_topics, top_n=10):
        """토픽별 키워드 출력"""
        print(f"\n{'='*80}")
        print(f"📋 토픽별 주요 키워드 ({n_topics}개 토픽, Top {top_n})")
        print(f"{'='*80}")
        
        model = self.models[n_topics]
        topics = self.topics_dict[n_topics]
        
        for topic_id in range(n_topics):
            count = (topics == topic_id).sum()
            words = model.show_topic(topic_id, topn=top_n)
            keywords = ', '.join([f"{word}({prob:.3f})" for word, prob in words])
            
            print(f"\n[Topic {topic_id}] ({count:,}개 문서)")
            print(f"  {keywords}")
        
        print(f"\n{'='*80}")
    
    def print_summary(self):
        """전체 결과 요약"""
        print(f"\n{'='*80}")
        print("📊 전체 학습 결과 요약")
        print(f"{'='*80}")
        
        # 요약 테이블
        summary_data = []
        for n_topics in sorted(self.models.keys()):
            topics = self.topics_dict[n_topics]
            coherence = self.coherence_scores[n_topics]
            perplexity = self.perplexity_scores[n_topics]
            doc_count = (topics != -1).sum()
            
            summary_data.append({
                '토픽 개수': n_topics,
                '문서 수': f"{doc_count:,}",
                'Coherence': f"{coherence:.4f}",
                'Perplexity': f"{perplexity:.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        print(f"\n{'='*80}")

# ============================================================================
# 엘보우 포인트 계산
# ============================================================================
def calculate_elbow_point(scores_dict, maximize=True):
    """엘보우 포인트 계산"""
    if len(scores_dict) < 3:
        return None
    
    topics = np.array(sorted(scores_dict.keys()))
    scores = np.array([scores_dict[k] for k in topics])
    
    if not maximize:
        scores = -scores
    
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    topics_norm = (topics - topics.min()) / (topics.max() - topics.min() + 1e-10)
    
    m = (scores_norm[-1] - scores_norm[0]) / (topics_norm[-1] - topics_norm[0] + 1e-10)
    b = scores_norm[0]
    
    distances = np.abs(scores_norm - (m * topics_norm + b)) / np.sqrt(m**2 + 1)
    elbow_idx = np.argmax(distances)
    elbow_point = topics[elbow_idx]
    
    return int(elbow_point)

# ============================================================================
# 메인 함수
# ============================================================================
def main():
    """메인 실행 함수"""
    start_time = time.time()
    
    print("="*80)
    print("🎯 LDA 토픽 모델링 시작")
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
    print(f"   토픽 개수: {TOPIC_NUMBERS}")
    print(f"   Passes: {PASSES}, Iterations: {ITERATIONS}")
    print(f"   Alpha: {ALPHA}, Eta: {ETA}")
    print(f"   Dictionary 필터링: no_below={NO_BELOW}, no_above={NO_ABOVE}, keep_n={KEEP_N}")
    print(f"   최소 명사 길이: {MIN_NOUN_LENGTH}")
    print(f"   불용어: {len(STOP_WORDS)}개")
    
    # ========================================
    # 3. LDA 실행
    # ========================================
    print("\n🚀 3. LDA 토픽 모델링 실행")
    
    lda = LDATopicModeling(df, STOP_WORDS, MIN_NOUN_LENGTH, verbose=True)
    
    # 전처리
    lda.preprocess(use_cache=True)
    
    # Dictionary & Corpus 생성
    lda.create_dict_corpus(NO_BELOW, NO_ABOVE, KEEP_N)
    
    # LDA 학습 (여러 토픽 개수)
    print(f"\n{'='*80}")
    print("📚 LDA 모델 학습")
    print(f"{'='*80}")
    
    for i, n_topics in enumerate(TOPIC_NUMBERS, 1):
        print(f"\n[{i}/{len(TOPIC_NUMBERS)}] {n_topics}개 토픽 학습")
        lda.train_lda(n_topics, PASSES, ITERATIONS, ALPHA, ETA)
    
    # ========================================
    # 4. 결과 요약
    # ========================================
    lda.print_summary()
    
    # 엘보우 포인트 계산
    coherence_elbow = calculate_elbow_point(lda.coherence_scores, maximize=True)
    perplexity_elbow = calculate_elbow_point(lda.perplexity_scores, maximize=False)
    
    print("\n🎯 추천 토픽 개수 (엘보우 포인트)")
    if coherence_elbow and perplexity_elbow:
        if coherence_elbow == perplexity_elbow:
            print(f"   ⭐ {coherence_elbow}개 토픽 (Coherence와 Perplexity 모두 최적)")
        else:
            print(f"   - Coherence 기준: {coherence_elbow}개 토픽")
            print(f"   - Perplexity 기준: {perplexity_elbow}개 토픽")
    
    # ========================================
    # 5. 결과 저장
    # ========================================
    print(f"\n{'='*80}")
    print("💾 4. 결과 저장")
    print(f"{'='*80}")
    
    # 가장 좋은 토픽 개수 선택 (Coherence 기준)
    best_n_topics = max(lda.coherence_scores.keys(),
                        key=lambda k: lda.coherence_scores[k])
    
    print(f"\n📌 저장할 토픽 개수: {best_n_topics}개 (Coherence 최고)")
    
    # 토픽별 키워드 출력
    lda.print_topics(best_n_topics, top_n=10)
    
    # 토픽 선택 UI
    print(f"\n{'='*80}")
    print("🎯 저장할 토픽 선택")
    print(f"{'='*80}")
    
    result_df = lda.get_result_df(best_n_topics)
    model = lda.models[best_n_topics]
    
    # 토픽별 정보 출력
    print(f"\n토픽별 문서 수:")
    for topic_id in range(best_n_topics):
        count = (result_df['lda_topic'] == topic_id).sum()
        pct = count / len(result_df) * 100
        words = model.show_topic(topic_id, topn=3)
        keywords = ', '.join([word for word, _ in words])
        print(f"  Topic {topic_id}: {count:,}개 ({pct:.1f}%) - {keywords}")
    
    # 사용자 입력
    print(f"\n💡 저장할 토픽을 선택하세요:")
    print("   1. 전체 토픽 저장 (Enter 또는 'all' 입력)")
    print("   2. 특정 토픽만 저장 (예: 0,2,5 또는 0-5 또는 0-5,9,11)")
    
    user_input = input("\n선택: ").strip()
    
    selected_topics = None
    
    if user_input == '' or user_input.lower() == 'all':
        # 전체 저장
        print(f"✅ 전체 {best_n_topics}개 토픽 저장")
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
            selected_topics = [t for t in selected_topics if 0 <= t < best_n_topics]
            
            if not selected_topics:
                print("⚠️ 유효한 토픽이 없습니다. 전체 토픽을 저장합니다.")
                selected_topics = None
            else:
                print(f"✅ {len(selected_topics)}개 토픽 선택: {selected_topics}")
                
                # 선택한 토픽 정보 출력
                selected_count = result_df[result_df['lda_topic'].isin(selected_topics)].shape[0]
                selected_pct = selected_count / len(result_df) * 100
                print(f"   - 선택한 토픽의 문서 수: {selected_count:,}개 ({selected_pct:.1f}%)")
                
        except Exception as e:
            print(f"⚠️ 입력 형식 오류: {e}")
            print("   전체 토픽을 저장합니다.")
            selected_topics = None
    
    # 결과 저장
    print(f"\n💾 결과 저장 중...")
    lda.save_results(best_n_topics, selected_topics=selected_topics)
    
    # Dictionary 저장
    dict_path = f"{OUTPUT_DIR}/lda_dictionary.dict"
    lda.dictionary.save(dict_path)
    print(f"✅ Dictionary 저장: {dict_path}")
    
    # ========================================
    # 6. 완료
    # ========================================
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("✅ LDA 토픽 모델링 완료!")
    print(f"{'='*80}")
    print(f"총 실행 시간: {total_time/60:.1f}분")
    print(f"결과 저장 위치: {OUTPUT_DIR}/")
    print(f"{'='*80}\n")

# ============================================================================
# 실행
# ============================================================================
if __name__ == "__main__":
    main()
