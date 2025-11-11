# BERTopic 토픽 모델링 설치 가이드

## 📋 개요
BERTopic (BERT-based Topic Modeling) 토픽 모델링을 위한 설치 가이드입니다.

---

## 🖥️ 시스템 요구사항

- **Python**: 3.8 이상 (권장: 3.10 또는 3.11)
- **메모리**: 최소 16GB RAM (32GB 권장)
- **GPU**: 선택사항 (CUDA 지원 시 더 빠름)
- **저장공간**: 5GB 이상 (임베딩 모델 캐시용)

---

## 📦 설치 방법

### 1. Conda 환경 생성

```bash
# 환경 생성
conda create -n bertopic python=3.11

# 환경 활성화
conda activate bertopic
```

### 2. 패키지 설치

#### Windows / Linux
```bash
pip install -r requirements_bertopic.txt
```

#### macOS (Apple Silicon: M1/M2/M3/M4)
```bash
pip install -r requirements_bertopic_mac.txt
```

### 3. 설치 확인

```bash
# Python 패키지 확인
python -c "from bertopic import BERTopic; print('✅ BERTopic OK')"
python -c "from sentence_transformers import SentenceTransformer; print('✅ Sentence Transformers OK')"
python -c "import streamlit; print('✅ Streamlit OK')"
```

---

## 🚀 실행 방법

### Streamlit 웹 앱
```bash
streamlit run bertopic_app_improved.py
```
브라우저에서 `http://localhost:8501` 자동 실행

### 로컬 Python 스크립트
```bash
# 파일 상단에서 설정 수정
# INPUT_CSV = './data/your_data.csv'

python bertopic_local.py
```

---

## 📂 필요한 파일

```
bertopic_project/
├── requirements_bertopic.txt      # 일반 환경용
├── requirements_bertopic_mac.txt  # macOS용
├── README_BERTopic.md            # 이 파일
│
├── bertopic_app_improved.py      # Streamlit 앱
├── bertopic_local.py             # 로컬 스크립트
│
├── data/                         # 데이터 폴더
│   └── your_data.csv            # sentence 컬럼 필수
│
├── BERTopic_results/             # 결과 (자동 생성)
└── BERTopic_cache/               # 캐시 (자동 생성)
```

---

## 📊 데이터 형식

CSV 파일에 **`sentence`** 컬럼 필수:

```csv
sentence,label,company
"HBM 메모리 가격이 상승했다",1,SK하이닉스
"반도체 시장이 회복세를 보이고 있다",1,삼성전자
```

---

## ⚙️ 주요 파라미터

### bertopic_local.py 설정 (파일 상단)

```python
# 입력 파일
INPUT_CSV = './data/your_data.csv'

# 임베딩 모델 (한국어)
EMBEDDING_MODEL = 'jhgan/ko-sroberta-multitask'
# 다른 옵션: 'paraphrase-multilingual-MiniLM-L12-v2'

# UMAP 파라미터 (차원 축소)
N_COMPONENTS = 5        # 차원 수
N_NEIGHBORS = 15        # 이웃 수
MIN_DIST = 0.0          # 최소 거리

# HDBSCAN 파라미터 (클러스터링)
MIN_CLUSTER_SIZE = 50   # 최소 클러스터 크기
MIN_SAMPLES = 10        # 최소 샘플 수

# 토픽 개수
TOPIC_MODE = 'auto'     # 'auto' 또는 숫자 (예: 20)

# Vectorizer
MAX_FEATURES = 200      # 최대 단어 수
MAX_DF = 0.8            # 최대 문서 빈도
NGRAM_MAX = 1           # N-gram 최대값

# 샘플링 (대용량 데이터)
USE_SAMPLING = False    # True로 변경하여 샘플링 사용
SAMPLE_SIZE = 50000     # 샘플 크기
```

---

## ❗ 문제 해결

### 1. 임베딩 모델 다운로드 느림
```
Downloading (…)88cf/.gitattributes: 100%
```

**해결:**
- 처음 실행 시 모델 다운로드 (1-2분 소요)
- 이후 자동 캐시 사용
- 위치: `~/.cache/huggingface/`
- 인터넷 연결 필수

### 2. 메모리 부족
```
RuntimeError: CUDA out of memory
또는
MemoryError: Unable to allocate
```

**해결:**
```python
# 샘플링 사용
USE_SAMPLING = True
SAMPLE_SIZE = 30000

# 클러스터 크기 증가
MIN_CLUSTER_SIZE = 100

# 배치 크기 감소 (코드 내)
batch_size = 16  # 기본 32에서 감소
```

### 3. UMAP/HDBSCAN 느림
```
학습에 1시간 이상 소요
```

**해결:**
- 샘플링 사용
- 파라미터 조정:
  ```python
  N_COMPONENTS = 3        # 5 → 3
  MIN_CLUSTER_SIZE = 100  # 50 → 100
  ```

### 4. Outlier가 너무 많음 (>50%)
```
Outlier: 6,543개 (65.4%)
```

**해결:**
```python
# 클러스터 크기 감소
MIN_CLUSTER_SIZE = 30   # 50 → 30
MIN_SAMPLES = 5         # 10 → 5

# 이웃 수 증가
N_NEIGHBORS = 25        # 15 → 25
```

### 5. macOS MPS 오류
```
RuntimeError: MPS backend out of memory
```

**해결:**
- CPU만 사용 (기본 설정)
- 배치 크기 감소
- 샘플링 사용

---

## 🎯 임베딩 모델 선택

### 한국어 특화
```python
# 추천 (성능 우수)
EMBEDDING_MODEL = 'jhgan/ko-sroberta-multitask'

# 대안 (빠른 속도)
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
```

### 다국어 지원
```python
EMBEDDING_MODEL = 'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
```

---

## 📈 성능 최적화

### 빠른 테스트 설정
```python
USE_SAMPLING = True
SAMPLE_SIZE = 10000
MIN_CLUSTER_SIZE = 100
N_COMPONENTS = 3
```

### 고품질 설정
```python
USE_SAMPLING = False
MIN_CLUSTER_SIZE = 30
MIN_SAMPLES = 5
N_COMPONENTS = 10
N_NEIGHBORS = 30
```

### 대용량 데이터 (100만 건 이상)
```python
USE_SAMPLING = True
SAMPLE_SIZE = 100000
MIN_CLUSTER_SIZE = 200
batch_size = 16  # 코드 내 수정
```

---

## 💾 결과 파일

### 자동 생성되는 파일
- `bertopic_result_YYYYMMDD_HHMMSS.csv` - 결과 데이터
- `bertopic_result_YYYYMMDD_HHMMSS.xlsx` - 키워드 포함 Excel
- `bertopic_model.pkl` - 학습된 모델
- `bertopic_metadata_YYYYMMDD_HHMMSS.json` - 메타데이터
- `embeddings_HASH.pkl` - 임베딩 캐시

---

## 📚 주요 패키지 설명

| 패키지 | 용도 |
|--------|------|
| bertopic | BERTopic 모델 |
| sentence-transformers | 문장 임베딩 생성 |
| umap-learn | 차원 축소 |
| hdbscan | 계층적 클러스터링 |
| streamlit | 웹 인터페이스 |
| plotly | 인터랙티브 시각화 |

---

## 🔍 Outlier 이해

### Outlier란?
- 토픽 -1로 할당된 문서
- 어떤 토픽에도 잘 맞지 않는 문서
- 이상치 또는 노이즈

### Outlier 비율 기준
- **우수**: < 25%
- **양호**: 25-35%
- **보통**: 35-45%
- **개선필요**: > 45%

### Outlier 줄이는 방법
1. `MIN_CLUSTER_SIZE` 감소
2. `MIN_SAMPLES` 감소
3. `N_NEIGHBORS` 증가
4. 데이터 전처리 개선

---

## 🆚 LDA vs BERTopic 비교

| 특징 | LDA | BERTopic |
|------|-----|----------|
| 속도 | 빠름 | 느림 |
| 메모리 | 적음 (8GB) | 많음 (16GB+) |
| 품질 | 보통 | 우수 |
| 한국어 | Okt 필요 | 임베딩 모델 |
| 토픽 수 | 수동 지정 | 자동 결정 |
| Outlier | 없음 | 있음 |

### 선택 가이드
- **빠른 분석 필요**: LDA
- **고품질 결과 필요**: BERTopic
- **대용량 데이터**: LDA (샘플링)
- **적은 데이터 (<1만)**: BERTopic

---

## 🔗 참고 자료

- [BERTopic Documentation](https://maartengr.github.io/BERTopic/)
- [Sentence Transformers](https://www.sbert.net/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)

---

## 📞 지원

문제 발생 시:
1. Python 버전: `python --version`
2. PyTorch 버전: `python -c "import torch; print(torch.__version__)"`
3. 패키지 목록: `pip list`
4. 메모리 사용량 확인
5. 에러 메시지 전체 복사
