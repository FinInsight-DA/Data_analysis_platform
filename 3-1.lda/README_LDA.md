# LDA 토픽 모델링 설치 가이드

## 📋 개요
LDA (Latent Dirichlet Allocation) 토픽 모델링을 위한 설치 가이드입니다.

---

## 🖥️ 시스템 요구사항

- **Python**: 3.8 이상 (권장: 3.10 또는 3.11)
- **메모리**: 최소 8GB RAM (16GB 권장)
- **Java**: OpenJDK 11 이상 (KoNLPy 한국어 형태소 분석용)

---

## 📦 설치 방법

### 1. Conda 환경 생성

```bash
# 환경 생성
conda create -n lda python=3.11

# 환경 활성화
conda activate lda
```

### 2. 패키지 설치

#### Windows / Linux
```bash
pip install -r requirements_lda.txt
```

#### macOS (Apple Silicon: M1/M2/M3/M4)
```bash
pip install -r requirements_lda_mac.txt
```

### 3. Java 설치 (필수)

#### macOS
```bash
# Homebrew로 Java 설치
brew install openjdk@11

# 환경변수 설정 (~/.zshrc에 추가)
echo 'export JAVA_HOME=/opt/homebrew/opt/openjdk@11' >> ~/.zshrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.zshrc

# 적용
source ~/.zshrc
```

#### Windows
1. [Oracle JDK 다운로드](https://www.oracle.com/java/technologies/downloads/)
2. 설치 후 시스템 환경변수에 JAVA_HOME 추가

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install openjdk-11-jdk
```

### 4. 설치 확인

```bash
# Java 확인
java -version

# Python 패키지 확인
python -c "from konlpy.tag import Okt; print('✅ KoNLPy OK')"
python -c "from gensim.models import LdaModel; print('✅ Gensim OK')"
python -c "import streamlit; print('✅ Streamlit OK')"
```

---

## 🚀 실행 방법

### Streamlit 웹 앱
```bash
streamlit run lda_app_improved.py
```
브라우저에서 `http://localhost:8501` 자동 실행

### 로컬 Python 스크립트
```bash
# 파일 상단에서 설정 수정
# INPUT_CSV = './data/your_data.csv'

python lda_local.py
```

---

## 📂 필요한 파일

```
lda_project/
├── requirements_lda.txt          # 일반 환경용
├── requirements_lda_mac.txt      # macOS용
├── README_LDA.md                 # 이 파일
│
├── lda_app_improved.py           # Streamlit 앱
├── lda_local.py                  # 로컬 스크립트
│
├── data/                         # 데이터 폴더
│   └── your_data.csv            # sentence 컬럼 필수
│
├── LDA_results/                  # 결과 (자동 생성)
└── LDA_cache/                    # 캐시 (자동 생성)
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

### lda_local.py 설정 (파일 상단)

```python
# 입력 파일
INPUT_CSV = './data/your_data.csv'

# 학습할 토픽 개수
TOPIC_NUMBERS = [5, 10, 15, 20]

# LDA 하이퍼파라미터
PASSES = 5              # 전체 코퍼스 반복 횟수
ITERATIONS = 50         # 각 문서당 반복 횟수
ALPHA = 'auto'          # 문서-토픽 분포
ETA = 'auto'            # 토픽-단어 분포

# Dictionary 필터링
NO_BELOW = 5            # 최소 문서 빈도
NO_ABOVE = 0.5          # 최대 문서 비율
KEEP_N = 1000           # 최대 단어 수

# 전처리
MIN_NOUN_LENGTH = 2     # 최소 명사 길이
```

---

## ❗ 문제 해결

### 1. KoNLPy 오류
```
JPype Error: Java gateway process exited
```

**해결:**
```bash
# Java 설치 확인
java -version

# JAVA_HOME 확인 (macOS/Linux)
echo $JAVA_HOME

# 환경변수 재설정
export JAVA_HOME=/opt/homebrew/opt/openjdk@11  # macOS
```

### 2. 메모리 부족
```
MemoryError: Unable to allocate...
```

**해결:**
- 토픽 개수 줄이기: `TOPIC_NUMBERS = [5, 10]`
- Dictionary 크기 줄이기: `KEEP_N = 500`
- 데이터 샘플링 사용

### 3. Coherence 계산 느림 (macOS)
```
FileNotFoundError in multiprocessing
```

**해결:**
- 이미 코드에 `processes=1` 적용됨
- 속도는 느리지만 안정적

### 4. 형태소 분석 느림
```
형태소 분석 중... 1시간 이상 소요
```

**해결:**
- 캐시 사용 (두 번째 실행부터 빠름)
- `use_cache=True` 확인
- 캐시 위치: `./LDA_cache/`

---

## 📈 성능 최적화

### 빠른 테스트 설정
```python
TOPIC_NUMBERS = [5, 10]
PASSES = 3
ITERATIONS = 30
KEEP_N = 500
```

### 고품질 설정
```python
TOPIC_NUMBERS = [10, 15, 20, 25, 30]
PASSES = 10
ITERATIONS = 100
KEEP_N = 2000
```

---

## 💾 결과 파일

### 자동 생성되는 파일
- `lda_N_topics_YYYYMMDD_HHMMSS.csv` - 결과 데이터
- `lda_N_topics_YYYYMMDD_HHMMSS.xlsx` - 키워드 포함 Excel
- `lda_model_N_topics.model` - 학습된 모델
- `lda_N_topics_metadata_YYYYMMDD_HHMMSS.json` - 메타데이터
- `lda_dictionary.dict` - Dictionary

---

## 📚 주요 패키지 설명

| 패키지 | 용도 |
|--------|------|
| gensim | LDA 모델 학습 |
| konlpy | 한국어 형태소 분석 |
| streamlit | 웹 인터페이스 |
| plotly | 인터랙티브 시각화 |
| pandas | 데이터 처리 |

---

## 🔗 참고 자료

- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [KoNLPy Documentation](https://konlpy.org/)
- [LDA 논문](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)

---

## 📞 지원

문제 발생 시:
1. Python 버전: `python --version`
2. Java 버전: `java -version`
3. 패키지 목록: `pip list`
4. 에러 메시지 전체 복사
