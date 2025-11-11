# -*- coding: utf-8 -*-
"""
HBM 프로젝트 - H/V 라벨링 자동화
"""

from google.colab import drive
drive.mount('/content/drive')

import json
import pandas as pd
from typing import Tuple
from io import TextIOWrapper

# ============================================================================
# 설정
# ============================================================================
INPUT_CSV = '/content/drive/MyDrive/sampled_10k_raw.csv'
OUTPUT_CSV = '/content/drive/MyDrive/sampled_10k_hv_labeled_2.csv'
TERM_DB_PATH = '/content/drive/MyDrive/term_db.json'

LABEL_DESCRIPTIONS = {
    'H': '수평적 통합',
    'V': '수직적 통합'
}

LABEL_TYPE_MAPPING = {
    'H': 'horizontal',
    'V': 'vertical'
}

# ============================================================================
# TERM_DB 로드 함수
# ============================================================================
def load_term_db(file_input):
    """TERM_DB 로드 함수 (JSON + TXT + PY 지원)"""
    TERM_DB = {}
    filename = getattr(file_input, "name", str(file_input)).lower()

    try:
        if filename.endswith(".json"):
            if isinstance(file_input, str):
                with open(file_input, "r", encoding="utf-8") as f:
                    TERM_DB = json.load(f)
            else:
                raw = file_input.read()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                TERM_DB = json.loads(raw)

        elif filename.endswith(".txt"):
            if isinstance(file_input, str):
                with open(file_input, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            else:
                if isinstance(file_input, TextIOWrapper):
                    lines = file_input.readlines()
                else:
                    lines = file_input.read().decode("utf-8").splitlines()

            current_label = None
            current_category = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("#"):
                    current_label = line[1:]
                    if current_label not in TERM_DB:
                        TERM_DB[current_label] = {}
                    continue

                if line.startswith("@"):
                    current_category = line[1:]
                    if current_label and current_category not in TERM_DB[current_label]:
                        TERM_DB[current_label][current_category] = []
                    continue

                if current_label and current_category:
                    TERM_DB[current_label][current_category].append(line)

        elif filename.endswith(".py"):
            import importlib.util
            spec = importlib.util.spec_from_file_location("term_db_module", file_input)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            TERM_DB = module.term_db

        else:
            raise ValueError("지원하지 않는 파일 형식입니다.")

    except Exception as e:
        print(f"[ERROR] TERM_DB 로드 실패: {e}")

    return TERM_DB


# ============================================================================
# 라벨링 함수
# ============================================================================
def detect_label_in_text(text: str, TERM_DB: dict, label_priority: list, default_label: str, min_matches: int = 1) -> Tuple[str, str, str, int]:
    """라벨 자동 감지"""
    if pd.isna(text) or not text:
        return (default_label, '공통', 'Unknown', 0)

    text = str(text).strip()
    label_matches = {label: [] for label in TERM_DB.keys()}

    for label_type, categories in TERM_DB.items():
        for category, terms in categories.items():
            for term in terms:
                if term in text:
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


def map_to_standard_labels(df):
    """표준 라벨 매핑"""
    df['HV_type'] = df['label'].map(LABEL_TYPE_MAPPING)
    return df


# ============================================================================
# 메인 실행
# ============================================================================
def main():
    # TERM_DB 로드
    TERM_DB = load_term_db(TERM_DB_PATH)

    # TERM_DB에서 자동 추출
    label_priority = ['V', 'H'] if 'V' in TERM_DB and 'H' in TERM_DB else list(TERM_DB.keys())
    default_label = label_priority[-1] if label_priority else 'H'

    print(f"라벨 기준: {'/'.join(TERM_DB.keys())}")

    # 데이터 로딩
    df = pd.read_csv(INPUT_CSV)
    print(f"데이터 로드: {len(df):,}개\n")

    # sentence 생성
    df['sentence'] = df['title'].fillna('') + ' ' + df['content'].fillna('')

    # 라벨링 실행
    results = df['sentence'].apply(lambda x: detect_label_in_text(x, TERM_DB, label_priority, default_label, min_matches=1))
    df['label'] = results.apply(lambda x: x[0])
    df['aspect_category'] = results.apply(lambda x: x[1])
    df['aspect_term'] = results.apply(lambda x: x[2])
    df['match_count'] = results.apply(lambda x: x[3])
    df = map_to_standard_labels(df)

    # 결과 출력
    print("[라벨링 결과]")

    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        desc = LABEL_DESCRIPTIONS.get(label, label)
        print(f"{label} ({desc}): {count:,}개 ({count/len(df)*100:.1f}%)")

    if 'company' in df.columns:
        print("\n[회사별 분포]")
        company_dist = pd.crosstab(df['company'], df['label'])
        print(company_dist)

    unknown_count = (df['aspect_term'] == 'Unknown').sum()
    print(f"\nUnknown: {unknown_count:,}개 ({unknown_count/len(df)*100:.1f}%)")

    # Unknown 제외
    df_original_len = len(df)
    df = df[df['match_count'] >= 1].copy()
    print(f"Unknown 제외: {df_original_len:,}개 -> {len(df):,}개\n")

    # 컬럼 순서 정리
    output_cols = ['title', 'content', 'sentence', 'company', 'inp_date',
                   'label', 'HV_type', 'match_count',
                   'aspect_category', 'aspect_term']
    output_cols = [col for col in output_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in output_cols]
    df = df[output_cols + other_cols]

    # 저장
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"결과 저장 완료: {OUTPUT_CSV}")
    return df


if __name__ == "__main__":
    main()
