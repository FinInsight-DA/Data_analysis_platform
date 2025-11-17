# 4.sentiment/sentiment_app.py

import os
import sys
import streamlit as st

# 현재 폴더를 모듈 검색 경로에 추가
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

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

# 기존 두 앱의 main()을 그대로 가져와서 탭 안에서 재사용
from labeling_app import main as labeling_main
from module_app import main as model_main


def main():
    st.markdown(
        '<div class="main-header">감성 분석</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["① 감성 라벨링", "② 모델 비교 · ABSA"])

    # 탭 1: 라벨링
    with tab1:
        labeling_main()

    # 탭 2: 모델 비교 + ABSA
    with tab2:
        model_main()


if __name__ == "__main__":
    main()