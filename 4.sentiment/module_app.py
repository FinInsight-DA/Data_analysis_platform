# module_app.py (최종 통합 자동진행률 버전)
import os
import tempfile
import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sentiment_module import run_selected_models
from sentiment_absa import ABSAModel

# =========================
# 페이지 설정 
# =========================
st.set_page_config(
    page_title="통합 감성 분석 대시보드",
    page_icon="📊",
    layout="wide"
)

st.title("📊 통합 감성 분석 대시보드")
st.write("데이터 업로드 → 모델 비교 학습 → ABSA 감성 분석 → 결과 CSV 다운로드까지 한 페이지에서 순차적으로 진행합니다.")

# =========================
# GPU 상태 표시
# =========================
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    st.success(f"⚡ GPU 사용 중: {gpu_name}")
else:
    st.warning("💻 GPU 미사용 - CPU로 실행됩니다.")

# =========================
# 초기화 버튼
# =========================
if st.button("🧹 초기화"):
    for key in ["final_result", "absa_result", "absa_model", "uploaded_file_path"]:
        if key in st.session_state:
            del st.session_state[key]
    st.info("세션이 초기화되었습니다. CSV를 다시 업로드해주세요.")

# =========================
# CSV 업로드
# =========================
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
st.markdown("**⚠️ CSV에는 반드시 `sentence`, `sentiment` 컬럼이 있어야 합니다.**")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 데이터 미리보기")
    st.dataframe(df.head(10))

    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False, encoding="utf-8-sig")
        temp_path = tmp.name
        st.session_state.uploaded_file_path = temp_path

    st.markdown("---")

    # =========================
    # 모델 선택
    # =========================
    st.subheader("🔍 모델 선택 및 학습")
    col1, col2 = st.columns(2)
    with col1:
        selected_ml = st.multiselect("전통 ML 모델 선택", ["RF", "SVM", "NB"], default=[])
    with col2:
        selected_dl = st.multiselect("딥러닝 모델 선택", ["KoBERT", "KoELECTRA", "KoRoBERTa", "BERT"], default=[])

    # =========================
    # 모델 학습 (자동 진행률)
    # =========================
    if st.button("🚀 모델 학습 시작"):
        if "final_result" in st.session_state:
            del st.session_state.final_result

        # 선택된 모든 모델 통합
        selected_models = selected_ml + selected_dl
        if not selected_models:
            st.warning("⚠️ 최소 한 개 이상의 모델을 선택해주세요.")
            st.stop()

        # 진행률 초기화
        progress_bar = st.progress(0)
        progress_text = st.empty()
        total_models = len(selected_models)
        current_index = 0
        results = []

        for model_name in selected_models:
            with st.spinner(f"🧠 [{model_name}] 학습 중..."):
                if model_name in ["RF", "SVM", "NB"]:
                    df_result = run_selected_models(selected_ml=[model_name], input_csv=temp_path)
                else:
                    df_result = run_selected_models(selected_dl=[model_name], input_csv=temp_path)
                results.append(df_result)

            # 모델 1개 완료 시 진행률 갱신
            current_index += 1
            pct = int((current_index / total_models) * 100)
            progress_bar.progress(pct / 100)
            progress_text.text(f"전체 진행률: {pct}%")

        # 완료 후 처리
        progress_bar.progress(1.0)
        progress_text.text("✅ 전체 진행 완료 (100%)")

        # 결과 저장
        st.session_state.final_result = pd.concat(results, ignore_index=True)
        st.success("🎉 모든 모델 학습이 완료되었습니다!")

    # =========================
    # 학습 결과 출력
    # =========================
    with st.container():
        st.subheader("📈 모델 성능 비교")
        if "final_result" in st.session_state:
            st.dataframe(st.session_state.final_result)

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=st.session_state.final_result, x='Model', y='Accuracy', ax=ax)
            ax.set_ylim(0, 1)
            st.pyplot(fig, clear_figure=True)

    st.markdown("---")

    # =========================
    # ABSA 감성 분석
    # =========================
    st.subheader("💬 ABSA 감성 분석")

    user_friendly_models = ["KoBERT", "KoELECTRA", "KoRoBERTa", "BERT"]
    model_mapping = {
        "KoBERT": "skt/kobert-base-v1",
        "KoELECTRA": "monologg/koelectra-base-v3-discriminator",
        "KoRoBERTa": "klue/roberta-base",
        "BERT": "bert-base-uncased"
    }

    model_choice_user = st.selectbox("사용할 ABSA 모델 선택", user_friendly_models, index=1)
    model_choice_path = model_mapping[model_choice_user]

    # ABSA 분석 버튼
    if st.button("🚀 ABSA 분석 시작"):
        for key in ["absa_result", "absa_model"]:
            if key in st.session_state:
                del st.session_state[key]

        with st.spinner(f"{model_choice_user} 모델 로딩 중..."):
            st.session_state.absa_model = ABSAModel(model_choice_path)
            model = st.session_state.absa_model

        progress_bar = st.progress(0)
        progress_text = st.empty()

        sentiments, confidences = [], []
        total = len(df)

        for i, sentence in enumerate(df['sentence'], start=1):
            label, conf = model.analyze_sentiment(sentence)
            sentiments.append(label)
            confidences.append(conf)

            if i % max(1, total // 100) == 0 or i == total:
                progress_bar.progress(i / total)
                progress_text.text(f"감성 분석 중: {i}/{total}")

        df['pred_label'] = sentiments
        df['confidence'] = confidences
        st.session_state.absa_result = df

    # =========================
    # ABSA 결과 출력 + CSV 다운로드
    # =========================
    if "absa_result" in st.session_state:
        st.success("🎉 ABSA 감성 분석 완료!")
        st.write("💡 감성 분석 결과 미리보기")
        st.dataframe(st.session_state.absa_result.head(10))

        download_file_name = f"{model_choice_user}_results.csv"
        csv_bytes = st.session_state.absa_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 ABSA 결과 CSV 다운로드",
            data=csv_bytes,
            file_name=download_file_name,
            mime="text/csv",
            key="download_absa"
        )

else:
    st.info("CSV 파일을 업로드하면 모델 성능 비교 및 ABSA 분석을 시작할 수 있습니다.")
