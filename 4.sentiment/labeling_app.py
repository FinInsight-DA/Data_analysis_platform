import os
import sys
import tempfile

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import streamlit as st
import pandas as pd
import torch
from sentiment_labeling import run_sentiment_labeling


def main():
    st.subheader("📌 감성 라벨링")

    # GPU/CPU 상태 표시
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"⚡ GPU 사용 중: {gpu_name}")
    else:
        st.warning("💻 GPU 미사용 - CPU로 실행됩니다.")

    # 세션 상태 초기화
    for key in ["df_result", "output_path", "labeling_in_progress", "labeling_done"]:
        if key not in st.session_state:
            if key in ["df_result", "output_path"]:
                st.session_state[key] = None
            else:
                st.session_state[key] = False

    # 파일 업로드
    uploaded_data = st.file_uploader("라벨링할 CSV 업로드", type=["csv"], key="label_csv")
    uploaded_dict = st.file_uploader("감성사전 JSON 업로드", type=["json"], key="label_json")

    if not uploaded_data or not uploaded_dict:
        st.info("CSV와 JSON 파일을 모두 업로드하면 감성 라벨링을 시작할 수 있습니다.")
    else:
        st.success("✅ 두 파일 모두 업로드 완료!")

        df_preview = pd.read_csv(uploaded_data)
        st.markdown("---")
        st.subheader("데이터 미리보기")
        st.dataframe(df_preview.head(10), use_container_width=True)

        if st.button("🚀 감성 라벨링 실행", key="run_labeling"):
            st.session_state.labeling_in_progress = True
            st.session_state.labeling_done = False
            st.rerun()

    # 실제 라벨링 실행 구간
    if st.session_state.labeling_in_progress and uploaded_data and uploaded_dict:
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_warning_text = st.empty()
        metric_rule = st.empty()
        metric_model_train = st.empty()
        metric_neutral = st.empty()

        # 업로드 파일을 임시 경로에 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_data, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_dict:
            tmp_data.write(uploaded_data.getbuffer())
            tmp_dict.write(uploaded_dict.getbuffer())
            tmp_data_path, tmp_dict_path = tmp_data.name, tmp_dict.name

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name

        # 진행률 콜백
        def progress_callback(stage, current, total):
            if total == 0:
                total = 1

            if stage == "rule":
                status_text.text("🧩 규칙 기반 라벨링 처리 중...")
                time_warning_text.text("⏳ 시간이 다소 소요될 수 있습니다. 잠시만 기다려주세요.")
                metric_rule.metric(label="🧩 규칙 기반 라벨링", value=f"{current} / {total}")
                progress_bar.progress(min(int(current / total * 33), 33) / 100)
            elif stage == "model_train":
                status_text.text("⚙️ 긍정·부정 학습 처리 중...")
                time_warning_text.text("⏳ 시간이 다소 소요될 수 있습니다. 잠시만 기다려주세요.")
                metric_model_train.metric(label="⚙️ 긍정·부정 학습", value=f"{current} / {total}")
                progress_bar.progress((33 + min(int(current / total * 33), 33)) / 100)
            elif stage == "neutral_labeling":
                status_text.text("📝 중립 문장 라벨링 처리 중...")
                time_warning_text.text("⏳ 시간이 다소 소요될 수 있습니다. 잠시만 기다려주세요.")
                metric_neutral.metric(label="📝 중립 문장 라벨링", value=f"{current} / {total}")
                progress_bar.progress((66 + min(int(current / total * 34), 34)) / 100)

        # 라벨링 실행
        df_result = run_sentiment_labeling(
            tmp_data_path,
            tmp_dict_path,
            progress_callback=progress_callback,
        )

        # 임시 파일 삭제
        os.remove(tmp_data_path)
        os.remove(tmp_dict_path)

        df_result.to_csv(output_path, index=False)
        st.session_state.df_result = df_result
        st.session_state.output_path = output_path

        st.session_state.labeling_in_progress = False
        st.session_state.labeling_done = True
        st.rerun()

    # 완료 메시지 + 결과 미리보기 + 다운로드
    if st.session_state.labeling_done and st.session_state.df_result is not None:
        st.success("✅ 전체 감성 라벨링 완료!")

        st.markdown("---")
        st.subheader("라벨링 결과 미리보기")
        st.dataframe(st.session_state.df_result.head(10), use_container_width=True)

        with open(st.session_state.output_path, "rb") as f:
            st.download_button(
                label="💾 라벨링 결과 CSV 다운로드",
                data=f,
                file_name="sentiment_labeling.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
