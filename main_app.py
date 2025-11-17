from __future__ import annotations
import streamlit as st
from pathlib import Path
from streamlit_option_menu import option_menu  # ← 추가
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent

# ===========================================================================
# 동적 임포트: 파일 경로에서 모듈을 불러와 특정 함수를 실행
# ===========================================================================
def run_page(pyfile: Path, func_name: str = "main") -> None:
    if not pyfile.exists():
        st.error(f"파일을 찾을 수 없습니다: {pyfile}")
        return
    spec = importlib.util.spec_from_file_location(pyfile.stem, pyfile)
    if spec is None or spec.loader is None:
        st.error(f"모듈 스펙 로드 실패: {pyfile}")
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[pyfile.stem] = module
    spec.loader.exec_module(module)
    if not hasattr(module, func_name):
        st.error(f"`{pyfile.name}`에 `{func_name}()` 함수가 없습니다. "
                 f"해당 파일의 UI 코드를 `{func_name}()`로 감싸 주세요.")
        return
    getattr(module, func_name)()

# ===========================================================================
# 페이지 라우팅 테이블 (좌측 사이드바 메뉴 ↔ 실제 파일 경로 매핑) 각 파일에는 반드시 `main()` 함수가 있어야 함
# ===========================================================================
PAGES = {
    " 크롤링": ROOT / "1.crawling" / "app_crawling.py",
    " H/V 라벨링": ROOT / "2.data_preparation" / "hv_labeling_app.py",
    " LDA": ROOT / "3-1.lda" / "lda_app.py",
    " BERTopic": ROOT / "3-2.bertopic" / "bertopic_app.py",
    " 감성 분석": ROOT / "4.sentiment" / "sentiment_app.py",  # ← 여기만 변경
    " OLS 회귀": ROOT / "5.ols" / "streamlit_app.py",
}


# ===========================================================================
# 앱 설정
# ===========================================================================
st.set_page_config(
    page_title="Data Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# CSS 설정
# ===========================================================================
st.markdown("""
<style>
/* 사이드바 폭 살짝 넓게 */
[data-testid="stSidebar"] { width: 260px; }
[data-testid="stSidebar"] div[role="radiogroup"] label { padding: 6px 8px; } /* fallback용 */
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# 사이드바 네비 (아이콘 메뉴)
# ===========================================================================
with st.sidebar:
    st.markdown("### 📚 메뉴")
    choice = option_menu(
        menu_title=None,
        options=list(PAGES.keys()),
        icons=[
            "cloud-download",   # ① 크롤링
            "tags",             # ② H/V 라벨링
            "list-task",        # ③ LDA
            "diagram-3",        # ④ BERTopic
            "emoji-smile",      # ⑤ 감성 분석
            "graph-up-arrow",   # ⑥ OLS 회귀
        ],
        menu_icon="list",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important"},
            "icon": {"font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "padding": "8px 10px",
                "border-radius": "8px",
                "color": "#334155",
            },
            "nav-link-selected": {
                "background-color": "#E8F0FE",
                "color": "#1d4ed8",
            },
        },
    )

target = PAGES[choice]

# 공통 안내(최초 진입시만)
with st.sidebar.expander("ℹ️ 사용 가이드", expanded=False):
    st.write(

        "- 환경 패키지는 폴더별 `requirements*.txt` 참고.\n"

    )

# 페이지 실행
run_page(target, func_name="main")
