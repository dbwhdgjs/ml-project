"""
4주차 Streamlit 대시보드 — 원자재 위험 알림 데모

실행:
    pip install streamlit
    streamlit run week4/code/streamlit_app.py

UI 원칙: ML 전문용어 대신 일상 언어 (정밀도→"맞힌 비율", 재현율→"놓치지 않은 비율")
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager

for fp in [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]:
    if os.path.exists(fp):
        try:
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = "AppleGothic"
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJ_DIR, "data", "commodity_vix_gpr_data.xlsx")
RESULTS_PATH = os.path.join(BASE_DIR, "reports", "results.json")

st.set_page_config(page_title="원자재 위험 알림", page_icon="📈", layout="wide")

# ============================================================
# 데이터 로드
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH, index_col=0)
    return df

@st.cache_data
def load_results():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)

df = load_data()
R = load_results()

# ============================================================
# 상단
# ============================================================
st.title("📈 원자재 위험 알림 서비스 (데모)")
st.caption(f"데이터: {df.index.min().date()} ~ {df.index.max().date()} · 총 {len(df):,}일")

with st.expander("이 서비스는 무엇인가요?"):
    st.markdown("""
- **위험 알림**: 내일 가격이 크게 흔들릴 가능성이 높으면 미리 알려줍니다. *놓치지 않는 것*이 우선이라
  조금 헛알람이 있어도 알림을 보냅니다.
- **매매 추천**: *맞힐 때만* 추천합니다. 추천이 나오면 그만큼 확신이 있다는 뜻입니다.
- 한 가지 모델로 두 가지 모드를 운영합니다 (확률 임계값을 다르게 줍니다).
""")

# ============================================================
# 탭: 자재별
# ============================================================
TARGETS = list(R["targets"].keys())
tabs = st.tabs([f"🔹 {t}" for t in TARGETS])

for tab, name in zip(tabs, TARGETS):
    with tab:
        tr = R["targets"][name]
        col = {"원유": "원유(WTI)", "금": "금", "천연가스": "천연가스", "은": "은"}[name]

        # 사이드에 임계값 조절
        c1, c2, c3 = st.columns([1, 1, 1])
        alert_thr = c1.slider(f"위험알림 민감도 (낮을수록 더 자주 울림)",
                              0.10, 0.60, R["config"]["alert_threshold"], 0.05,
                              key=f"alert_{name}")
        timing_thr = c2.slider(f"매매추천 확신도 (높을수록 신중)",
                               0.50, 0.90, R["config"]["timing_threshold"], 0.05,
                               key=f"timing_{name}")
        c3.metric("이상변동 기준", f"±{tr['anom_thr_pct']}%")

        # 상단 메트릭
        st.markdown(f"### 📊 {name} 모델 (5-Fold 평균)")
        m = tr["fold_avg"]
        a, b, c, d, e = st.columns(5)
        a.metric("전체 정확도", f"{m['accuracy']*100:.1f}%")
        b.metric("맞힌 비율 (정밀도)", f"{m['precision']*100:.1f}%")
        c.metric("놓치지 않은 비율 (재현율)", f"{m['recall']*100:.1f}%")
        d.metric("종합 F1", f"{m['f1']:.3f}")
        e.metric("AUC", f"{m['roc_auc']:.3f}")

        # 투트랙 모드별 성능
        st.markdown("### 🚨 모드별 성능 (최근 fold 기준)")
        mode_cols = st.columns(2)
        with mode_cols[0]:
            st.markdown("**위험알림 모드** — 놓치지 않기 우선")
            a = tr["alert"]
            st.write(f"- 놓치지 않은 비율: **{a['recall']*100:.1f}%**")
            st.write(f"- 알림이 실제로 맞은 비율: **{a['precision']*100:.1f}%**")
            st.write(f"- 임계값: {a['threshold']}")
        with mode_cols[1]:
            st.markdown("**매매추천 모드** — 확신할 때만")
            t = tr["timing"]
            st.write(f"- 추천이 실제로 맞은 비율: **{t['precision']*100:.1f}%**")
            st.write(f"- 놓치지 않은 비율: **{t['recall']*100:.1f}%**")
            st.write(f"- 임계값: {t['threshold']}")

        # 최근 가격 차트
        st.markdown(f"### 📉 {name} 가격 (최근 6개월)")
        recent = df[col].tail(180)
        ret = recent.pct_change() * 100
        anom_mask = ret.abs() >= tr["anom_thr_pct"]

        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(recent.index, recent.values, color="#1f77b4", lw=1.5)
        if anom_mask.any():
            ax.scatter(recent.index[anom_mask], recent[anom_mask],
                       color="#d62728", s=35, zorder=5, label=f"이상변동(±{tr['anom_thr_pct']}%↑)")
        ax.set_xlabel("날짜"); ax.set_ylabel("가격")
        ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# 하단: 4종 비교
# ============================================================
st.markdown("---")
st.markdown("### 🧭 4종 원자재 한눈에 비교")
rows = []
for name in TARGETS:
    m = R["targets"][name]["fold_avg"]
    rows.append({
        "원자재": name,
        "기준(±%)": R["targets"][name]["anom_thr_pct"],
        "정확도": f"{m['accuracy']*100:.1f}%",
        "정밀도": f"{m['precision']*100:.1f}%",
        "재현율": f"{m['recall']*100:.1f}%",
        "F1": f"{m['f1']:.3f}",
        "AUC": f"{m['roc_auc']:.3f}",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.caption("ⓒ 머신러닝 프로젝트 4주차 · 임태후 · 유종헌")
