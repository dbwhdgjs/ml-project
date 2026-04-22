import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="원자재 이상변동 감지", layout="centered", initial_sidebar_state="collapsed")

# 페이지 상태 초기화
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# === 투트랙 모드 시작 (제거 시 이 블록 + show_oil 내 투트랙 블록 + render_prediction의 mode_key 분기 삭제) ===
MODES = {
    'risk': {
        'label': '위험 경보', 'icon': '🛡', 'threshold': 0.3,
        'focus': '놓치지 않기 우선', 'color': '#F04452',
        'desc': '작은 위험 신호도 놓치지 않고 알려드립니다',
        'target': '변동성이 걱정되는 투자자',
    },
    'timing': {
        'label': '매수 타이밍', 'icon': '🎯', 'threshold': 0.7,
        'focus': '확실할 때만 추천', 'color': '#3182F6',
        'desc': '확신이 들 때만 매수·매도를 추천합니다',
        'target': '신중한 투자 결정이 필요한 기관',
    },
}
# === 투트랙 모드 끝 ===

# Toss 스타일 CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&display=swap');

* { font-family: 'Noto Sans KR', sans-serif; }

.stApp {
    background-color: #F5F6F8;
}

.main-header {
    text-align: center;
    padding: 30px 0 10px 0;
}
.main-header h1 {
    font-size: 24px;
    font-weight: 700;
    color: #191F28;
    margin: 0;
}
.main-header p {
    font-size: 14px;
    color: #8B95A1;
    margin: 5px 0 0 0;
}

.card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 28px;
    margin: 12px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.card-label {
    font-size: 14px;
    font-weight: 500;
    color: #8B95A1;
    margin-bottom: 8px;
}

.card-value {
    font-size: 40px;
    font-weight: 900;
    line-height: 1.2;
    margin-bottom: 4px;
}

.card-sub {
    font-size: 13px;
    color: #8B95A1;
    margin-top: 4px;
}

.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
}

.badge-danger {
    background: #FFF0F0;
    color: #F04452;
}
.badge-safe {
    background: #E8F7EF;
    color: #00C853;
}
.badge-warn {
    background: #FFF8E6;
    color: #FF9800;
}

.badge-up {
    background: #FFF0F0;
    color: #F04452;
}
.badge-down {
    background: #E8F3FF;
    color: #3182F6;
}

.prob-bar-bg {
    background: #F2F3F5;
    border-radius: 8px;
    height: 12px;
    margin: 16px 0 8px 0;
    position: relative;
}
.prob-bar {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}
.prob-threshold {
    position: absolute;
    top: -4px;
    bottom: -4px;
    width: 0;
    border-left: 2px dashed #4E5968;
    z-index: 2;
}
.prob-threshold-label {
    position: absolute;
    top: -20px;
    transform: translateX(-50%);
    font-size: 10px;
    font-weight: 700;
    color: #4E5968;
    white-space: nowrap;
    background: #FFFFFF;
    padding: 0 4px;
}

.divider {
    height: 1px;
    background: #F2F3F5;
    margin: 20px 0;
}

.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
}
.info-label {
    font-size: 14px;
    color: #8B95A1;
}
.info-value {
    font-size: 14px;
    font-weight: 700;
    color: #191F28;
}

.date-pill {
    display: inline-block;
    background: #F2F3F5;
    border-radius: 12px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
    color: #4E5968;
    margin-bottom: 16px;
}

.bottom-note {
    text-align: center;
    font-size: 12px;
    color: #B0B8C1;
    padding: 20px 0 40px 0;
}

.home-desc {
    font-size: 15px;
    color: #4E5968;
    line-height: 1.7;
    text-align: center;
    margin: 8px 0 24px 0;
}

.commodity-btn {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 24px;
    border-radius: 16px;
    margin: 10px 0;
    cursor: pointer;
    transition: all 0.2s ease;
}
.commodity-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.commodity-btn-active {
    background: #FFFFFF;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.commodity-btn-disabled {
    background: #F2F3F5;
    cursor: default;
    opacity: 0.6;
}
.commodity-btn-disabled:hover {
    transform: none;
    box-shadow: none;
}
.commodity-icon {
    font-size: 28px;
    margin-right: 16px;
}
.commodity-name {
    font-size: 16px;
    font-weight: 700;
    color: #191F28;
}
.commodity-name-disabled {
    font-size: 16px;
    font-weight: 700;
    color: #B0B8C1;
}
.commodity-status {
    font-size: 12px;
    color: #8B95A1;
}
.commodity-arrow {
    font-size: 18px;
    color: #B0B8C1;
}
.coming-soon-badge {
    display: inline-block;
    background: #F2F3F5;
    color: #B0B8C1;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 10px;
}

/* hide streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 900px !important; }
</style>
""", unsafe_allow_html=True)


# ========== 메인 페이지 ==========
def show_home():
    # 최신 데이터셋 요약 (파일 mtime + 마지막 날짜 조회)
    _base = os.path.dirname(os.path.abspath(__file__))
    latest_path = os.path.join(_base, '..', '..', 'data', 'commodity_vix_gpr_latest.xlsx')
    live_path = os.path.join(_base, '..', '..', 'data', 'commodity_vix_gpr_live.xlsx')
    try:
        lat_df = pd.read_excel(latest_path, index_col=0)
        data_rows = len(lat_df)
        data_period_start = lat_df.index[0].strftime('%Y.%m')
        data_period_end = lat_df.index[-1].strftime('%Y.%m.%d')
        live_df = pd.read_excel(live_path, index_col=0)
        gpr_last_str = live_df.index[-1].strftime('%Y.%m.%d')
        mtime = _file_mtime(latest_path)
        updated_at = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M') if mtime else '—'
    except Exception:
        data_rows, data_period_start, data_period_end = 0, '—', '—'
        gpr_last_str, updated_at = '—', '—'

    st.markdown(f"""
    <div class="main-header" style="padding: 50px 0 20px 0;">
        <h1 style="font-size: 28px;">원자재 가격 변동 알림</h1>
        <p>전쟁·정치 불안에 민감한 원자재, 미리 신호를 포착합니다</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <div class="home-desc">
            전쟁·정치 불안 지수와 시장 불안 지수를 활용해<br>
            원자재 가격의 <b>큰 폭 변동을 미리 감지</b>하고<br>
            <b>가격 흐름을 예측</b>하는 서비스입니다.
        </div>
        <div style="display:flex; justify-content:center; gap:24px; margin-top:8px;">
            <div style="text-align:center;">
                <div style="font-size:24px; font-weight:900; color:#3182F6;">{data_rows:,}</div>
                <div style="font-size:12px; color:#8B95A1;">분석 데이터(일)</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:24px; font-weight:900; color:#3182F6;">7</div>
                <div style="font-size:12px; color:#8B95A1;">분석 지표</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:24px; font-weight:900; color:#3182F6;">AI</div>
                <div style="font-size:12px; color:#8B95A1;">예측 엔진</div>
            </div>
        </div>
        <div style="margin-top:20px; padding-top:16px; border-top:1px solid #F2F3F5;
                    display:flex; justify-content:space-between; font-size:12px; color:#8B95A1;">
            <span>데이터 기간 {data_period_start} ~ {data_period_end}</span>
            <span>GPR {gpr_last_str} · 갱신 {updated_at}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:8px;">
        <div class="card-label" style="padding:0 4px; margin-bottom:4px;">원자재 선택</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🛢️  원유 (WTI)  —  분석 가능", key="btn_oil", use_container_width=True):
        st.session_state.page = 'oil'
        st.rerun()

    st.button("🥇  금 (Gold)  —  준비 중", key="btn_gold", use_container_width=True, disabled=True)
    st.button("🥈  은 (Silver)  —  준비 중", key="btn_silver", use_container_width=True, disabled=True)
    st.button("🔥  천연가스  —  준비 중", key="btn_gas", use_container_width=True, disabled=True)

    st.markdown(f"""
    <div class="bottom-note" style="padding-top:30px;">
        전쟁·정치 불안에 반응하는 원자재 가격 알림 서비스<br>
        {data_period_start} ~ {data_period_end} · AI 기반 예측 (교차검증 적용)<br><br>
        임태후 · 유종헌
    </div>
    """, unsafe_allow_html=True)


# ========== 원유 분석 페이지 ==========
def _file_mtime(path):
    """캐시 무효화용 파일 수정시각"""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@st.cache_data
def load_and_train(filepath, mtime):
    """mtime 인자는 파일이 바뀌면 캐시를 자동 무효화하기 위한 cache key"""
    _ = mtime  # 사용 X — cache invalidation 전용
    data = pd.read_excel(filepath, index_col=0)
    data['원유_수익률'] = data['원유(WTI)'].pct_change() * 100
    for col in ['금', '천연가스', '은']:
        data[f'{col}_수익률'] = data[col].pct_change() * 100
    data['이상변동'] = (data['원유_수익률'].abs() >= 2.0).astype(int)
    data['방향'] = (data['원유_수익률'] > 0).astype(int)
    data = data.dropna()

    features = ['VIX', 'GPR지수', 'GPR_실제행동', 'GPR_위협',
                '금_수익률', '천연가스_수익률', '은_수익률']

    X = data[features].values
    y_anom = data['이상변동'].values
    y_dir = data['방향'].values
    y_ret = data['원유_수익률'].values

    split = len(data) - 30
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:split])
    X_test = scaler.transform(X[split:])

    _xgb_common = dict(n_estimators=50, max_depth=4, learning_rate=0.1,
                       tree_method='hist', n_jobs=1, random_state=42, verbosity=0)

    m_anom = XGBClassifier(scale_pos_weight=1.8, eval_metric='logloss', **_xgb_common)
    m_anom.fit(X_train, y_anom[:split])

    m_dir = XGBClassifier(eval_metric='logloss', **_xgb_common)
    m_dir.fit(X_train, y_dir[:split])

    m_ret = XGBRegressor(**_xgb_common)
    m_ret.fit(X_train, y_ret[:split])

    prob_anom = m_anom.predict_proba(X_test)[:, 1]
    pred_dir = m_dir.predict(X_test)
    pred_ret = m_ret.predict(X_test)
    dates = data.index[split:]
    prices = data['원유(WTI)'].values[split:]

    return dates, prices, prob_anom, pred_dir, pred_ret


def render_prediction(label, tag, tag_color, data_date, prob, direction, ret, tomorrow,
                      threshold=0.5, mode_key=None):
    """예측 카드 렌더링.
    mode_key='risk' → 위험 경보 레이아웃 / 'timing' → 매수 타이밍 레이아웃 / None → 기존 3단계"""

    ret_sign = "+" if ret > 0 else ""

    # 데이터 기준 태그 (공통)
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin:20px 0 8px 0;">
        <span style="font-size:15px; font-weight:700; color:#191F28;">{label}</span>
        <span style="font-size:11px; font-weight:700; color:{tag_color}; background:{tag_color}18; padding:3px 10px; border-radius:8px;">{tag}</span>
    </div>
    """, unsafe_allow_html=True)

    # === 투트랙: 위험 경보 레이아웃 (제거 가능) ===
    if mode_key == 'risk':
        is_alert = prob >= threshold
        if is_alert:
            action, action_icon = '포지션 관리 권장', '⚠'
            action_color, action_bg = '#F04452', '#FFF0F0'
            action_sub = '큰 폭의 가격 변동이 예상됩니다'
        else:
            action, action_icon = '안정 구간', '✓'
            action_color, action_bg = '#00C853', '#E8F7EF'
            action_sub = '통상적인 변동 범위 내에 있습니다'

        prob_color = '#F04452' if is_alert else '#00C853'
        dir_icon = "↑" if direction == 1 else "↓"
        dir_text = "상승" if direction == 1 else "하락"
        dir_color = "#F04452" if direction == 1 else "#3182F6"
        ret_color = "#F04452" if ret > 0 else "#3182F6"

        # 액션 + 이상변동 확률 통합 카드
        st.markdown(f"""
        <div class="card" style="background:{action_bg}; border-left:4px solid {action_color};">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
                <span style="font-size:32px;">{action_icon}</span>
                <div>
                    <div style="font-size:22px; font-weight:900; color:{action_color};">{action}</div>
                    <div style="font-size:13px; color:#4E5968; margin-top:2px;">{action_sub}</div>
                </div>
            </div>
            <div class="divider"></div>
            <div class="card-label">급변동 가능성</div>
            <div style="font-size:12px; color:#B0B8C1; margin:-4px 0 10px 0;">
                내일 원유 가격이 하루에 ±2% 이상 크게 움직일 확률
            </div>
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="font-size:36px; font-weight:900; color:{prob_color};">{prob:.0%}</div>
            </div>
            <div class="prob-bar-bg" style="margin-top:16px;">
                <div class="prob-bar" style="width:{prob*100:.0f}%; background:{prob_color};"></div>
                <div class="prob-threshold" style="left:{threshold*100:.0f}%;"></div>
                <div class="prob-threshold-label" style="left:{threshold*100:.0f}%;">알림 기준 {threshold:.0%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 참고 정보 (방향 + 수익률, 작게)
        st.markdown(f"""
        <div class="card" style="background:#F8F9FA; padding:16px 24px;">
            <div style="font-size:12px; font-weight:700; color:#8B95A1; margin-bottom:8px;">참고 정보</div>
            <div style="display:flex; gap:24px;">
                <div>
                    <span style="font-size:12px; color:#8B95A1;">가격 방향</span>
                    <span style="font-size:16px; font-weight:700; color:{dir_color}; margin-left:8px;">{dir_icon} {dir_text}</span>
                </div>
                <div>
                    <span style="font-size:12px; color:#8B95A1;">예상 수익률</span>
                    <span style="font-size:16px; font-weight:700; color:{ret_color}; margin-left:8px;">{ret_sign}{ret:.2f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # === 투트랙: 매수 타이밍 레이아웃 (제거 가능) ===
    elif mode_key == 'timing':
        is_volatile = prob >= threshold
        dir_agree = (direction == 1 and ret > 0) or (direction == 0 and ret < 0)

        if is_volatile:
            action, action_icon = '관망 권장', '⚪'
            action_color, action_bg = '#8B95A1', '#F2F3F5'
            action_sub = '변동성이 커서 예측이 불확실합니다'
        elif not dir_agree:
            action, action_icon = '관망 권장', '⚪'
            action_color, action_bg = '#8B95A1', '#F2F3F5'
            action_sub = '분석 신호가 엇갈려 추천하기 어렵습니다'
        elif direction == 1:
            action, action_icon = '매수 고려', '🟢'
            action_color, action_bg = '#F04452', '#FFF0F0'
            action_sub = f'상승이 예상됩니다 · 예상 수익률 {ret_sign}{ret:.2f}%'
        else:
            action, action_icon = '매도 고려', '🔴'
            action_color, action_bg = '#3182F6', '#E8F3FF'
            action_sub = f'하락이 예상됩니다 · 예상 수익률 {ret:.2f}%'

        # 추천 액션 카드
        st.markdown(f"""
        <div class="card" style="background:{action_bg}; border-left:4px solid {action_color};">
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="font-size:32px;">{action_icon}</span>
                <div>
                    <div style="font-size:22px; font-weight:900; color:{action_color};">{action}</div>
                    <div style="font-size:13px; color:#4E5968; margin-top:2px;">{action_sub}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 변동성 지표 카드
        prob_color = '#F04452' if is_volatile else '#00C853'
        vol_text = '높음 — 추천 불가' if is_volatile else '낮음 — 추천 가능'

        st.markdown(f"""
        <div class="card">
            <div class="card-label">변동성 지표</div>
            <div style="font-size:12px; color:#B0B8C1; margin:-4px 0 10px 0;">
                이 수치가 낮을수록 예측을 더 신뢰할 수 있습니다
            </div>
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="font-size:28px; font-weight:900; color:{prob_color};">{prob:.0%}</div>
                <span class="badge" style="background:{prob_color}1A; color:{prob_color};">{vol_text}</span>
            </div>
            <div class="prob-bar-bg" style="margin-top:20px;">
                <div class="prob-bar" style="width:{prob*100:.0f}%; background:{prob_color};"></div>
                <div class="prob-threshold" style="left:{threshold*100:.0f}%;"></div>
                <div class="prob-threshold-label" style="left:{threshold*100:.0f}%;">추천 기준 {threshold:.0%}</div>
            </div>
            <div class="card-sub" style="margin-top:8px;">
                {threshold:.0%} 이상이면 변동성이 커서 추천하지 않습니다
            </div>
        </div>
        """, unsafe_allow_html=True)

    # === 기존 레이아웃 (투트랙 미사용 시) ===
    else:
        if prob >= 0.5:
            prob_color, prob_text = "#F04452", "큰 폭 변동이 예상됩니다"
            prob_badge = '<span class="badge badge-danger">위험</span>'
            warn_msg = "급변동 가능성이 높습니다. 큰 폭의 가격 변동에 유의하세요."
        elif prob >= 0.3:
            prob_color, prob_text = "#FF9800", "변동성이 높을 수 있습니다"
            prob_badge = '<span class="badge badge-warn">주의</span>'
            warn_msg = "변동성이 다소 높은 구간입니다. 포지션 관리에 주의하세요."
        else:
            prob_color, prob_text = "#00C853", "시장은 안정적일 것으로 예상됩니다"
            prob_badge = '<span class="badge badge-safe">안전</span>'
            warn_msg = ""

        dir_icon = "↑" if direction == 1 else "↓"
        dir_text = "상승" if direction == 1 else "하락"
        dir_color = "#F04452" if direction == 1 else "#3182F6"
        ret_color = "#F04452" if ret > 0 else "#3182F6"

        st.markdown(f"""
        <div class="card">
            <div class="card-label">급변동 가능성</div>
            <div style="font-size:12px; color:#B0B8C1; margin:-4px 0 10px 0;">
                내일 원유 가격이 하루에 ±2% 이상 움직일 확률입니다.
            </div>
            <div style="display:flex; align-items:center; gap:12px;">
                <div class="card-value" style="color:{prob_color};">{prob:.0%}</div>
                {prob_badge}
            </div>
            <div class="prob-bar-bg" style="margin-top:28px;">
                <div class="prob-bar" style="width:{prob*100:.0f}%; background:{prob_color};"></div>
            </div>
            <div class="card-sub">{prob_text}</div>
            {'<div style="background:#FFF8E6; border-radius:12px; padding:12px 16px; margin-top:14px; font-size:12px; color:#E8890C; line-height:1.6;">⚠ ' + warn_msg + '</div>' if warn_msg else ''}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display:flex; gap:12px;">
            <div class="card" style="flex:1;">
                <div class="card-label">가격 방향</div>
                <div class="card-value" style="color:{dir_color}; font-size:36px;">{dir_icon} {dir_text}</div>
                <div class="card-sub">기준: {data_date.strftime('%m/%d')} 데이터</div>
            </div>
            <div class="card" style="flex:1;">
                <div class="card-label">수익률 예측</div>
                <div class="card-value" style="color:{ret_color}; font-size:36px;">{ret_sign}{ret:.2f}%</div>
                <div class="card-sub">예상 변동폭</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def show_oil():
    # 뒤로가기 버튼
    if st.button("← 돌아가기", key="back"):
        st.session_state.page = 'home'
        st.rerun()

    # === 투트랙 모드 세션 초기화 (제거 가능) ===
    if 'mode' not in st.session_state:
        st.session_state.mode = 'risk'
    mode = MODES[st.session_state.mode]
    # === END ===

    # 데이터셋 로드 (GPR은 live의 마지막 값으로 고정된 상태 + 원자재/VIX는 최신까지)
    _base = os.path.dirname(os.path.abspath(__file__))
    latest_path = os.path.join(_base, '..', '..', 'data', 'commodity_vix_gpr_latest.xlsx')
    live_path = os.path.join(_base, '..', '..', 'data', 'commodity_vix_gpr_live.xlsx')
    dates, prices, prob, pred_dir, pred_ret = load_and_train(latest_path, _file_mtime(latest_path))
    # GPR 실제 반영 마지막 날짜 확인용
    gpr_src = pd.read_excel(live_path, index_col=0)
    gpr_last = gpr_src.index[-1]
    data_last = dates[-1]

    # 오늘 & 내일
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    if tomorrow.weekday() == 5:
        tomorrow += timedelta(days=2)
    elif tomorrow.weekday() == 6:
        tomorrow += timedelta(days=1)

    # 헤더
    st.markdown(f"""
    <div class="main-header">
        <h1>원유 급변동 알림</h1>
        <p>AI 기반 실시간 분석</p>
    </div>
    """, unsafe_allow_html=True)

    # 현재가 카드
    price = prices[-1]
    st.markdown(f"""
    <div class="card">
        <div style="text-align:center;">
            <span class="date-pill">오늘 {today.strftime('%Y년 %m월 %d일')}</span>
        </div>
        <div style="text-align:center;">
            <div class="card-label">WTI 원유 최근 종가</div>
            <div class="card-value" style="color:#191F28;">${price:.2f}</div>
            <div class="card-sub">최근 데이터 기준 ({data_last.strftime('%Y.%m.%d')})</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === 투트랙 모드 토글 UI (제거 가능) ===
    st.markdown("""
    <div style="margin-top:16px;">
        <div class="card-label" style="padding:0 4px; margin-bottom:6px;">서비스 모드</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(f"{MODES['risk']['icon']}  {MODES['risk']['label']}",
                     key='btn_mode_risk', use_container_width=True,
                     type='primary' if st.session_state.mode == 'risk' else 'secondary'):
            st.session_state.mode = 'risk'
            st.rerun()
    with col_b:
        if st.button(f"{MODES['timing']['icon']}  {MODES['timing']['label']}",
                     key='btn_mode_timing', use_container_width=True,
                     type='primary' if st.session_state.mode == 'timing' else 'secondary'):
            st.session_state.mode = 'timing'
            st.rerun()

    # 모드 설명 뱃지
    st.markdown(f"""
    <div style="background:{mode['color']}0F; border-left:3px solid {mode['color']};
                padding:12px 16px; border-radius:8px; margin:8px 0 0 0;">
        <div style="font-size:12px; color:{mode['color']}; font-weight:700; margin-bottom:2px;">
            {mode['focus']} · 경보 기준 {mode['threshold']:.0%}
        </div>
        <div style="font-size:13px; color:#4E5968; line-height:1.5;">
            {mode['desc']}<br>
            <span style="color:#8B95A1; font-size:12px;">대상: {mode['target']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # === END 투트랙 토글 ===

    # 예측 대상일
    st.markdown(f"""
    <div style="text-align:center; margin:20px 0 0 0;">
        <span style="font-size:13px; font-weight:700; color:#3182F6;">
            {tomorrow.strftime('%m월 %d일')} 예측
        </span>
    </div>
    """, unsafe_allow_html=True)

    # 통합 예측 카드 (GPR 고정 + 최신 원자재/VIX · 투트랙 모드 반영)
    render_prediction(
        label="예측",
        tag=f"위기지수 {gpr_last.strftime('%m/%d')} · 시장지표 {data_last.strftime('%m/%d')}",
        tag_color="#3182F6",
        data_date=data_last,
        prob=prob[-1], direction=pred_dir[-1], ret=pred_ret[-1],
        tomorrow=tomorrow,
        threshold=mode['threshold'],
        mode_key=st.session_state.mode,
    )

    # 수익률 참고사항
    st.markdown("""
    <div class="card" style="background:#F8F9FA; padding:20px 24px;">
        <div style="font-size:12px; color:#8B95A1; line-height:1.7;">
            <span style="font-weight:700; color:#4E5968;">수익률 예측 참고사항</span><br>
            수익률 숫자는 급변동 감지에 비해 정확도가 낮고,
            실제보다 보수적으로 예측하는 경향이 있습니다.
            급변동 가능성과 가격 방향을 중심으로 참고하시고,
            수익률 숫자는 보조 지표로 활용하세요.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 상세 정보 카드
    st.markdown(f"""
    <div class="card">
        <div class="card-label">상세 분석</div>
        <div class="divider"></div>
        <div class="info-row">
            <span class="info-label">예측 대상일</span>
            <span class="info-value">{tomorrow.strftime('%Y년 %m월 %d일')}</span>
        </div>
        <div class="info-row">
            <span class="info-label">위기 지수 반영일</span>
            <span class="info-value">{gpr_last.strftime('%Y.%m.%d')}</span>
        </div>
        <div class="info-row">
            <span class="info-label">시장데이터 기준</span>
            <span class="info-value">{data_last.strftime('%Y.%m.%d')}</span>
        </div>
        <div class="info-row">
            <span class="info-label">분석 방식</span>
            <span class="info-value">AI 예측 (교차검증)</span>
        </div>
        <div class="info-row">
            <span class="info-label">학습 기간</span>
            <span class="info-value">2020.01 ~ 2026.04</span>
        </div>
        <div class="info-row">
            <span class="info-label">급변동 기준</span>
            <span class="info-value">하루 ±2% 이상 움직임</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 하단 면책 고지
    st.markdown("""
    <div class="bottom-note">
        본 서비스는 참고용으로만 활용하시기 바라며,<br>
        투자 권유가 아닙니다. 투자 판단의 책임은 본인에게 있습니다.<br><br>
        전쟁·정치 불안에 반응하는 원자재 가격 알림 서비스<br>
        임태후 · 유종헌
    </div>
    """, unsafe_allow_html=True)


# ========== 페이지 라우팅 ==========
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'oil':
    show_oil()
