"""
1주차 기본 모델: 원유(WTI) 이상 변동 감지 & 가격 방향 예측
- 모델 A: 이상 변동 감지 (Random Forest Classifier)
- 모델 B: 가격 방향 예측 & 매매 타이밍 분석

Colab에서 실행 시: !pip install yfinance xlrd
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# STEP 1. 데이터 수집
# ============================================================
def load_data():
    """원자재 가격 + VIX + GPR Index 수집"""
    tickers = {
        'CL=F': '원유(WTI)',
        'GC=F': '금',
        'NG=F': '천연가스',
        'SI=F': '은',
        '^VIX': 'VIX'
    }

    price_data = []
    for ticker, name in tickers.items():
        df = yf.download(ticker, start='2020-01-01', end='2026-03-20', auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Close']].rename(columns={'Close': name})
        price_data.append(df)

    data = pd.concat(price_data, axis=1)
    data.index.name = '날짜'
    data = data.dropna()

    # GPR Index
    gpr_url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    gpr = pd.read_excel(gpr_url)
    gpr['날짜'] = pd.to_datetime(gpr['date'])
    gpr = gpr[['날짜', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT']]
    gpr.columns = ['날짜', 'GPR지수', 'GPR_실제행동', 'GPR_위협']
    gpr = gpr[gpr['날짜'] >= '2020-01-01'].set_index('날짜')

    data = data.join(gpr, how='inner')
    return data


# ============================================================
# STEP 2. Feature 생성
# ============================================================
def prepare_features(data):
    """수익률 계산 및 이상변동 레이블 생성"""
    # 원유 수익률
    data['원유_수익률'] = data['원유(WTI)'].pct_change() * 100

    # 이상변동: |수익률| >= 2%
    threshold = 2.0
    data['이상변동'] = (data['원유_수익률'].abs() >= threshold).astype(int)

    # 다른 원자재 수익률
    for col in ['금', '천연가스', '은']:
        data[f'{col}_수익률'] = data[col].pct_change() * 100

    # 내일 수익률/방향 (가격 예측용)
    data['내일_원유_수익률'] = data['원유_수익률'].shift(-1)
    data['내일_방향'] = (data['내일_원유_수익률'] > 0).astype(int)

    data = data.dropna()
    return data


# ============================================================
# 모델 A: 이상 변동 감지
# ============================================================
def run_anomaly_detection(data):
    """이상 변동 감지 모델 학습 및 평가"""
    features = ['VIX', 'GPR지수', 'GPR_실제행동', 'GPR_위협',
                '금_수익률', '천연가스_수익률', '은_수익률']

    X = data[features]
    y = data['이상변동']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=" * 50)
    print("[ 모델 A: 이상 변동 감지 결과 ]")
    print("=" * 50)
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n혼동 행렬:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    print(f"\nFeature 중요도:")
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    for name, val in importance.items():
        print(f"  {name:15s}: {val:.4f}")

    return model


# ============================================================
# 모델 B: 가격 방향 예측 & 매매 타이밍
# ============================================================
def run_price_prediction(data):
    """가격 방향 예측 및 매매 시뮬레이션"""
    features = ['VIX', 'GPR지수', 'GPR_실제행동', 'GPR_위협',
                '원유_수익률', '금_수익률', '천연가스_수익률', '은_수익률']

    # 시간순 분할
    split = int(len(data) * 0.8)
    train = data.iloc[:split]
    test = data.iloc[split:].copy()

    X_train, X_test = train[features], test[features]

    # 방향 예측 (분류)
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X_train, train['내일_방향'])
    test['예측_방향'] = clf.predict(X_test)

    # 수익률 예측 (회귀)
    reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    reg.fit(X_train, train['내일_원유_수익률'])
    test['예측_수익률'] = reg.predict(X_test)

    # 매매 신호
    test['매수신호'] = ((test['예측_방향'] == 1) & (test['예측_수익률'] > 0.3)).astype(int)
    test['매도신호'] = ((test['예측_방향'] == 0) & (test['예측_수익률'] < -0.3)).astype(int)

    print("\n" + "=" * 50)
    print("[ 모델 B: 가격 방향 예측 결과 ]")
    print("=" * 50)
    print(f"방향 정확도: {accuracy_score(test['내일_방향'], test['예측_방향']):.1%}")

    buy = test[test['매수신호'] == 1]
    sell = test[test['매도신호'] == 1]
    if len(buy) > 0:
        print(f"매수 신호: {len(buy)}회, 적중률 {(buy['내일_원유_수익률'] > 0).mean():.1%}")
    if len(sell) > 0:
        print(f"매도 신호: {len(sell)}회, 적중률 {(sell['내일_원유_수익률'] < 0).mean():.1%}")

    # 시뮬레이션
    initial = 1000000
    hold_val = initial * (1 + test['내일_원유_수익률'] / 100).cumprod().iloc[-1]
    signal_val = initial * (1 + test['내일_원유_수익률'] / 100 * test['매수신호']).cumprod().iloc[-1]
    print(f"\n100만원 시뮬레이션:")
    print(f"  그냥 보유: {hold_val:,.0f}원")
    print(f"  모델 매수: {signal_val:,.0f}원")

    return clf, reg


# ============================================================
# 실행
# ============================================================
if __name__ == '__main__':
    print("데이터 수집 중...")
    data = load_data()
    print(f"수집 완료: {len(data)}일")

    data = prepare_features(data)
    anomaly_model = run_anomaly_detection(data)
    price_clf, price_reg = run_price_prediction(data)
