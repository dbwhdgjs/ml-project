"""
중간평가 모델 시연
- 데이터 로드 → 5개 모델 학습 → 성능 비교 → 예측 시연
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.base import clone

import warnings
warnings.filterwarnings('ignore')
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'commodity_vix_gpr_data.xlsx')


def wait():
    input("\n  [Enter] 다음 단계 →")


# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n" + "=" * 60)
print("  데이터 로드")
print("=" * 60)

data = pd.read_excel(DATA_PATH, index_col=0)
data['원유_수익률'] = data['원유(WTI)'].pct_change() * 100
for col in ['금', '천연가스', '은']:
    data[f'{col}_수익률'] = data[col].pct_change() * 100
data['이상변동'] = (data['원유_수익률'].abs() >= 2.0).astype(int)
data = data.dropna()

normal = (data['이상변동'] == 0).sum()
anomaly = (data['이상변동'] == 1).sum()
print(f"\n  기간: {data.index[0].date()} ~ {data.index[-1].date()} ({len(data)}일)")
print(f"  정상: {normal}일 ({normal/len(data)*100:.1f}%)  |  이상변동: {anomaly}일 ({anomaly/len(data)*100:.1f}%)")

wait()

# ============================================================
# 2. 모델 학습 (TimeSeriesSplit 5-Fold)
# ============================================================
print("\n" + "=" * 60)
print("  모델 학습 — TimeSeriesSplit 5-Fold")
print("=" * 60)

features = ['VIX', 'GPR지수', 'GPR_실제행동', 'GPR_위협',
            '금_수익률', '천연가스_수익률', '은_수익률']
X = data[features].values
y = data['이상변동'].values

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=42, eval_metric='logloss', verbosity=0),
    'KNN': KNeighborsClassifier(n_neighbors=5),
}

tscv = TimeSeriesSplit(n_splits=5)
results = {name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
           for name in models}
last_fold_preds = {}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n  Fold {fold}: 학습 {len(train_idx)}일 → 테스트 {len(test_idx)}일")
    for name, model in models.items():
        m = clone(model)
        m.fit(X_train_scaled, y_train)
        y_pred = m.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name]['accuracy'].append(acc)
        results[name]['precision'].append(prec)
        results[name]['recall'].append(rec)
        results[name]['f1'].append(f1)

        if fold == 5:
            y_proba = None
            if hasattr(m, 'predict_proba'):
                y_proba = m.predict_proba(X_test_scaled)[:, 1]
            last_fold_preds[name] = {
                'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba
            }

        print(f"    {name:22s} | Acc {acc:.3f} | Prec {prec:.3f} | Rec {rec:.3f} | F1 {f1:.3f}")

wait()

# ============================================================
# 3. 성능 비교 결과
# ============================================================
print("\n" + "=" * 60)
print("  모델별 평균 성능 비교")
print("=" * 60)

summary = {}
print(f"\n  {'모델':22s} | {'Accuracy':>8s} | {'Precision':>9s} | {'Recall':>8s} | {'F1':>8s}")
print(f"  {'-'*65}")
for name, metrics in results.items():
    avg = {k: np.mean(v) for k, v in metrics.items()}
    summary[name] = avg
    print(f"  {name:22s} | {avg['accuracy']:8.3f} | {avg['precision']:9.3f} | {avg['recall']:8.3f} | {avg['f1']:8.3f}")

best_f1 = max(summary, key=lambda x: summary[x]['f1'])
best_rec = max(summary, key=lambda x: summary[x]['recall'])
print(f"\n  ★ F1 기준 최적: {best_f1} ({summary[best_f1]['f1']:.3f})")
print(f"  ★ Recall 기준 최적: {best_rec} ({summary[best_rec]['recall']:.3f})")

wait()

# ============================================================
# 5. XGBoost 예측 시연 (이상변동 + 방향 + 수익률)
# ============================================================
from xgboost import XGBRegressor

print("\n" + "=" * 60)
print("  XGBoost 예측 시연")
print("=" * 60)

# 방향 레이블 추가
data['방향'] = (data['원유_수익률'] > 0).astype(int)  # 1=상승, 0=하락

X_all = data[features].values
y_anomaly = data['이상변동'].values
y_direction = data['방향'].values
y_return = data['원유_수익률'].values

# 학습/테스트 분리 (마지막 30일 테스트)
split = len(data) - 30
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_all[:split])
X_test_s = scaler.transform(X_all[split:])

# ① 이상변동 감지
model_anomaly = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                               random_state=42, eval_metric='logloss', verbosity=0)
model_anomaly.fit(X_train_s, y_anomaly[:split])
pred_anomaly = model_anomaly.predict(X_test_s)
prob_anomaly = model_anomaly.predict_proba(X_test_s)[:, 1]

# ② 가격 방향 예측
model_dir = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                           random_state=42, eval_metric='logloss', verbosity=0)
model_dir.fit(X_train_s, y_direction[:split])
pred_dir = model_dir.predict(X_test_s)
prob_dir = model_dir.predict_proba(X_test_s)[:, 1]

# ③ 수익률 예측
model_ret = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                          random_state=42, verbosity=0)
model_ret.fit(X_train_s, y_return[:split])
pred_ret = model_ret.predict(X_test_s)

# 최근 30일 결과 출력
test_dates = data.index[split:]
actual_ret = y_return[split:]
actual_anomaly = y_anomaly[split:]
actual_dir = y_direction[split:]

print(f"\n  최근 30일 예측 결과 (XGBoost)")
print(f"    날짜        이상확률  이상예측  실제    방향예측  실제    수익률예측  실제")
print(f"  {'─'*85}")
for i in range(len(test_dates)):
    date_str = str(test_dates[i].date())
    anom_str = "이상  " if pred_anomaly[i] == 1 else "정상  "
    anom_act = "이상" if actual_anomaly[i] == 1 else "정상"
    dir_str = "상승  " if pred_dir[i] == 1 else "하락  "
    dir_act = "상승" if actual_dir[i] == 1 else "하락"
    print(f"    {date_str}   {prob_anomaly[i]:5.1%}     {anom_str}  {anom_act}    {dir_str}  {dir_act}    {pred_ret[i]:>+7.2f}%  {actual_ret[i]:>+7.2f}%")

# 정확도 요약
anom_acc = accuracy_score(actual_anomaly, pred_anomaly)
dir_acc = accuracy_score(actual_dir, pred_dir)
ret_mae = np.mean(np.abs(pred_ret - actual_ret))

print(f"\n  요약:")
print(f"    이상변동 감지 정확도: {anom_acc:.1%}")
print(f"    가격 방향 예측 정확도: {dir_acc:.1%}")
print(f"    수익률 예측 평균 오차: {ret_mae:.2f}%p")

print("\n  시연 완료")
