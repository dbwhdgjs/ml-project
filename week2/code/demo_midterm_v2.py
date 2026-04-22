"""
중간평가 시연 코드 v2
흐름: 프로젝트 소개 → 데이터 → 1주차 한계 → 2주차 개선 → 성능 비교 → 예측 시연 → 향후 계획
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import clone

import warnings
warnings.filterwarnings('ignore')
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'commodity_vix_gpr_data.xlsx')


def wait(msg="다음 단계"):
    input(f"\n  [Enter] {msg} →")


def section(title, step):
    print("\n")
    print("=" * 65)
    print(f"  STEP {step}. {title}")
    print("=" * 65)


# ============================================================
# STEP 1. 프로젝트 소개
# ============================================================
section("프로젝트 소개", 1)

print("""
  프로젝트: 지정학적 리스크 기반 원자재 이상변동 감지

  목표:
    ① 원유(WTI) 가격의 이상 변동(±2% 이상)을 사전 감지
    ② 지정학적 리스크(GPR Index)와 시장 공포(VIX)를 활용

  조원: 20232501 임태후(조장), 20232514 유종헌

  사용 데이터:
    - 원유(WTI), 금, 천연가스, 은 가격 (Yahoo Finance)
    - VIX 공포지수 (CBOE)
    - GPR 지정학적 리스크 지수 (Caldara & Iacoviello)
    - 기간: 2020-01-01 ~ 2026-03-16
""")

wait("데이터 로드")

# ============================================================
# STEP 2. 데이터 로드 & 기본 통계
# ============================================================
section("데이터 로드 & 기본 통계", 2)

data = pd.read_excel(DATA_PATH, index_col=0)
data['원유_수익률'] = data['원유(WTI)'].pct_change() * 100
for col in ['금', '천연가스', '은']:
    data[f'{col}_수익률'] = data[col].pct_change() * 100
data['이상변동'] = (data['원유_수익률'].abs() >= 2.0).astype(int)
data = data.dropna()

normal = (data['이상변동'] == 0).sum()
anomaly = (data['이상변동'] == 1).sum()

print(f"\n  전체 기간: {data.index[0].date()} ~ {data.index[-1].date()}")
print(f"  총 데이터: {len(data)}일")
print(f"  Feature 7개: VIX, GPR지수, GPR_실제행동, GPR_위협, 금/천연가스/은 수익률")
print(f"\n  이상변동 분포:")
print(f"    정상 (|수익률| < 2%):  {normal}일 ({normal/len(data)*100:.1f}%)")
print(f"    이상 (|수익률| >= 2%): {anomaly}일 ({anomaly/len(data)*100:.1f}%)")
print(f"\n  ※ 클래스 불균형: 이상변동이 전체의 약 {anomaly/len(data)*100:.0f}%에 불과")

# 원유 가격 + 이상변동 시각화
fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})

ax1 = axes[0]
ax1.plot(data.index, data['원유(WTI)'], color='#1976D2', lw=1.5)
anomaly_dates = data[data['이상변동'] == 1].index
anomaly_prices = data.loc[anomaly_dates, '원유(WTI)']
ax1.scatter(anomaly_dates, anomaly_prices, color='red', s=20, alpha=0.6, zorder=5, label='이상변동 발생일')
ax1.set_title('원유(WTI) 가격 추이 & 이상변동 발생일', fontsize=15, fontweight='bold')
ax1.set_ylabel('가격 (USD)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.bar(data.index, data['원유_수익률'], color=['red' if abs(r) >= 2 else '#90CAF9' for r in data['원유_수익률']], width=1.5)
ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='±2% 기준선')
ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
ax2.set_title('원유 일별 수익률', fontsize=13, fontweight='bold')
ax2.set_ylabel('수익률 (%)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

wait("1주차 한계 분석")

# ============================================================
# STEP 3. 1주차 한계 분석
# ============================================================
section("1주차 결과 & 한계점", 3)

print("""
  1주차: Random Forest + Random Split

  결과:
    Accuracy  : 0.683
    Precision : 0.640
    Recall    : 0.283  ← 핵심 문제
    F1-Score  : 0.393
""")

# 1주차 방식으로 실제 재현
features = ['VIX', 'GPR지수', 'GPR_실제행동', 'GPR_위협',
            '금_수익률', '천연가스_수익률', '은_수익률']
X = data[features].values
y = data['이상변동'].values

X_train_rs, X_test_rs, y_train_rs, y_test_rs = train_test_split(
    X, y, test_size=0.2, random_state=42
)
rf_week1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_week1.fit(X_train_rs, y_train_rs)
y_pred_rs = rf_week1.predict(X_test_rs)

week1_results = {
    'accuracy': accuracy_score(y_test_rs, y_pred_rs),
    'precision': precision_score(y_test_rs, y_pred_rs, zero_division=0),
    'recall': recall_score(y_test_rs, y_pred_rs, zero_division=0),
    'f1': f1_score(y_test_rs, y_pred_rs, zero_division=0)
}

print(f"  [재현 결과]")
print(f"    Accuracy : {week1_results['accuracy']:.3f}")
print(f"    Precision: {week1_results['precision']:.3f}")
print(f"    Recall   : {week1_results['recall']:.3f}")
print(f"    F1-Score : {week1_results['f1']:.3f}")

print("""
  한계점 3가지:

  ① Random Split 사용 → 미래 데이터가 학습에 포함 (Data Leakage)
     시계열 데이터인데 랜덤으로 섞어서 분할하면 비현실적

  ② Random Forest 1개만 사용 → 다른 모델과 비교 없음
     어떤 모델이 이 문제에 적합한지 판단 불가

  ③ Recall 28.3% → 이상변동 72%를 놓침
     이상변동을 감지하는 것이 목적인데, 대부분 놓치고 있음
""")

wait("2주차 개선 - 모델 학습")

# ============================================================
# STEP 4. 2주차 개선 - 5개 모델 × TimeSeriesSplit
# ============================================================
section("2주차 개선: 5개 모델 × TimeSeriesSplit 5-Fold", 4)

print("""
  개선 사항:
    ① Random Split → TimeSeriesSplit (시간 순서 보장, 미래 유출 방지)
    ② 1개 모델 → 5개 모델 비교
       - Random Forest: 앙상블 트리 (1주차 베이스라인)
       - Logistic Regression: 선형 모델 (해석 용이)
       - SVM: 고차원 분류 (커널 트릭)
       - XGBoost: 부스팅 앙상블 (순차 학습)
       - KNN: 거리 기반 (유사 패턴 탐색)
    ③ StandardScaler 적용 (SVM, KNN, LR은 스케일에 민감)
""")

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

print(f"\n  학습 시작 (TimeSeriesSplit 5-Fold)...\n")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    anom_train = y_train.sum()
    anom_test = y_test.sum()
    print(f"  Fold {fold}: 학습 {len(train_idx)}일(이상{anom_train}) → 테스트 {len(test_idx)}일(이상{anom_test})")

    for name, model in models.items():
        m = clone(model)
        m.fit(X_train_s, y_train)
        y_pred = m.predict(X_test_s)

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
                y_proba = m.predict_proba(X_test_s)[:, 1]
            last_fold_preds[name] = {
                'model': m, 'scaler': scaler,
                'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba,
                'X_test': X_test_s, 'test_idx': test_idx
            }

    print(f"         {'모델':20s} | Acc    Prec   Rec    F1")
    for name in models:
        r = results[name]
        print(f"         {name:20s} | {r['accuracy'][-1]:.3f}  {r['precision'][-1]:.3f}  {r['recall'][-1]:.3f}  {r['f1'][-1]:.3f}")
    print()

wait("성능 비교 결과")

# ============================================================
# STEP 5. 성능 비교 + 1주차 vs 2주차
# ============================================================
section("성능 비교 결과", 5)

# 평균 계산
summary = {}
for name, metrics in results.items():
    summary[name] = {k: np.mean(v) for k, v in metrics.items()}

print(f"\n  모델별 평균 성능 (TimeSeriesSplit 5-Fold)")
print(f"  {'모델':22s} | {'Accuracy':>8s} | {'Precision':>9s} | {'Recall':>8s} | {'F1':>8s}")
print(f"  {'-'*65}")
for name, avg in summary.items():
    # F1 최고면 ★ 표시
    marker = ""
    if name == max(summary, key=lambda x: summary[x]['f1']):
        marker = " ★ F1 최고"
    if name == max(summary, key=lambda x: summary[x]['recall']):
        marker += " ★ Recall 최고"
    print(f"  {name:22s} | {avg['accuracy']:8.3f} | {avg['precision']:9.3f} | {avg['recall']:8.3f} | {avg['f1']:8.3f}{marker}")

best_f1_name = max(summary, key=lambda x: summary[x]['f1'])
best_rec_name = max(summary, key=lambda x: summary[x]['recall'])

# 1주차 vs 2주차 비교
print(f"\n\n  ── 1주차 vs 2주차 비교 (Random Forest 기준) ──")
print(f"  {'지표':12s} | {'1주차(Random Split)':>20s} | {'2주차(TimeSeriesSplit)':>22s} | {'변화':>8s}")
print(f"  {'-'*72}")

week2_rf = summary['Random Forest']
for metric, label in [('accuracy','Accuracy'), ('precision','Precision'), ('recall','Recall'), ('f1','F1-Score')]:
    v1 = week1_results[metric]
    v2 = week2_rf[metric]
    diff = v2 - v1
    arrow = "▲" if diff > 0 else "▼" if diff < 0 else "─"
    print(f"  {label:12s} | {v1:20.3f} | {v2:22.3f} | {arrow} {abs(diff):.3f}")

print(f"\n  해석:")
print(f"    - TimeSeriesSplit은 더 현실적인 평가 → 1주차보다 수치가 다를 수 있음")
print(f"    - 1주차의 Random Split은 미래 데이터 유출로 성능이 부풀려졌을 가능성")
print(f"    - 2주차에서 5개 모델을 비교한 결과, {best_f1_name}이 F1 기준 최적")

# 시각화: 모델 비교 + ROC + Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# (1) 모델별 성능 비교 바 차트
ax = axes[0]
model_names = list(summary.keys())
metrics_list = ['accuracy', 'precision', 'recall', 'f1']
labels = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(model_names))
width = 0.18
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

for i, (metric, label, color) in enumerate(zip(metrics_list, labels, colors)):
    vals = [summary[m][metric] for m in model_names]
    ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

ax.set_title('모델별 성능 비교', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=9)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# (2) ROC 커브
ax = axes[1]
roc_colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
for (name, pred), color in zip(last_fold_preds.items(), roc_colors):
    if pred['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(pred['y_test'], pred['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_title('ROC 커브 비교', fontsize=14, fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (3) 1주차 vs 2주차 RF 비교
ax = axes[2]
categories = ['Accuracy', 'Precision', 'Recall', 'F1']
w1_vals = [week1_results[m] for m in metrics_list]
w2_vals = [week2_rf[m] for m in metrics_list]
x_pos = np.arange(len(categories))
w = 0.35
ax.bar(x_pos - w/2, w1_vals, w, label='1주차 (Random Split)', color='#BBDEFB', edgecolor='#1976D2', lw=1.5)
ax.bar(x_pos + w/2, w2_vals, w, label='2주차 (TimeSeriesSplit)', color='#1976D2', edgecolor='#0D47A1', lw=1.5)
for i, (v1, v2) in enumerate(zip(w1_vals, w2_vals)):
    ax.text(i - w/2, v1 + 0.02, f'{v1:.3f}', ha='center', fontsize=9)
    ax.text(i + w/2, v2 + 0.02, f'{v2:.3f}', ha='center', fontsize=9)
ax.set_title('1주차 vs 2주차 (RF 비교)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

wait("실제 예측 시연")

# ============================================================
# STEP 6. 실제 예측 시연 - 핵심 케이스
# ============================================================
section("실제 예측 시연 (XGBoost)", 6)

# 방향/수익률 레이블
data['방향'] = (data['원유_수익률'] > 0).astype(int)

X_all = data[features].values
y_anomaly = data['이상변동'].values
y_direction = data['방향'].values
y_return = data['원유_수익률'].values

# 마지막 30일 테스트
split = len(data) - 30
scaler_final = StandardScaler()
X_train_f = scaler_final.fit_transform(X_all[:split])
X_test_f = scaler_final.transform(X_all[split:])

# 3개 모델 학습
model_anom = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                            random_state=42, eval_metric='logloss', verbosity=0)
model_anom.fit(X_train_f, y_anomaly[:split])

model_dir = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                           random_state=42, eval_metric='logloss', verbosity=0)
model_dir.fit(X_train_f, y_direction[:split])

model_ret = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                          random_state=42, verbosity=0)
model_ret.fit(X_train_f, y_return[:split])

# 예측
pred_anom = model_anom.predict(X_test_f)
prob_anom = model_anom.predict_proba(X_test_f)[:, 1]
pred_dir = model_dir.predict(X_test_f)
pred_ret = model_ret.predict(X_test_f)

test_dates = data.index[split:]
actual_ret = y_return[split:]
actual_anom = y_anomaly[split:]
actual_dir = y_direction[split:]

# 핵심 케이스 추출: 이상변동 확률 높은 날 + 실제 이상변동 날
print(f"\n  최근 30일 중 주목할 케이스:\n")

# 케이스 1: 이상확률 가장 높았던 날
top_idx = np.argmax(prob_anom)
print(f"  ┌─ 케이스 1: 이상변동 확률 최고일 ─────────────────────────")
print(f"  │ 날짜: {test_dates[top_idx].date()}")
print(f"  │ 이상변동 확률: {prob_anom[top_idx]:.1%}")
print(f"  │ 모델 판단: {'⚠️  이상변동' if pred_anom[top_idx] == 1 else '정상'}")
print(f"  │ 실제: {'⚠️  이상변동' if actual_anom[top_idx] == 1 else '정상'} (수익률 {actual_ret[top_idx]:+.2f}%)")
print(f"  │ 방향 예측: {'상승 ↑' if pred_dir[top_idx] == 1 else '하락 ↓'} → 실제: {'상승 ↑' if actual_dir[top_idx] == 1 else '하락 ↓'}")
print(f"  │ 당일 VIX: {data.iloc[split + top_idx]['VIX']:.1f}, GPR: {data.iloc[split + top_idx]['GPR지수']:.1f}")
print(f"  └─────────────────────────────────────────────────────")

# 케이스 2: 실제 이상변동이 있었던 날 중 하나
real_anomaly_idx = np.where(actual_anom == 1)[0]
if len(real_anomaly_idx) > 0:
    case_idx = real_anomaly_idx[0]
    hit = "감지 성공 ✓" if pred_anom[case_idx] == 1 else "감지 실패 ✗"
    print(f"\n  ┌─ 케이스 2: 실제 이상변동 발생일 ────────────────────────")
    print(f"  │ 날짜: {test_dates[case_idx].date()}")
    print(f"  │ 실제 수익률: {actual_ret[case_idx]:+.2f}% (이상변동)")
    print(f"  │ 이상변동 확률: {prob_anom[case_idx]:.1%}")
    print(f"  │ 모델 판단: {hit}")
    print(f"  │ 수익률 예측: {pred_ret[case_idx]:+.2f}% → 실제: {actual_ret[case_idx]:+.2f}%")
    print(f"  │ 당일 VIX: {data.iloc[split + case_idx]['VIX']:.1f}, GPR: {data.iloc[split + case_idx]['GPR지수']:.1f}")
    print(f"  └─────────────────────────────────────────────────────")

# 케이스 3: 정상인데 정확히 판단한 날
normal_correct = np.where((actual_anom == 0) & (pred_anom == 0))[0]
if len(normal_correct) > 0:
    case_idx = normal_correct[len(normal_correct)//2]
    print(f"\n  ┌─ 케이스 3: 정상 → 정상 (정확 판단) ──────────────────────")
    print(f"  │ 날짜: {test_dates[case_idx].date()}")
    print(f"  │ 실제 수익률: {actual_ret[case_idx]:+.2f}% (정상)")
    print(f"  │ 이상변동 확률: {prob_anom[case_idx]:.1%} → 정상 판단 ✓")
    print(f"  │ 당일 VIX: {data.iloc[split + case_idx]['VIX']:.1f}, GPR: {data.iloc[split + case_idx]['GPR지수']:.1f}")
    print(f"  └─────────────────────────────────────────────────────")

# 30일 전체 정확도 요약
anom_acc = accuracy_score(actual_anom, pred_anom)
dir_acc = accuracy_score(actual_dir, pred_dir)
ret_mae = np.mean(np.abs(pred_ret - actual_ret))
anom_detected = np.sum((actual_anom == 1) & (pred_anom == 1))
anom_total = np.sum(actual_anom == 1)

print(f"\n  ── 30일 전체 요약 ──")
print(f"    이상변동 감지 정확도: {anom_acc:.1%}")
print(f"    이상변동 감지율(Recall): {anom_detected}/{anom_total}건")
print(f"    가격 방향 예측 정확도: {dir_acc:.1%}")
print(f"    수익률 예측 평균 오차: ±{ret_mae:.2f}%p")

# 30일 예측 시각화
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

ax1 = axes[0]
ax1.plot(test_dates, actual_ret, 'o-', color='#1976D2', lw=2, markersize=5, label='실제 수익률')
ax1.plot(test_dates, pred_ret, 's--', color='#F44336', lw=1.5, markersize=4, alpha=0.7, label='예측 수익률')
ax1.axhline(y=2, color='red', linestyle=':', alpha=0.4)
ax1.axhline(y=-2, color='red', linestyle=':', alpha=0.4)
ax1.fill_between(test_dates, 2, max(max(actual_ret), max(pred_ret))+1, alpha=0.05, color='red')
ax1.fill_between(test_dates, -2, min(min(actual_ret), min(pred_ret))-1, alpha=0.05, color='red')
ax1.set_title('최근 30일: 수익률 예측 vs 실제', fontsize=14, fontweight='bold')
ax1.set_ylabel('수익률 (%)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

ax2 = axes[1]
colors_bar = []
for i in range(len(test_dates)):
    if actual_anom[i] == 1 and pred_anom[i] == 1:
        colors_bar.append('#4CAF50')  # 감지 성공
    elif actual_anom[i] == 1 and pred_anom[i] == 0:
        colors_bar.append('#F44336')  # 놓침
    elif actual_anom[i] == 0 and pred_anom[i] == 1:
        colors_bar.append('#FF9800')  # 오경보
    else:
        colors_bar.append('#90CAF9')  # 정상→정상

ax2.bar(test_dates, prob_anom * 100, color=colors_bar, width=1.5)
ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% 기준선')
ax2.set_title('이상변동 예측 확률 (초록=감지성공, 빨강=놓침, 주황=오경보, 파랑=정상)', fontsize=12, fontweight='bold')
ax2.set_ylabel('이상변동 확률 (%)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

wait("한계 & 향후 계획")

# ============================================================
# STEP 7. 현재 한계 & 향후 개선 계획
# ============================================================
section("현재 한계 & 향후 개선 계획", 7)

print(f"""
  현재 한계:

    ① Recall 부족 → 이상변동을 여전히 상당수 놓침
       - 원인: 클래스 불균형 (이상변동 ~{anomaly/len(data)*100:.0f}%)
       - 해결: SMOTE, class_weight 조정

    ② Feature 부족 → 당일 데이터만 사용
       - 원인: 시차(Lag) 반영 없음, 기술 지표 없음
       - 해결: 이동평균, RSI, 볼린저밴드, 전일 VIX 변화율 등 추가

    ③ 하이퍼파라미터 미튜닝
       - 현재: 기본값 사용
       - 해결: GridSearch / RandomizedSearch로 최적화

  향후 계획:

    3주차 (제출③ 5/14):
      - 시차 Feature 추가 (VIX_lag1, GPR_lag1 등)
      - 기술 지표 추가 (이동평균, RSI)
      - SMOTE로 클래스 불균형 해결
      - 하이퍼파라미터 튜닝

    4주차 (제출④ 6/4):
      - 4개 원자재 통합 모델
      - Streamlit 대시보드
      - API 자동 데이터 수집
""")

print("\n" + "=" * 65)
print("  시연 완료")
print("=" * 65)
