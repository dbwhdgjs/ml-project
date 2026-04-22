"""
2주차 모델 확장 & 비교
- 1주차 한계 해결: Random Split → TimeSeriesSplit
- 5개 모델 비교: Random Forest, Logistic Regression, SVM, XGBoost, KNN
- 이상 변동 감지 (모델 A) 중심으로 성능 비교

Colab에서 실행 시: !pip install xgboost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'commodity_vix_gpr_data.xlsx')
CHARTS_DIR = os.path.join(BASE_DIR, 'charts_week2')
os.makedirs(CHARTS_DIR, exist_ok=True)


# ============================================================
# STEP 1. 데이터 로드 & Feature 생성 (1주차와 동일)
# ============================================================
def load_and_prepare():
    """1주차에서 저장한 통합 데이터를 불러와 Feature 생성"""
    data = pd.read_excel(DATA_PATH, index_col=0)
    print(f"데이터 로드: {len(data)}일, {data.index[0].date()} ~ {data.index[-1].date()}")

    # 수익률 계산
    data['원유_수익률'] = data['원유(WTI)'].pct_change() * 100
    for col in ['금', '천연가스', '은']:
        data[f'{col}_수익률'] = data[col].pct_change() * 100

    # 이상변동 레이블: |원유 수익률| >= 2%
    data['이상변동'] = (data['원유_수익률'].abs() >= 2.0).astype(int)

    data = data.dropna()

    normal = (data['이상변동'] == 0).sum()
    anomaly = (data['이상변동'] == 1).sum()
    print(f"정상: {normal}일 ({normal/len(data)*100:.1f}%), 이상변동: {anomaly}일 ({anomaly/len(data)*100:.1f}%)")

    return data


# ============================================================
# STEP 2. 모델 정의
# ============================================================
def get_models():
    """비교할 5개 모델 반환"""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42
        ),
        'SVM': SVC(
            kernel='rbf', probability=True, random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='logloss', verbosity=0
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5
        ),
    }
    return models


# ============================================================
# STEP 3. TimeSeriesSplit 교차검증
# ============================================================
def evaluate_models(data):
    """
    TimeSeriesSplit(5-Fold)로 각 모델을 평가.

    1주차와 다른 점:
    - Random Split → TimeSeriesSplit (시간 순서 보장)
    - 1개 모델 → 5개 모델 비교
    - StandardScaler 적용 (SVM, KNN, Logistic 등은 스케일에 민감)
    """
    features = ['VIX', 'GPR지수', 'GPR_실제행동', 'GPR_위협',
                '금_수익률', '천연가스_수익률', '은_수익률']

    X = data[features].values
    y = data['이상변동'].values

    models = get_models()
    tscv = TimeSeriesSplit(n_splits=5)

    # 결과 저장
    results = {name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
               for name in models}

    # 마지막 Fold의 예측값 저장 (혼동 행렬, ROC용)
    last_fold_preds = {}

    print("\n" + "=" * 60)
    print("TimeSeriesSplit 5-Fold 교차검증")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 스케일링 (학습 데이터 기준으로 fit → 테스트에 transform)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n--- Fold {fold}: 학습 {len(train_idx)}일 → 테스트 {len(test_idx)}일 ---")

        for name, model in models.items():
            model_clone = clone_model(model)
            model_clone.fit(X_train_scaled, y_train)
            y_pred = model_clone.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            results[name]['accuracy'].append(acc)
            results[name]['precision'].append(prec)
            results[name]['recall'].append(rec)
            results[name]['f1'].append(f1)

            # 마지막 Fold 저장
            if fold == 5:
                y_proba = None
                if hasattr(model_clone, 'predict_proba'):
                    y_proba = model_clone.predict_proba(X_test_scaled)[:, 1]
                last_fold_preds[name] = {
                    'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba
                }

            print(f"  {name:22s} | Acc {acc:.3f} | Prec {prec:.3f} | Rec {rec:.3f} | F1 {f1:.3f}")

    return results, last_fold_preds


def clone_model(model):
    """모델 복제 (매 Fold마다 새로운 모델로 학습)"""
    from sklearn.base import clone
    return clone(model)


# ============================================================
# STEP 4. 결과 요약 & 비교표
# ============================================================
def summarize_results(results):
    """5-Fold 평균 성능 비교표"""
    print("\n" + "=" * 60)
    print("모델별 평균 성능 비교 (TimeSeriesSplit 5-Fold)")
    print("=" * 60)

    summary = {}
    print(f"{'모델':22s} | {'Accuracy':>8s} | {'Precision':>9s} | {'Recall':>8s} | {'F1':>8s}")
    print("-" * 65)

    for name, metrics in results.items():
        avg = {k: np.mean(v) for k, v in metrics.items()}
        summary[name] = avg
        print(f"{name:22s} | {avg['accuracy']:8.3f} | {avg['precision']:9.3f} | {avg['recall']:8.3f} | {avg['f1']:8.3f}")

    # 최적 모델 선정 (F1 기준)
    best_model = max(summary, key=lambda x: summary[x]['f1'])
    print(f"\n★ F1-Score 기준 최적 모델: {best_model} (F1: {summary[best_model]['f1']:.3f})")

    # Recall 기준도 확인
    best_recall = max(summary, key=lambda x: summary[x]['recall'])
    print(f"★ Recall 기준 최적 모델: {best_recall} (Recall: {summary[best_recall]['recall']:.3f})")

    return summary


# ============================================================
# STEP 5. 시각화
# ============================================================
def plot_comparison(summary):
    """모델별 성능 비교 막대 그래프"""
    models = list(summary.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [summary[m][metric] for m in models]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('모델', fontsize=13)
    ax.set_ylabel('점수', fontsize=13)
    ax.set_title('모델별 성능 비교 (TimeSeriesSplit 5-Fold 평균)', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'chart1_model_comparison.png'), dpi=150)
    plt.close()
    print("차트 저장: chart1_model_comparison.png")


def plot_roc_curves(last_fold_preds):
    """마지막 Fold 기준 ROC 커브 비교"""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    for (name, pred), color in zip(last_fold_preds.items(), colors):
        if pred['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(pred['y_test'], pred['y_proba'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC=0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC 커브 비교 (마지막 Fold)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'chart2_roc_curves.png'), dpi=150)
    plt.close()
    print("차트 저장: chart2_roc_curves.png")


def plot_confusion_matrices(last_fold_preds):
    """마지막 Fold 기준 혼동 행렬 비교"""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    cmap = plt.cm.Blues

    for ax, (name, pred) in zip(axes, last_fold_preds.items()):
        cm = confusion_matrix(pred['y_test'], pred['y_pred'])
        ax.imshow(cm, cmap=cmap, aspect='auto')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('예측', fontsize=11)
        ax.set_ylabel('실제', fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['정상', '이상'], fontsize=10)
        ax.set_yticklabels(['정상', '이상'], fontsize=10)

        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max()/2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=14, fontweight='bold', color=color)

    plt.suptitle('혼동 행렬 비교 (마지막 Fold)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'chart3_confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("차트 저장: chart3_confusion_matrices.png")


def plot_fold_detail(results):
    """Fold별 F1-Score 변화 추이 (모델 안정성 확인)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    folds = range(1, 6)

    for (name, metrics), color in zip(results.items(), colors):
        ax.plot(folds, metrics['f1'], 'o-', color=color, lw=2, markersize=8, label=name)

    ax.set_xlabel('Fold', fontsize=13)
    ax.set_ylabel('F1-Score', fontsize=13)
    ax.set_title('Fold별 F1-Score 변화 (모델 안정성)', fontsize=16, fontweight='bold')
    ax.set_xticks(folds)
    ax.set_xticklabels([f'Fold {i}' for i in folds], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'chart4_fold_f1.png'), dpi=150)
    plt.close()
    print("차트 저장: chart4_fold_f1.png")


# ============================================================
# STEP 6. 1주차 vs 2주차 비교
# ============================================================
def compare_with_week1(summary):
    """1주차 Random Forest (Random Split) vs 2주차 결과 비교"""
    print("\n" + "=" * 60)
    print("1주차 vs 2주차 비교 (Random Forest 기준)")
    print("=" * 60)

    # 1주차 결과 (보고서에서 가져옴)
    week1 = {'accuracy': 0.6827, 'precision': 0.6400, 'recall': 0.2832, 'f1': 0.3926}
    week2_rf = summary['Random Forest']

    print(f"{'지표':12s} | {'1주차 (Random Split)':>20s} | {'2주차 (TimeSeriesSplit)':>22s} | {'변화':>8s}")
    print("-" * 72)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        v1 = week1[metric]
        v2 = week2_rf[metric]
        diff = v2 - v1
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        label = {'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1-Score'}[metric]
        print(f"{label:12s} | {v1:20.3f} | {v2:22.3f} | {arrow} {abs(diff):.3f}")

    print("\n해석:")
    if week2_rf['recall'] < week1['recall']:
        print("  - Recall이 하락한 것은 TimeSeriesSplit이 더 현실적인 평가이기 때문")
        print("  - 1주차의 높은 성능은 미래 데이터 누출(data leakage)로 부풀려진 결과였을 가능성")
    else:
        print("  - Recall이 개선됨! TimeSeriesSplit 환경에서도 이상변동 감지 능력 향상")


# ============================================================
# 실행
# ============================================================
if __name__ == '__main__':
    # 1. 데이터 로드
    data = load_and_prepare()

    # 2. 모델 평가 (TimeSeriesSplit 5-Fold)
    results, last_fold_preds = evaluate_models(data)

    # 3. 결과 요약
    summary = summarize_results(results)

    # 4. 시각화
    plot_comparison(summary)
    plot_roc_curves(last_fold_preds)
    plot_confusion_matrices(last_fold_preds)
    plot_fold_detail(results)

    # 5. 1주차 대비 비교
    compare_with_week1(summary)

    print("\n✅ 2주차 분석 완료!")
    print(f"차트 저장 위치: {CHARTS_DIR}")
