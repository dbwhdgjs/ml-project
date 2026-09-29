"""
3주차: 반복 수정 (Feature 엔지니어링 + SMOTE + 하이퍼파라미터 튜닝)
- Feature: 7개 → 23개 (Lag, MA, 변동성, RSI 추가)
- 클래스 불균형: SMOTE 적용
- 하이퍼파라미터: GridSearchCV (XGBoost)
- 평가: TimeSeriesSplit 5-Fold
- 산출물: 비교표 + 5종 차트 + 결과 JSON
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트
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

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "commodity_vix_gpr_data.xlsx")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# 1. Feature 엔지니어링
# ============================================================
def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["원유_수익률"] = df["원유(WTI)"].pct_change() * 100
    for col in ["금", "천연가스", "은"]:
        df[f"{col}_수익률"] = df[col].pct_change() * 100

    # Lag
    for lag in [1, 3, 5]:
        df[f"VIX_lag{lag}"] = df["VIX"].shift(lag)
        df[f"GPR_lag{lag}"] = df["GPR지수"].shift(lag)
        df[f"원유수익률_lag{lag}"] = df["원유_수익률"].shift(lag)

    # 이동평균
    df["원유_MA5"] = df["원유(WTI)"].rolling(5).mean()
    df["원유_MA20"] = df["원유(WTI)"].rolling(20).mean()
    df["VIX_MA5"] = df["VIX"].rolling(5).mean()
    df["MA_ratio"] = df["원유_MA5"] / df["원유_MA20"]

    # 변동성
    df["변동성_20d"] = df["원유_수익률"].rolling(20).std()
    df["VIX_변동성_20d"] = df["VIX"].rolling(20).std()

    # RSI(14)
    delta = df["원유(WTI)"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # 레이블
    df["이상변동"] = (df["원유_수익률"].abs() >= 2.0).astype(int)

    return df.dropna()


FEATURE_BASIC = [
    "VIX", "GPR지수", "GPR_실제행동", "GPR_위협",
    "금_수익률", "천연가스_수익률", "은_수익률",
]
FEATURE_FULL = FEATURE_BASIC + [
    "VIX_lag1", "VIX_lag3", "VIX_lag5",
    "GPR_lag1", "GPR_lag3", "GPR_lag5",
    "원유수익률_lag1", "원유수익률_lag3", "원유수익률_lag5",
    "원유_MA5", "원유_MA20", "VIX_MA5", "MA_ratio",
    "변동성_20d", "VIX_변동성_20d", "RSI_14",
]


# ============================================================
# 2. 모델 정의
# ============================================================
def get_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 random_state=42, eval_metric="logloss", verbosity=0),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


# ============================================================
# 3. 평가 (TimeSeriesSplit 5-Fold + SMOTE)
# ============================================================
def evaluate(X, y, use_smote: bool, label: str):
    tscv = TimeSeriesSplit(n_splits=5)
    models = get_models()
    results = {n: {"accuracy": [], "precision": [], "recall": [], "f1": []} for n in models}
    last_fold = {}

    print(f"\n=== {label} ===")
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        ytr, yte = y[tr], y[te]

        if use_smote and ytr.sum() >= 6:
            sm = SMOTE(random_state=42, k_neighbors=min(5, int(ytr.sum()) - 1))
            Xtr, ytr = sm.fit_resample(Xtr, ytr)

        for name, m in models.items():
            mc = clone(m)
            mc.fit(Xtr, ytr)
            yp = mc.predict(Xte)
            results[name]["accuracy"].append(accuracy_score(yte, yp))
            results[name]["precision"].append(precision_score(yte, yp, zero_division=0))
            results[name]["recall"].append(recall_score(yte, yp, zero_division=0))
            results[name]["f1"].append(f1_score(yte, yp, zero_division=0))
            if fold == 5:
                yproba = mc.predict_proba(Xte)[:, 1] if hasattr(mc, "predict_proba") else None
                last_fold[name] = {"y_test": yte, "y_pred": yp, "y_proba": yproba}

    summary = {n: {k: float(np.mean(v)) for k, v in mt.items()} for n, mt in results.items()}
    return results, summary, last_fold


# ============================================================
# 4. XGBoost 하이퍼파라미터 튜닝
# ============================================================
def tune_xgb(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    tr, te = list(tscv.split(X))[-1]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr]); Xte = scaler.transform(X[te])
    ytr, yte = y[tr], y[te]

    sm = SMOTE(random_state=42)
    Xtr_s, ytr_s = sm.fit_resample(Xtr, ytr)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
    }
    base = XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0)
    grid = GridSearchCV(base, param_grid, scoring="f1", cv=TimeSeriesSplit(n_splits=3), n_jobs=-1)
    grid.fit(Xtr_s, ytr_s)

    best = grid.best_estimator_
    yp = best.predict(Xte)
    yproba = best.predict_proba(Xte)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(yte, yp)),
        "precision": float(precision_score(yte, yp, zero_division=0)),
        "recall": float(recall_score(yte, yp, zero_division=0)),
        "f1": float(f1_score(yte, yp, zero_division=0)),
        "roc_auc": float(roc_auc_score(yte, yproba)),
    }
    return best, grid.best_params_, metrics, (yte, yp, yproba)


# ============================================================
# 5. 시각화
# ============================================================
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

def chart_comparison(summary_w2, summary_w3, path):
    models = list(summary_w3.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, metric, lbl in zip(axes.flat, metrics, labels):
        v2 = [summary_w2[m][metric] for m in models]
        v3 = [summary_w3[m][metric] for m in models]
        ax.bar(x - width/2, v2, width, label="2주차", color="#90A4AE", alpha=0.85)
        ax.bar(x + width/2, v3, width, label="3주차", color="#1565C0", alpha=0.9)
        for i, (a, b) in enumerate(zip(v2, v3)):
            ax.text(i - width/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
            ax.text(i + width/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_title(lbl, fontsize=13, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, fontsize=9)
        ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
    plt.suptitle("2주차 vs 3주차 모델별 성능 비교", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def chart_recall_focus(summary_w2, summary_w3, path):
    models = list(summary_w3.keys())
    v2 = [summary_w2[m]["recall"] for m in models]
    v3 = [summary_w3[m]["recall"] for m in models]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - width/2, v2, width, label="2주차 (SMOTE 미적용)", color="#90A4AE")
    b2 = ax.bar(x + width/2, v3, width, label="3주차 (SMOTE + Lag/지표)", color="#E53935")
    for bar, v in zip(b1, v2):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    for bar, v in zip(b2, v3):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}", ha="center",
                fontsize=10, fontweight="bold")
    ax.set_title("Recall 변화: 2주차 → 3주차 (이상변동 감지율)", fontsize=15, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Recall", fontsize=12); ax.set_ylim(0, max(max(v3), max(v2)) * 1.3 + 0.05)
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def chart_fold_f1(results_w3, path):
    fig, ax = plt.subplots(figsize=(11, 6))
    folds = range(1, 6)
    for (name, m), c in zip(results_w3.items(), COLORS):
        ax.plot(folds, m["f1"], "o-", color=c, lw=2, markersize=8, label=name)
    ax.set_xlabel("Fold", fontsize=12); ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title("Fold별 F1-Score 변화 (3주차)", fontsize=15, fontweight="bold")
    ax.set_xticks(list(folds)); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def chart_feature_importance(model, features, path):
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(imp.index[::-1], imp.values[::-1], color="#FF9800", alpha=0.9)
    ax.set_title("Feature 중요도 Top 15 (튜닝된 XGBoost + SMOTE)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def chart_roc_curves(last_fold, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    for (name, p), c in zip(last_fold.items(), COLORS):
        if p["y_proba"] is None:
            continue
        fpr, tpr, _ = roc_curve(p["y_test"], p["y_proba"])
        a = auc_score(p["y_test"], p["y_proba"])
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC 커브 비교 (3주차, 마지막 Fold)", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def auc_score(y, p):
    try:
        return roc_auc_score(y, p)
    except Exception:
        return float("nan")


# ============================================================
# 메인
# ============================================================
def main():
    raw = pd.read_excel(DATA_PATH, index_col=0)
    df = build_features(raw)
    print(f"데이터: {len(df)}일, 정상 {(df['이상변동']==0).sum()} / 이상 {(df['이상변동']==1).sum()}")

    # 2주차 조건: 기본 Feature 7개, SMOTE 미적용
    X2 = df[FEATURE_BASIC].values
    # 3주차 조건: 확장 Feature 23개, SMOTE 적용
    X3 = df[FEATURE_FULL].values
    y = df["이상변동"].values

    res_w2, sum_w2, _ = evaluate(X2, y, use_smote=False, label="2주차 재현 (Feature 7개, SMOTE 없음)")
    res_w3, sum_w3, last_w3 = evaluate(X3, y, use_smote=True, label="3주차 (Feature 23개, SMOTE 적용)")

    # XGBoost 튜닝
    print("\n=== XGBoost 하이퍼파라미터 튜닝 ===")
    best_model, best_params, best_metrics, _ = tune_xgb(X3, y)
    print(f"최적 파라미터: {best_params}")
    print(f"튜닝 후 성능: {best_metrics}")

    # 차트 저장
    chart_comparison(sum_w2, sum_w3, os.path.join(CHARTS_DIR, "chart1_w2_vs_w3.png"))
    chart_recall_focus(sum_w2, sum_w3, os.path.join(CHARTS_DIR, "chart2_recall_focus.png"))
    chart_fold_f1(res_w3, os.path.join(CHARTS_DIR, "chart3_fold_f1.png"))
    chart_feature_importance(best_model, FEATURE_FULL, os.path.join(CHARTS_DIR, "chart4_feature_importance.png"))
    chart_roc_curves(last_w3, os.path.join(CHARTS_DIR, "chart5_roc.png"))

    # 결과 저장
    out = {
        "n_days": int(len(df)),
        "n_anomaly": int((df["이상변동"] == 1).sum()),
        "n_normal": int((df["이상변동"] == 0).sum()),
        "summary_w2": sum_w2,
        "summary_w3": sum_w3,
        "best_xgb_params": best_params,
        "best_xgb_metrics": best_metrics,
        "feature_importance_top10": pd.Series(
            best_model.feature_importances_, index=FEATURE_FULL
        ).sort_values(ascending=False).head(10).to_dict(),
    }
    with open(os.path.join(REPORTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n결과 저장: {os.path.join(REPORTS_DIR, 'results.json')}")
    print("차트 저장 완료")


if __name__ == "__main__":
    main()
