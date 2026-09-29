"""
4주차 (제출④, 적용 관점): 4개 원자재 통합 + 투트랙 서비스
- 타겟: 원유(WTI), 금, 천연가스, 은 각각의 이상변동(|일일수익률| >= 임계)
- 동일 파이프라인 한 벌로 4개 자재 동시 학습
- 투트랙:
    (A) 위험경보 — Recall 우선 (낮은 threshold)
    (B) 매수타이밍 — Precision 우선 (높은 threshold)
- 평가: TimeSeriesSplit 5-Fold (week3 동일)
- Feature 엔지니어링: week3 베이스 유지 + 자재 간 cross feature
- 산출물: 4종 원자재별 성능, 투트랙 표, results.json
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "commodity_vix_gpr_data.xlsx")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# 자재별 설정
# ============================================================
TARGETS = {
    "원유(WTI)": {"label": "원유",     "anom_thr_pct": 2.0},
    "금":        {"label": "금",       "anom_thr_pct": 1.0},
    "천연가스":  {"label": "천연가스", "anom_thr_pct": 3.0},
    "은":        {"label": "은",       "anom_thr_pct": 2.0},
}

# 투트랙 임계값 (확률 threshold)
ALERT_THR = 0.30   # 위험경보: 낮춰서 Recall 확보
TIMING_THR = 0.70  # 매수타이밍: 높여서 Precision 확보

# week3 best XGB
XGB_PARAMS = dict(
    learning_rate=0.1, max_depth=6, n_estimators=200,
    tree_method="hist", n_jobs=1, random_state=42, verbosity=0,
    eval_metric="logloss",
)


# ============================================================
# Feature 엔지니어링 (자재별로 타겟이 달라지므로 함수화)
# ============================================================
def build_features(raw: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = raw.copy()
    for col in ["원유(WTI)", "금", "천연가스", "은"]:
        df[f"{col}_수익률"] = df[col].pct_change() * 100

    # Lag (타겟 수익률 + VIX/GPR)
    tgt_ret = f"{target_col}_수익률"
    for lag in [1, 3, 5]:
        df[f"VIX_lag{lag}"] = df["VIX"].shift(lag)
        df[f"GPR_lag{lag}"] = df["GPR지수"].shift(lag)
        df[f"타겟수익률_lag{lag}"] = df[tgt_ret].shift(lag)

    # 이동평균 (타겟 가격)
    df["타겟_MA5"] = df[target_col].rolling(5).mean()
    df["타겟_MA20"] = df[target_col].rolling(20).mean()
    df["MA_ratio"] = df["타겟_MA5"] / df["타겟_MA20"]
    df["VIX_MA5"] = df["VIX"].rolling(5).mean()

    # 변동성
    df["변동성_20d"] = df[tgt_ret].rolling(20).std()
    df["VIX_변동성_20d"] = df["VIX"].rolling(20).std()

    # RSI(14) on target
    delta = df[target_col].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI_14"] = 100 - 100 / (1 + rs)

    return df


def feature_cols(target_col: str) -> list:
    other = [c for c in ["원유(WTI)", "금", "천연가스", "은"] if c != target_col]
    cols = [
        "VIX", "GPR지수", "GPR_실제행동", "GPR_위협",
        *[f"{c}_수익률" for c in other],
        "VIX_lag1", "VIX_lag3", "VIX_lag5",
        "GPR_lag1", "GPR_lag3", "GPR_lag5",
        "타겟수익률_lag1", "타겟수익률_lag3", "타겟수익률_lag5",
        "타겟_MA5", "타겟_MA20", "MA_ratio", "VIX_MA5",
        "변동성_20d", "VIX_변동성_20d", "RSI_14",
    ]
    return cols


# ============================================================
# 평가 (TimeSeriesSplit + SMOTE + XGB)
# ============================================================
def evaluate_target(raw: pd.DataFrame, target_col: str, anom_pct: float):
    df = build_features(raw, target_col)
    df["이상변동"] = (df[f"{target_col}_수익률"].abs() >= anom_pct).astype(int)
    df = df.dropna()

    feats = feature_cols(target_col)
    X = df[feats].values
    y = df["이상변동"].values

    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    last_y_true, last_y_proba = None, None

    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # SMOTE (소수클래스 충분할 때만)
        if y_tr.sum() >= 6:
            try:
                X_tr_s, y_tr = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_tr_s, y_tr)
            except ValueError:
                pass

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr_s, y_tr)
        proba = model.predict_proba(X_te_s)[:, 1]
        pred = (proba >= 0.5).astype(int)

        m = {
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred, zero_division=0),
            "f1": f1_score(y_te, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_te, proba) if len(set(y_te)) > 1 else float("nan"),
        }
        fold_metrics.append(m)
        last_y_true, last_y_proba = y_te, proba

    avg = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}

    # 투트랙 (마지막 fold 기준)
    def dual(thr):
        p = (last_y_proba >= thr).astype(int)
        return {
            "threshold": thr,
            "precision": float(precision_score(last_y_true, p, zero_division=0)),
            "recall": float(recall_score(last_y_true, p, zero_division=0)),
            "f1": float(f1_score(last_y_true, p, zero_division=0)),
            "cm": confusion_matrix(last_y_true, p).tolist(),
        }

    return {
        "n_total": int(len(df)),
        "n_anomaly": int(y.sum()),
        "n_normal": int(len(y) - y.sum()),
        "anom_ratio": float(y.mean()),
        "fold_avg": avg,
        "fold_each": fold_metrics,
        "alert": dual(ALERT_THR),    # 위험경보 (Recall 우선)
        "timing": dual(TIMING_THR),  # 매수타이밍 (Precision 우선)
        "default_05": dual(0.5),
    }


# ============================================================
# Run
# ============================================================
def main():
    raw = pd.read_excel(DATA_PATH, index_col=0)
    print(f"[load] shape={raw.shape}, range={raw.index.min().date()} ~ {raw.index.max().date()}")

    results = {"targets": {}, "config": {
        "alert_threshold": ALERT_THR,
        "timing_threshold": TIMING_THR,
        "xgb": XGB_PARAMS,
    }}

    for col, cfg in TARGETS.items():
        print(f"\n=== {cfg['label']} (이상기준 ±{cfg['anom_thr_pct']}%) ===")
        r = evaluate_target(raw, col, cfg["anom_thr_pct"])
        results["targets"][cfg["label"]] = {
            "anom_thr_pct": cfg["anom_thr_pct"],
            **r,
        }
        print(f"  n={r['n_total']}, 이상={r['n_anomaly']} ({r['anom_ratio']:.1%})")
        a = r["fold_avg"]
        print(f"  [5-Fold avg] Acc={a['accuracy']:.3f}  Prec={a['precision']:.3f}  "
              f"Rec={a['recall']:.3f}  F1={a['f1']:.3f}  AUC={a['roc_auc']:.3f}")
        print(f"  [위험경보 thr={ALERT_THR}]  Rec={r['alert']['recall']:.3f}  Prec={r['alert']['precision']:.3f}")
        print(f"  [매수타이밍 thr={TIMING_THR}]  Prec={r['timing']['precision']:.3f}  Rec={r['timing']['recall']:.3f}")

    out_path = os.path.join(REPORTS_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n[saved] {out_path}")
    return results


if __name__ == "__main__":
    main()
