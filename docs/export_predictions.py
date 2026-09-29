"""
모델 학습 후 4종 원자재 예측 결과를 JSON으로 저장.
이 JSON을 index.html이 읽어서 정적 페이지(최종 데모)로 표시합니다.

- Feature: week4 모델과 동일 (lag / 이동평균 / RSI / 변동성 + 자재 간 cross)
- 4종 원자재(원유·금·천연가스·은) 각각 이상변동/방향/수익률 예측
- 자재별 5-Fold 성능(투트랙 포함)은 week4 results.json에서 머지
- 출력: docs/predictions.json (targets 4종 구조)

실행: python docs/export_predictions.py
"""
import os
import json
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "data")
OUT_PATH = os.path.join(_HERE, "predictions.json")
RESULTS_PATH = os.path.join(_HERE, "..", "week4", "reports", "results.json")

LATEST_PATH = os.path.join(DATA_DIR, "commodity_vix_gpr_latest.xlsx")
LIVE_PATH = os.path.join(DATA_DIR, "commodity_vix_gpr_live.xlsx")

# 자재별 설정 (week4_model.py와 동일)
TARGETS = {
    "원유(WTI)": {"label": "원유",     "anom_thr_pct": 2.0, "decimals": 2},
    "금":        {"label": "금",       "anom_thr_pct": 1.0, "decimals": 2},
    "천연가스":  {"label": "천연가스", "anom_thr_pct": 3.0, "decimals": 3},
    "은":        {"label": "은",       "anom_thr_pct": 2.0, "decimals": 3},
}
TEST_DAYS = 30

XGB_PARAMS = dict(
    learning_rate=0.1, max_depth=6, n_estimators=200,
    tree_method="hist", n_jobs=1, random_state=42, verbosity=0,
    eval_metric="logloss",
)


def build_features(raw: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = raw.copy()
    for col in ["원유(WTI)", "금", "천연가스", "은"]:
        df[f"{col}_수익률"] = df[col].pct_change() * 100

    tgt_ret = f"{target_col}_수익률"
    for lag in [1, 3, 5]:
        df[f"VIX_lag{lag}"] = df["VIX"].shift(lag)
        df[f"GPR_lag{lag}"] = df["GPR지수"].shift(lag)
        df[f"타겟수익률_lag{lag}"] = df[tgt_ret].shift(lag)

    df["타겟_MA5"] = df[target_col].rolling(5).mean()
    df["타겟_MA20"] = df[target_col].rolling(20).mean()
    df["MA_ratio"] = df["타겟_MA5"] / df["타겟_MA20"]
    df["VIX_MA5"] = df["VIX"].rolling(5).mean()
    df["변동성_20d"] = df[tgt_ret].rolling(20).std()
    df["VIX_변동성_20d"] = df["VIX"].rolling(20).std()

    delta = df[target_col].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["RSI_14"] = 100 - 100 / (1 + rs)
    return df


def feature_cols(target_col: str) -> list:
    other = [c for c in ["원유(WTI)", "금", "천연가스", "은"] if c != target_col]
    return [
        "VIX", "GPR지수", "GPR_실제행동", "GPR_위협",
        *[f"{c}_수익률" for c in other],
        "VIX_lag1", "VIX_lag3", "VIX_lag5",
        "GPR_lag1", "GPR_lag3", "GPR_lag5",
        "타겟수익률_lag1", "타겟수익률_lag3", "타겟수익률_lag5",
        "타겟_MA5", "타겟_MA20", "MA_ratio", "VIX_MA5",
        "변동성_20d", "VIX_변동성_20d", "RSI_14",
    ]


def predict_target(raw: pd.DataFrame, target_col: str, anom_pct: float):
    df = build_features(raw, target_col)
    tgt_ret = f"{target_col}_수익률"
    df["이상변동"] = (df[tgt_ret].abs() >= anom_pct).astype(int)
    df["방향"] = (df[tgt_ret] > 0).astype(int)
    df = df.dropna()

    feats = feature_cols(target_col)
    X = df[feats].values
    y_anom = df["이상변동"].values
    y_dir = df["방향"].values
    y_ret = df[tgt_ret].values

    split = len(df) - TEST_DAYS
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[:split])
    X_te = scaler.transform(X[split:])

    # 이상변동 분류기 — SMOTE로 소수클래스 보강 (week4와 동일 철학)
    Xa, ya = X_tr, y_anom[:split]
    if ya.sum() >= 6:
        try:
            Xa, ya = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_tr, y_anom[:split])
        except ValueError:
            pass
    m_anom = XGBClassifier(**XGB_PARAMS)
    m_anom.fit(Xa, ya)

    m_dir = XGBClassifier(**XGB_PARAMS)
    m_dir.fit(X_tr, y_dir[:split])

    m_ret = XGBRegressor(
        learning_rate=0.1, max_depth=6, n_estimators=200,
        tree_method="hist", n_jobs=1, random_state=42, verbosity=0,
    )
    m_ret.fit(X_tr, y_ret[:split])

    prob_anom = m_anom.predict_proba(X_te)[:, 1]
    pred_dir = m_dir.predict(X_te)
    pred_ret = m_ret.predict(X_te)

    dates_test = [d.strftime("%Y-%m-%d") for d in df.index[split:]]
    prices_test = df[target_col].values[split:].astype(float).tolist()

    return {
        "latest": {
            "price": float(prices_test[-1]),
            "prob_anom": float(prob_anom[-1]),
            "pred_dir": int(pred_dir[-1]),
            "pred_ret": float(pred_ret[-1]),
        },
        "history": {
            "dates": dates_test,
            "prices": prices_test,
            "prob_anom": [float(x) for x in prob_anom],
            "pred_dir": [int(x) for x in pred_dir],
            "pred_ret": [float(x) for x in pred_ret],
        },
    }


def main():
    data = pd.read_excel(LATEST_PATH, index_col=0)
    live_df = pd.read_excel(LIVE_PATH, index_col=0)

    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    out = {
        "meta": {
            "data_rows": int(len(data)),
            "data_period_start": data.index[0].strftime("%Y.%m"),
            "data_period_end": data.index[-1].strftime("%Y.%m.%d"),
            "gpr_last": live_df.index[-1].strftime("%Y.%m.%d"),
            "data_last": data.index[-1].strftime("%Y.%m.%d"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "config": {
            "alert_threshold": results["config"]["alert_threshold"],
            "timing_threshold": results["config"]["timing_threshold"],
        },
        "targets": {},
    }

    for col, cfg in TARGETS.items():
        label = cfg["label"]
        print(f"=== {label} (이상기준 ±{cfg['anom_thr_pct']}%) ===")
        pr = predict_target(data, col, cfg["anom_thr_pct"])
        perf = results["targets"][label]
        out["targets"][label] = {
            "label": label,
            "anom_thr_pct": cfg["anom_thr_pct"],
            "decimals": cfg["decimals"],
            "latest": pr["latest"],
            "history": pr["history"],
            "perf": {
                "fold_avg": perf["fold_avg"],
                "alert": perf["alert"],
                "timing": perf["timing"],
                "anom_ratio": perf["anom_ratio"],
            },
        }
        l = pr["latest"]
        print(f"  price=${l['price']:.{cfg['decimals']}f}  prob_anom={l['prob_anom']:.1%}  "
              f"dir={'상승' if l['pred_dir']==1 else '하락'}  ret={l['pred_ret']:+.2f}%")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {OUT_PATH}  ({len(out['targets'])}종 원자재)")


if __name__ == "__main__":
    main()
