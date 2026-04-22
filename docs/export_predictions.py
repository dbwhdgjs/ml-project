"""
모델 학습 후 예측 결과를 JSON으로 저장.
이 JSON을 index.html이 읽어서 정적 페이지로 표시합니다.

실행: python docs/export_predictions.py
"""
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, '..', 'data')
OUT_PATH = os.path.join(_HERE, 'predictions.json')

latest_path = os.path.join(DATA_DIR, 'commodity_vix_gpr_latest.xlsx')
live_path = os.path.join(DATA_DIR, 'commodity_vix_gpr_live.xlsx')

data = pd.read_excel(latest_path, index_col=0)
live_df = pd.read_excel(live_path, index_col=0)

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

common = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
              tree_method='hist', n_jobs=1, random_state=42, verbosity=0)

m_anom = XGBClassifier(scale_pos_weight=1.8, eval_metric='logloss', **common)
m_anom.fit(X_train, y_anom[:split])

m_dir = XGBClassifier(eval_metric='logloss', **common)
m_dir.fit(X_train, y_dir[:split])

m_ret = XGBRegressor(**common)
m_ret.fit(X_train, y_ret[:split])

prob_anom = m_anom.predict_proba(X_test)[:, 1]
pred_dir = m_dir.predict(X_test)
pred_ret = m_ret.predict(X_test)

dates_test = [d.strftime('%Y-%m-%d') for d in data.index[split:]]
prices_test = data['원유(WTI)'].values[split:].tolist()

out = {
    'meta': {
        'data_rows': int(len(data)),
        'data_period_start': data.index[0].strftime('%Y.%m'),
        'data_period_end': data.index[-1].strftime('%Y.%m.%d'),
        'gpr_last': live_df.index[-1].strftime('%Y.%m.%d'),
        'data_last': data.index[-1].strftime('%Y.%m.%d'),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    },
    'latest': {
        'price': float(prices_test[-1]),
        'prob_anom': float(prob_anom[-1]),
        'pred_dir': int(pred_dir[-1]),
        'pred_ret': float(pred_ret[-1]),
    },
    'history': {
        'dates': dates_test,
        'prices': prices_test,
        'prob_anom': [float(x) for x in prob_anom],
        'pred_dir': [int(x) for x in pred_dir],
        'pred_ret': [float(x) for x in pred_ret],
    },
}

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"Saved: {OUT_PATH}")
print(f"  data_rows: {out['meta']['data_rows']}")
print(f"  latest price: ${out['latest']['price']:.2f}")
print(f"  prob_anom: {out['latest']['prob_anom']:.1%}")
print(f"  pred_dir: {'상승' if out['latest']['pred_dir']==1 else '하락'}")
print(f"  pred_ret: {out['latest']['pred_ret']:+.2f}%")
