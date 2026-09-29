"""
4주차 차트 생성
1) 4종 원자재 성능 비교 (Acc/Prec/Rec/F1/AUC)
2) 투트랙: 위험경보 vs 매수타이밍 (Recall vs Precision)
3) Threshold sweep — 원유 기준 Precision/Recall 곡선
4) 자재별 이상변동 비율
5) week1 원유 단일 모델 vs week4 원유 비교
"""
import os, json
import numpy as np
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(CHARTS_DIR, exist_ok=True)

with open(os.path.join(REPORTS_DIR, "results.json"), encoding="utf-8") as f:
    R = json.load(f)

TARGETS = list(R["targets"].keys())
COLORS = {"원유": "#d62728", "금": "#bcbd22", "천연가스": "#1f77b4", "은": "#7f7f7f"}

# ============================================================
# 1) 4종 원자재 성능 비교 (그룹 막대)
# ============================================================
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
metric_labels = ["정확도", "정밀도", "재현율", "F1", "AUC"]
x = np.arange(len(metrics))
w = 0.2

fig, ax = plt.subplots(figsize=(10, 5.5))
for i, t in enumerate(TARGETS):
    vals = [R["targets"][t]["fold_avg"][m] for m in metrics]
    ax.bar(x + (i - 1.5) * w, vals, w, label=t, color=COLORS.get(t, None))
ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0, 1); ax.set_ylabel("점수")
ax.set_title("4종 원자재 통합 모델 성능 (TimeSeriesSplit 5-Fold 평균)", fontsize=12)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "chart1_4종비교.png"), dpi=130, bbox_inches="tight")
plt.close()

# ============================================================
# 2) 투트랙: 위험경보 vs 매수타이밍 (자재별 Recall/Precision)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = TARGETS

alert_rec = [R["targets"][t]["alert"]["recall"] for t in labels]
alert_pre = [R["targets"][t]["alert"]["precision"] for t in labels]
timing_rec = [R["targets"][t]["timing"]["recall"] for t in labels]
timing_pre = [R["targets"][t]["timing"]["precision"] for t in labels]

xi = np.arange(len(labels))
axes[0].bar(xi - 0.2, alert_rec, 0.4, label="재현율", color="#d62728")
axes[0].bar(xi + 0.2, alert_pre, 0.4, label="정밀도", color="#ff9896")
axes[0].set_xticks(xi); axes[0].set_xticklabels(labels)
axes[0].set_ylim(0, 1); axes[0].set_title(f"위험경보 (임계값 {R['config']['alert_threshold']}) — 재현율 우선")
axes[0].axhline(0.5, ls="--", c="gray", alpha=0.5)
axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(xi - 0.2, timing_pre, 0.4, label="정밀도", color="#1f77b4")
axes[1].bar(xi + 0.2, timing_rec, 0.4, label="재현율", color="#aec7e8")
axes[1].set_xticks(xi); axes[1].set_xticklabels(labels)
axes[1].set_ylim(0, 1); axes[1].set_title(f"매수타이밍 (임계값 {R['config']['timing_threshold']}) — 정밀도 우선")
axes[1].axhline(0.5, ls="--", c="gray", alpha=0.5)
axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("투트랙 서비스 — 임계값에 따른 트레이드오프", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "chart2_투트랙.png"), dpi=130, bbox_inches="tight")
plt.close()

# ============================================================
# 3) Threshold sweep — 각 자재 P/R 곡선 (default·alert·timing 점 표시)
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for t in TARGETS:
    pts = []
    for key in ["alert", "default_05", "timing"]:
        d = R["targets"][t][key]
        pts.append((d["recall"], d["precision"]))
    pts = sorted(pts)
    rec = [p[0] for p in pts]; pre = [p[1] for p in pts]
    ax.plot(rec, pre, "o-", label=t, color=COLORS.get(t), lw=2, markersize=8)

ax.set_xlabel("재현율 (Recall) — 놓치지 않기")
ax.set_ylabel("정밀도 (Precision) — 맞히기")
ax.set_title("임계값별 P/R 위치 — 좌상단(정밀도)=매수타이밍, 우하단(재현율)=위험경보")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(); ax.grid(alpha=0.3)
ax.text(0.78, 0.05, "← 위험경보\n(threshold ↓)", fontsize=9, color="gray")
ax.text(0.05, 0.80, "매수타이밍 →\n(threshold ↑)", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "chart3_PR곡선.png"), dpi=130, bbox_inches="tight")
plt.close()

# ============================================================
# 4) 자재별 이상변동 비율 & 기준
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
ratios = [R["targets"][t]["anom_ratio"] * 100 for t in TARGETS]
thrs = [R["targets"][t]["anom_thr_pct"] for t in TARGETS]
bars = ax.bar(TARGETS, ratios, color=[COLORS[t] for t in TARGETS])
for b, v, thr in zip(bars, ratios, thrs):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%\n(±{thr}%)",
            ha="center", fontsize=10)
ax.set_ylabel("이상변동일 비율 (%)")
ax.set_ylim(0, max(ratios) * 1.25)
ax.set_title("자재별 이상변동 정의와 발생 빈도")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "chart4_이상비율.png"), dpi=130, bbox_inches="tight")
plt.close()

# ============================================================
# 5) 진화 비교: week1 → week3 → week4 (원유 기준)
# ============================================================
# week1: Recall 28.3%, F1 39.3% (메모리 기록)
# week3: best XGB Recall 38.4%, F1 44.1%
# week4: 원유 fold_avg
w4 = R["targets"]["원유"]["fold_avg"]
data = {
    "1주차\n(RF 단일)":    {"recall": 0.283, "f1": 0.393, "precision": 0.640},
    "3주차\n(XGB 튜닝)":   {"recall": 0.384, "f1": 0.441, "precision": 0.519},
    "4주차\n(통합·투트랙)": {"recall": w4["recall"], "f1": w4["f1"], "precision": w4["precision"]},
}
ms = ["precision", "recall", "f1"]
ms_label = ["정밀도", "재현율", "F1"]
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(data))
w = 0.25
for i, (m, lab) in enumerate(zip(ms, ms_label)):
    vals = [data[k][m] for k in data]
    ax.bar(x + (i - 1) * w, vals, w, label=lab)
ax.set_xticks(x); ax.set_xticklabels(list(data.keys()))
ax.set_ylim(0, 0.8); ax.set_ylabel("점수")
ax.set_title("원유 모델 성능 진화 (1주차 → 4주차)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "chart5_진화비교.png"), dpi=130, bbox_inches="tight")
plt.close()

print("Saved charts:")
for f in sorted(os.listdir(CHARTS_DIR)):
    print(" ", f)
