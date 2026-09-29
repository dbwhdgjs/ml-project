"""
4주차 API 자동수집 스크립트
- yfinance: 원유(CL=F), 금(GC=F), 천연가스(NG=F), 은(SI=F), VIX(^VIX)
- GPR: Caldara & Iacoviello 공개 엑셀 (gpr_daily.xlsx)
- 산출:
    data/commodity_vix_gpr_latest.xlsx  (최신본 갱신)
    data/update_log.json (실행 로그 append)

cron 예시 (매일 오전 9시):
    0 9 * * * cd /path/to/ML_project && /usr/bin/python3 week4/code/auto_update.py >> data/cron.log 2>&1
"""
import os, json
from datetime import datetime, timezone, timedelta
import pandas as pd

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJ_DIR, "data")
LATEST = os.path.join(DATA_DIR, "commodity_vix_gpr_latest.xlsx")
LIVE = os.path.join(DATA_DIR, "commodity_vix_gpr_live.xlsx")
LOG = os.path.join(DATA_DIR, "update_log.json")

KST = timezone(timedelta(hours=9))
TICKERS = {
    "원유(WTI)": "CL=F",
    "금": "GC=F",
    "천연가스": "NG=F",
    "은": "SI=F",
    "VIX": "^VIX",
}
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


def fetch_commodities(start="2020-01-01") -> pd.DataFrame:
    import yfinance as yf
    frames = {}
    for name, ticker in TICKERS.items():
        h = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if h.empty:
            print(f"  [warn] {name}({ticker}) empty")
            continue
        frames[name] = h["Close"].squeeze()
    df = pd.concat(frames, axis=1).ffill().dropna()
    df.index.name = "날짜"
    return df


def fetch_gpr() -> pd.DataFrame:
    try:
        g = pd.read_excel(GPR_URL)
    except Exception as e:
        print(f"  [warn] GPR fetch failed: {e}")
        return pd.DataFrame()
    # 컬럼 표준화
    rename = {"DATE": "날짜", "date": "날짜", "Date": "날짜",
              "GPRD": "GPR지수", "GPRD_ACT": "GPR_실제행동", "GPRD_THREAT": "GPR_위협"}
    g = g.rename(columns=rename)
    if "날짜" not in g.columns:
        return pd.DataFrame()
    g["날짜"] = pd.to_datetime(g["날짜"])
    g = g.set_index("날짜").sort_index()
    keep = [c for c in ["GPR지수", "GPR_실제행동", "GPR_위협"] if c in g.columns]
    return g[keep]


def merge_and_save(comm: pd.DataFrame, gpr: pd.DataFrame) -> pd.DataFrame:
    if gpr.empty:
        # 기존 latest의 GPR 컬럼 재활용
        if os.path.exists(LATEST):
            old = pd.read_excel(LATEST, index_col=0)
            gpr = old[[c for c in ["GPR지수", "GPR_실제행동", "GPR_위협"] if c in old.columns]]
    merged = comm.join(gpr, how="left").ffill().dropna()
    merged.to_excel(LATEST)
    if not gpr.empty:
        comm.join(gpr, how="left").to_excel(LIVE)
    return merged


def append_log(merged: pd.DataFrame, ok: bool):
    entry = {
        "run_at": datetime.now(KST).isoformat(),
        "rows": int(len(merged)) if ok else 0,
        "latest_last": str(merged.index[-1].date()) if ok else None,
        "updated": ok,
    }
    log = []
    if os.path.exists(LOG):
        try:
            log = json.load(open(LOG))
        except Exception:
            log = []
    log.append(entry)
    json.dump(log[-200:], open(LOG, "w"), ensure_ascii=False, indent=2)
    print(f"[log] {entry}")


def main():
    print(f"[start] {datetime.now(KST).isoformat()}")
    try:
        comm = fetch_commodities()
        print(f"  commodities: {comm.shape}, last={comm.index[-1].date()}")
        gpr = fetch_gpr()
        print(f"  gpr: {gpr.shape}")
        merged = merge_and_save(comm, gpr)
        print(f"  merged: {merged.shape}, saved to {LATEST}")
        append_log(merged, True)
    except Exception as e:
        print(f"[error] {e}")
        append_log(pd.DataFrame(), False)
        raise


if __name__ == "__main__":
    main()
