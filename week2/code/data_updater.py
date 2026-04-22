"""
원자재/VIX/GPR 데이터 자동 갱신 스크립트

사용법:
    python data_updater.py --update         # 한 번만 갱신 (지금 즉시)
    python data_updater.py --watch          # 매일 KST 06~10시 사이 30분 간격으로 자동 갱신
    python data_updater.py --analyze        # 누적 로그 기반 최적 갱신 시각 분석

생성 파일:
    ../../data/commodity_vix_gpr_live.xlsx    (GPR 있는 날까지 inner join)
    ../../data/commodity_vix_gpr_latest.xlsx  (GPR forward-fill + 원자재/VIX 최신)
    ../../data/update_log.json                (갱신 이력)

데이터 출처:
    yfinance: CL=F, GC=F, NG=F, SI=F, ^VIX
    GPR: matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


# ---------- 설정 ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # ML_project/
DATA_DIR = BASE_DIR / 'data'
LIVE_PATH = DATA_DIR / 'commodity_vix_gpr_live.xlsx'
LATEST_PATH = DATA_DIR / 'commodity_vix_gpr_latest.xlsx'
LOG_PATH = DATA_DIR / 'update_log.json'

KST = ZoneInfo('Asia/Seoul')

TICKERS = {
    'CL=F': '원유(WTI)',
    'GC=F': '금',
    'NG=F': '천연가스',
    'SI=F': '은',
    '^VIX': 'VIX',
}
GPR_URL = 'https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls'
GPR_COLS = ['GPR지수', 'GPR_실제행동', 'GPR_위협']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


# ---------- 데이터 수집 ----------
def fetch_prices(start='2020-01-01'):
    end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    frames = []
    for ticker, name in TICKERS.items():
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f'yfinance에서 {ticker} 데이터 수집 실패')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        frames.append(df[['Close']].rename(columns={'Close': name}))
    data = pd.concat(frames, axis=1)
    data.index.name = '날짜'
    return data.dropna()


def fetch_gpr():
    gpr = pd.read_excel(GPR_URL)
    gpr['날짜'] = pd.to_datetime(gpr['date'])
    gpr = gpr[['날짜', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT']]
    gpr.columns = ['날짜'] + GPR_COLS
    gpr = gpr[gpr['날짜'] >= '2020-01-01'].set_index('날짜')
    return gpr


def build_datasets():
    prices = fetch_prices()
    gpr = fetch_gpr()

    # live: GPR이 실제로 있는 날까지만 (inner join)
    live = prices.join(gpr, how='inner')

    # latest: 원자재/VIX 전체 + GPR forward-fill
    latest = prices.join(gpr, how='left')
    latest[GPR_COLS] = latest[GPR_COLS].ffill()
    latest = latest.dropna()

    return live, latest


# ---------- 로그 ----------
def save_log(entry):
    log = []
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, 'r') as f:
                log = json.load(f)
        except json.JSONDecodeError:
            log = []
    log.append(entry)
    log = log[-500:]
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def last_date(path):
    if not path.exists():
        return None
    df = pd.read_excel(path, index_col=0)
    return df.index[-1] if len(df) else None


# ---------- 주요 동작 ----------
def do_update():
    prev_live = last_date(LIVE_PATH)
    prev_latest = last_date(LATEST_PATH)
    logging.info(f'기존 live 마지막: {prev_live}, latest 마지막: {prev_latest}')

    live, latest = build_datasets()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    live.to_excel(LIVE_PATH)
    latest.to_excel(LATEST_PATH)

    new_live = live.index[-1]
    new_latest = latest.index[-1]
    updated = (prev_latest is None) or (new_latest > prev_latest)

    logging.info(f'저장 완료 → live: {new_live.date()}, latest: {new_latest.date()} (갱신: {updated})')

    save_log({
        'run_at': datetime.now(KST).isoformat(),
        'live_last': str(new_live.date()),
        'latest_last': str(new_latest.date()),
        'updated': bool(updated),
    })
    return prev_latest, new_latest, updated


def analyze_optimal_time():
    if not LOG_PATH.exists():
        logging.info('update_log.json 없음. --watch 실행 후 재시도.')
        return
    with open(LOG_PATH, 'r') as f:
        log = json.load(f)

    updated = [e for e in log if e.get('updated')]
    if not updated:
        logging.info('성공적인 갱신 로그가 아직 없음.')
        return

    times = []
    for e in updated:
        t = datetime.fromisoformat(e['run_at'])
        times.append(t.hour + t.minute / 60 + t.second / 3600)

    avg = sum(times) / len(times)
    h, m = int(avg), int((avg - int(avg)) * 60)
    logging.info(f'성공 갱신 건수: {len(updated)} / 전체 로그: {len(log)}')
    logging.info(f'평균 최적 시각: KST {h:02d}:{m:02d}')
    logging.info(f'가장 빠름: KST {min(times):.2f}h  ·  가장 늦음: KST {max(times):.2f}h')

    # 추천: 평균보다 15분 빠르게 체크 시작
    rec_h = max(0, h - (1 if m < 15 else 0))
    rec_m = (m - 15) % 60
    logging.info(f'→ 추천 watch 시작 시각: KST {rec_h:02d}:{rec_m:02d} (평균 -15분)')


def do_watch(start_hour=6, end_hour=10, interval_min=30):
    logging.info(f'감시 모드: 매일 KST {start_hour:02d}:00 ~ {end_hour:02d}:00, {interval_min}분 간격')
    updated_date = None

    while True:
        now = datetime.now(KST)
        today = now.date()

        # 오늘 이미 갱신 성공 → 내일 start_hour까지 대기
        if updated_date == today:
            next_run = datetime.combine(today + timedelta(days=1), dtime(start_hour), tzinfo=KST)
            sleep_s = (next_run - now).total_seconds()
            logging.info(f'오늘 갱신 완료. 다음 실행: {next_run.strftime("%Y-%m-%d %H:%M")} (+{sleep_s/3600:.1f}h)')
            time.sleep(max(sleep_s, 60))
            continue

        # 아직 윈도우 전 → start_hour까지 대기
        if now.hour < start_hour:
            next_run = datetime.combine(today, dtime(start_hour), tzinfo=KST)
            sleep_s = (next_run - now).total_seconds()
            logging.info(f'{start_hour}시까지 대기 (+{sleep_s/60:.0f}분)')
            time.sleep(max(sleep_s, 60))
            continue

        # 윈도우 종료 → 내일로
        if now.hour >= end_hour:
            next_run = datetime.combine(today + timedelta(days=1), dtime(start_hour), tzinfo=KST)
            sleep_s = (next_run - now).total_seconds()
            logging.info(f'오늘 윈도우 종료. 다음 실행: {next_run.strftime("%Y-%m-%d %H:%M")}')
            time.sleep(max(sleep_s, 60))
            continue

        # 갱신 시도
        try:
            _, _, updated = do_update()
            if updated:
                logging.info('✓ 새 데이터 반영 성공')
                updated_date = today
                continue
            logging.info(f'새 데이터 없음. {interval_min}분 후 재시도.')
        except Exception as e:
            logging.exception(f'갱신 실패: {e}')

        time.sleep(interval_min * 60)


# ---------- 엔트리 포인트 ----------
def main():
    parser = argparse.ArgumentParser(description='원자재/VIX/GPR 데이터 자동 갱신')
    parser.add_argument('--update', action='store_true', help='지금 한 번만 갱신')
    parser.add_argument('--watch', action='store_true', help='매일 자동 갱신 (데몬)')
    parser.add_argument('--analyze', action='store_true', help='최적 갱신 시각 분석')
    parser.add_argument('--start-hour', type=int, default=6, help='watch 시작 시각 (KST, 기본 6)')
    parser.add_argument('--end-hour', type=int, default=10, help='watch 종료 시각 (KST, 기본 10)')
    parser.add_argument('--interval-min', type=int, default=30, help='watch 재시도 간격 (분, 기본 30)')
    args = parser.parse_args()

    if args.update:
        do_update()
    elif args.watch:
        do_watch(args.start_hour, args.end_hour, args.interval_min)
    elif args.analyze:
        analyze_optimal_time()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
