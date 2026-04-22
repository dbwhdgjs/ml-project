# 지정학적 리스크를 반영한 원자재 이상 변동 감지 및 가격 예측

머신러닝 학기 프로젝트

## 프로젝트 구조

```
ML_project/
├── data/                          # 데이터셋
│   ├── commodity_vix_data.xlsx    # 원자재 가격 + VIX (원본)
│   └── commodity_vix_gpr_data.xlsx # 원자재 + VIX + GPR (통합)
├── code/                          # 코드
│   └── week1_model.py             # 1주차 기본 모델
├── charts/                        # 시각화 차트
│   ├── chart1_prices.png          # 원자재 가격 추이
│   ├── chart2_risk.png            # VIX & GPR 리스크 지표
│   ├── chart3_corr.png            # 상관관계 히트맵
│   ├── chart4_anomaly.png         # 이상변동 산점도
│   ├── chart5_cm.png              # 혼동행렬
│   └── chart6_importance.png      # Feature 중요도
├── reports/                       # 보고서
│   ├── 1주차_보고서.pdf
│   └── 1주차_보고서.md
└── README.md
```

## 데이터

| 데이터 | 출처 | 기간 |
|--------|------|------|
| 원유(WTI), 금, 천연가스, 은 | Yahoo Finance (yfinance) | 2020-01 ~ 2026-03 |
| VIX (변동성 지수) | CBOE (Yahoo Finance) | 2020-01 ~ 2026-03 |
| GPR Index (지정학적 리스크) | Caldara & Iacoviello (FRB) | 2020-01 ~ 2026-03 |

## 제출 일정

| 제출 | 마감 | 내용 | 상태 |
|------|------|------|------|
| 제출 1 | 3/26 | 초기 설계 & 1차 실행 | 진행중 |
| 제출 2 | 4/9 | 모델 확장 & 비교 | 예정 |
| 제출 3 | 5/14 | 반복 수정 기록 | 예정 |
| 제출 4 | 6/4 | 적용 관점 정리 | 예정 |

## 환경

- Python 3.9
- Google Colab
- 주요 라이브러리: yfinance, pandas, numpy, scikit-learn, matplotlib
