"""4주차 보고서 PDF 생성 — 제출④ (적용 관점)
4개 원자재 통합 + 투트랙 서비스 + B2C/B2B 설계 + 자동화 파이프라인
"""
import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader

# ── 폰트 ──
font_registered = False
for fp in [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]:
    if os.path.exists(fp):
        try:
            pdfmetrics.registerFont(TTFont("Korean", fp))
            font_registered = True
            break
        except Exception:
            continue
FONT = "Korean" if font_registered else "Helvetica"

S = {
    "title": ParagraphStyle("title", fontName=FONT, fontSize=26, leading=34,
                            alignment=TA_CENTER, spaceAfter=6*mm,
                            textColor=colors.HexColor("#1a1a2e")),
    "subtitle": ParagraphStyle("subtitle", fontName=FONT, fontSize=14, leading=20,
                               alignment=TA_CENTER, spaceAfter=3*mm,
                               textColor=colors.HexColor("#444444")),
    "h1": ParagraphStyle("h1", fontName=FONT, fontSize=18, leading=24,
                         spaceBefore=8*mm, spaceAfter=4*mm,
                         textColor=colors.HexColor("#16213e")),
    "h2": ParagraphStyle("h2", fontName=FONT, fontSize=14, leading=20,
                         spaceBefore=5*mm, spaceAfter=3*mm,
                         textColor=colors.HexColor("#0f3460")),
    "body": ParagraphStyle("body", fontName=FONT, fontSize=11.5, leading=18,
                           spaceAfter=2*mm, textColor=colors.HexColor("#333333")),
    "bullet": ParagraphStyle("bullet", fontName=FONT, fontSize=11.5, leading=18,
                             spaceAfter=1.5*mm, leftIndent=8*mm, bulletIndent=3*mm,
                             textColor=colors.HexColor("#333333")),
    "callout": ParagraphStyle("callout", fontName=FONT, fontSize=11.5, leading=18,
                              spaceAfter=2*mm, leftIndent=4*mm, rightIndent=4*mm,
                              textColor=colors.HexColor("#1a1a2e"),
                              backColor=colors.HexColor("#f0f4ff"),
                              borderPadding=6),
    "caption": ParagraphStyle("caption", fontName=FONT, fontSize=10, leading=14,
                              alignment=TA_CENTER, spaceAfter=2*mm,
                              textColor=colors.HexColor("#666666")),
}


def make_table(data, col_widths=None, head_bg="#16213e"):
    body = ParagraphStyle("tcell", fontName=FONT, fontSize=10.5, leading=14,
                          textColor=colors.HexColor("#333333"))
    head = ParagraphStyle("thcell", fontName=FONT, fontSize=10.5, leading=14,
                          textColor=colors.white)
    fmt = []
    for i, row in enumerate(data):
        fmt.append([Paragraph(str(c), head if i == 0 else body) for c in row])
    t = Table(fmt, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(head_bg)),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def add_image(elements, path, max_width=160*mm, max_height=180*mm, caption=None):
    if not os.path.exists(path):
        return
    ir = ImageReader(path)
    iw, ih = ir.getSize()
    ratio = iw / ih
    w = max_width
    h = w / ratio
    if h > max_height:
        h = max_height
        w = h * ratio
    img = Image(path, width=w, height=h)
    img.hAlign = "CENTER"
    elements.append(img)
    if caption:
        elements.append(Paragraph(caption, S["caption"]))


def p(text): return Paragraph(text, S["body"])
def b(text): return Paragraph(f"• {text}", S["bullet"])
def callout(text): return Paragraph(text, S["callout"])


def build():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    charts_dir = os.path.join(os.path.dirname(base_dir), "charts")
    out = os.path.join(base_dir, "4주차_보고서.pdf")

    with open(os.path.join(base_dir, "results.json"), encoding="utf-8") as f:
        R = json.load(f)
    cfg = R["config"]
    T = R["targets"]
    TNAMES = list(T.keys())

    E = []

    # ───────── 표지 ─────────
    E += [
        Spacer(1, 40*mm),
        Paragraph("머신러닝 학기 프로젝트", S["subtitle"]),
        Paragraph("제출 ④ — 4주차 보고서", S["title"]),
        Paragraph("적용 관점: 4개 원자재 통합 · 투트랙 서비스 · 자동화 파이프라인", S["subtitle"]),
        Spacer(1, 20*mm),
        Paragraph("지정학적 리스크 기반 원자재 이상 변동 감지 및 가격 예측", S["subtitle"]),
        Spacer(1, 40*mm),
        Paragraph("임태후 · 유종헌", S["subtitle"]),
        Paragraph("제출일: 2026-06-04", S["subtitle"]),
        PageBreak(),
    ]

    # ───────── 1. 4주차 목표 ─────────
    E += [Paragraph("1. 4주차 목표 — 적용 관점 4대 질문", S["h1"])]
    E += [p(
        "“이 모델을 실제로 사용한다면?”이라는 관점에서 다음 네 가지 질문에 답한다. "
        "본 보고서의 4~7장은 각 질문에 정면으로 대응하는 구조로 작성되었다."
    )]
    qmap = [
        ["#", "질문", "본 보고서 대응 섹션"],
        ["①", "<b>실제 사용 시나리오</b> — 누가, 어떤 상황에서 쓰는가?", "4장"],
        ["②", "<b>데이터 규모 적절성</b> — 현실에서 충분한 데이터를 확보할 수 있는가?", "5장"],
        ["③", "<b>비용 및 리스크</b> — 오분류가 발생하면 어떤 위험이 있는가?", "6장"],
        ["④", "<b>모델 한계와 일반화</b> — 다른 환경에서도 일반화할 수 있는가?", "7장"],
    ]
    E += [make_table(qmap, col_widths=[10*mm, 130*mm, 30*mm])]
    E += [Spacer(1, 4*mm)]
    E += [p(
        "이를 위한 기술적 토대로 ① 4개 원자재 통합 모델, ② 투트랙 서비스 구조, "
        "③ 자동화·서비스화 인프라를 함께 구축했으며, 2~3장에서 먼저 다룬다."
    )]

    E += [PageBreak()]

    # ───────── 2. 4개 원자재 통합 ─────────
    E += [Paragraph("2. 4개 원자재 통합 모델", S["h1"])]
    E += [Paragraph("2.1. 데이터와 자재별 이상기준", S["h2"])]
    E += [p(
        f"원유(WTI) · 금 · 천연가스 · 은 4개 원자재의 일별 가격에 "
        f"VIX(공포지수) 및 GPR(지정학적 리스크) 지수를 결합한 통합 데이터셋을 사용했다. "
        f"분석 기간은 2020-01-02 ~ 2026-03-16이며 총 1,558 거래일이다. "
        f"자재마다 변동성 수준이 다르기 때문에 ‘이상변동’의 정의도 자재별로 다르게 설정했다."
    )]
    anom_tbl = [["원자재", "이상기준 (일일 |수익률|)", "이상 발생 비율", "전체 일수"]]
    for n in TNAMES:
        d = T[n]
        anom_tbl.append([
            n, f"≥ ±{d['anom_thr_pct']}%",
            f"{d['anom_ratio']*100:.1f}% ({d['n_anomaly']:,}일)",
            f"{d['n_total']:,}일",
        ])
    E += [make_table(anom_tbl, col_widths=[28*mm, 50*mm, 50*mm, 36*mm])]

    add_image(E, os.path.join(charts_dir, "chart4_이상비율.png"),
              caption="[그림 1] 자재별 이상변동 정의와 발생 빈도 — 변동성이 큰 천연가스(±3%)도 절반 가까이 이상으로 분류")

    E += [PageBreak(), Paragraph("2.2. 모델 구성", S["h2"])]
    xgb = cfg["xgb"]
    E += [b(f"<b>알고리즘</b>: XGBoost ({xgb['n_estimators']} trees, max_depth={xgb['max_depth']}, lr={xgb['learning_rate']}) — 3주차 GridSearchCV 최적값 그대로 적용")]
    E += [b("<b>Feature 23개</b>: VIX·GPR 3종 + 타 자재 일일수익률 3개 + Lag(1/3/5) + 이동평균(MA5/MA20/비율) + 20일 변동성 + RSI(14)")]
    E += [b("<b>클래스 불균형</b>: SMOTE (k=5) 적용, 단 학습 fold의 양성표본이 6 미만일 경우 자동 skip")]
    E += [b("<b>검증</b>: TimeSeriesSplit 5-Fold (미래 데이터 누수 차단)")]
    E += [b("<b>타겟 변경 시 동일 코드 한 벌</b>로 4개 자재 모두 학습 — 운영·유지보수 측면에서 유리")]

    E += [Paragraph("2.3. 자재별 성능 (5-Fold 평균)", S["h2"])]
    perf_tbl = [["원자재", "정확도", "정밀도", "재현율", "F1", "AUC"]]
    for n in TNAMES:
        m = T[n]["fold_avg"]
        perf_tbl.append([
            n,
            f"{m['accuracy']*100:.1f}%",
            f"{m['precision']*100:.1f}%",
            f"{m['recall']*100:.1f}%",
            f"{m['f1']:.3f}",
            f"{m['roc_auc']:.3f}",
        ])
    E += [make_table(perf_tbl, col_widths=[28*mm, 25*mm, 25*mm, 25*mm, 25*mm, 25*mm])]

    add_image(E, os.path.join(charts_dir, "chart1_4종비교.png"),
              caption="[그림 2] 4종 원자재 통합 모델 성능 비교 — 은·금이 AUC 0.7대로 우수, 천연가스는 변동성 과다로 정확도 낮음")

    E += [callout(
        "🔍 <b>관찰</b>: 같은 모델·같은 파이프라인이라도 자재마다 성능이 크게 다르다. "
        "은과 금은 AUC 0.7대로 안정적이지만 천연가스는 변동성 자체가 커 0.57로 낮다. "
        "이는 향후 자재별로 임계값을 별도 튜닝해야 할 근거가 된다."
    )]

    E += [PageBreak()]

    # ───────── 3. 투트랙 서비스 ─────────
    E += [Paragraph("3. 투트랙 서비스 — 한 모델, 두 모드", S["h1"])]
    E += [Paragraph("3.1. 왜 투트랙인가", S["h2"])]
    E += [p(
        "중간 발표 시 교수님께서 ‘이상변동을 놓치는 것(제2종 오류)이 잘못된 경보(제1종 오류)보다 "
        "투자자에게 훨씬 치명적’이라는 점을 지적하셨다. 행동경제학의 <b>손실회피 효과</b>에 따르면 "
        "사람은 같은 금액의 손실을 이익의 약 2배로 무겁게 느낀다. 따라서 위험경보는 ‘조금 헛알람이 있더라도 "
        "놓치지 않는 것’이 우선되어야 한다."
    )]
    E += [p(
        "반대로 매매 추천은 ‘하기로 했으면 맞아야’ 한다. 잘못된 매수 추천은 곧바로 손실이고 "
        "사용자 신뢰를 잃는다. 즉 두 기능은 평가 지표가 본질적으로 다르므로 "
        "<b>같은 모델·같은 데이터에서 확률 임계값만 다르게 적용</b>하여 두 서비스를 동시에 운영한다."
    )]
    track_tbl = [
        ["모드", "임계값", "우선 지표", "용도", "메시지 예시"],
        ["위험경보", f"확률 ≥ {cfg['alert_threshold']}",
         "재현율(Recall)", "내일 큰 폭 변동 가능성 알림",
         "“내일 원유 변동성 주의”"],
        ["매수타이밍", f"확률 ≥ {cfg['timing_threshold']}",
         "정밀도(Precision)", "확신할 때만 매매 추천",
         "“상승 가능성이 충분히 높습니다”"],
    ]
    E += [make_table(track_tbl, col_widths=[26*mm, 28*mm, 30*mm, 40*mm, 36*mm])]

    E += [Paragraph("3.2. 실제 결과", S["h2"])]
    dual_tbl = [["원자재", "위험경보 재현율", "위험경보 정밀도", "매수타이밍 정밀도", "매수타이밍 재현율"]]
    for n in TNAMES:
        a = T[n]["alert"]; t = T[n]["timing"]
        dual_tbl.append([
            n,
            f"{a['recall']*100:.1f}%",
            f"{a['precision']*100:.1f}%",
            f"{t['precision']*100:.1f}%",
            f"{t['recall']*100:.1f}%",
        ])
    E += [make_table(dual_tbl, col_widths=[24*mm, 32*mm, 32*mm, 34*mm, 32*mm])]

    add_image(E, os.path.join(charts_dir, "chart2_투트랙.png"),
              caption="[그림 3] 투트랙 모드별 트레이드오프 — 임계값을 낮추면 재현율 ↑, 높이면 정밀도 ↑")

    add_image(E, os.path.join(charts_dir, "chart3_PR곡선.png"),
              caption="[그림 4] 임계값별 정밀도-재현율 위치 — 좌상단은 매수타이밍, 우하단은 위험경보 영역")

    E += [callout(
        "💡 <b>핵심 결과</b>: 천연가스는 위험경보 모드에서 재현율 75.0%에 도달하여 "
        "이상변동의 3/4을 미리 잡아낸다. 반대로 금·은의 매수타이밍 모드는 정밀도 66~67%로 "
        "‘추천이 나오면 약 2/3는 실제로 맞는’ 수준의 신뢰도를 보인다. "
        "한 모델로 ‘방어용’과 ‘공격용’ 두 서비스를 동시에 운영 가능함이 데이터로 확인되었다."
    )]

    E += [PageBreak()]

    # ════════════════════════════════════════════════════════════
    # 적용 관점 4대 질문 (5~8장)
    # ════════════════════════════════════════════════════════════

    # ───────── 4. [질문①] 실제 사용 시나리오 ─────────
    E += [Paragraph("4. [적용 관점 ①] 실제 사용 시나리오", S["h1"])]
    E += [p(
        "<b>핵심 질문</b>: 누가, 어떤 상황에서 이 모델을 사용할 수 있는가?"
    )]
    E += [Paragraph("4.1. 두 종류의 사용자 (B2C / B2B)", S["h2"])]
    E += [p(
        "중간발표 피드백 중 ‘B2C와 B2B는 지향점이 다르므로 명확히 구분해야 한다’는 지적에 따라, "
        "본 모델의 잠재 사용자를 두 시장으로 분리해 시나리오를 설계했다."
    )]
    seg_tbl = [
        ["구분", "B2C — 개인 투자자", "B2B — 기관/원자재 구매 기업"],
        ["타깃 모드", "위험경보 + 매수타이밍 보조",
         "위험경보 중심 (헷지·원가 관리)"],
        ["핵심 가치",
         "큰 손실 회피, 심리적 안정",
         "구매 시점 최적화, 가격 리스크 관리"],
        ["UI/UX", "푸시 알림, 일상 언어",
         "관리자 대시보드, 보고서·KPI 중심"],
        ["설득 대상", "사용자 본인 (다수)",
         "구매·재무 책임자 (소수)"],
        ["요구사항", "쉬운 설명, 친근한 톤, 무료 체험",
         "데이터 정합성, SLA, 감사 로그, API"],
    ]
    E += [make_table(seg_tbl, col_widths=[26*mm, 64*mm, 70*mm])]

    E += [Paragraph("4.2. 구체적 사용 시나리오 3종", S["h2"])]
    E += [b(
        "<b>시나리오 A · 개인 투자자 (B2C)</b><br/>"
        "원유 ETF에 적립식으로 투자 중인 30대 직장인. 매일 아침 출근길에 "
        "‘오늘 원유 시장 큰 변동성 주의’ 푸시를 받고, 추가 매수 일정을 하루 미룬다. "
        "→ 사용 모드: <b>위험경보</b> (재현율 우선). 헛알람이 있어도 손실 회피가 우선."
    )]
    E += [b(
        "<b>시나리오 B · 항공/물류 기업 원가관리팀 (B2B)</b><br/>"
        "원유 가격에 연료비가 직접 연동되는 기업이 월간 헷지 비중을 결정. "
        "위험경보 일수가 평월 대비 30% 이상 증가하면 자동으로 헷지 비중을 상향하는 룰을 사내 ERP와 연동. "
        "→ 사용 모드: <b>위험경보</b> 집계 지표 + 자동 트리거."
    )]
    E += [b(
        "<b>시나리오 C · 자산운용사 트레이딩 데스크 (B2B)</b><br/>"
        "여러 원자재를 동시에 모니터링. 4개 자재의 위험 확률을 한 화면에서 비교하고, "
        "매수타이밍 신호(확률 0.7↑)가 뜬 자재만 발주 검토. "
        "→ 사용 모드: <b>매수타이밍</b> (정밀도 우선). 추천이 나오면 실제로 맞아야 함."
    )]
    E += [callout(
        "📌 본 모델은 ‘하나의 수익 시그널’이 아니라 <b>의사결정 보조 도구</b>로 위치한다. "
        "최종 매매 결정은 사용자가 내리며, 모델은 그 결정의 근거 한 가지를 추가로 제공한다."
    )]

    E += [PageBreak()]

    # ───────── 5. [질문②] 데이터 규모 적절성 ─────────
    E += [Paragraph("5. [적용 관점 ②] 데이터 규모 적절성", S["h1"])]
    E += [p(
        "<b>핵심 질문</b>: 현실 환경에서 충분한 데이터를 확보할 수 있는가?"
    )]
    E += [Paragraph("5.1. 현재 데이터 규모 점검", S["h2"])]
    data_tbl = [
        ["항목", "현재 확보량", "평가"],
        ["관측 일수", "1,558 거래일 (≈ 6년 3개월)", "○ 통상적 ML 분류 기준 충분"],
        ["이상변동일 (원유 ±2%)", "547일", "○ 양성표본 비율 35.6% — 균형 양호"],
        ["이상변동일 (금 ±1%)", "447일", "○ 29.1% — SMOTE 없이도 학습 가능"],
        ["이상변동일 (천연가스 ±3%)", "713일", "○ 46.4% — 거의 균형"],
        ["이상변동일 (은 ±2%)", "408일", "△ 26.5% — 가장 적음, SMOTE 효과 큼"],
        ["Feature 수", "23개 (가격·VIX·GPR·파생)", "○ 자료 다양성 확보"],
        ["시장 국면", "코로나 폭락·전쟁(우-러)·금리 인상 다수 포함", "◎ 위기 국면 학습 적합"],
    ]
    E += [make_table(data_tbl, col_widths=[55*mm, 60*mm, 45*mm])]

    E += [Paragraph("5.2. 데이터 확보의 지속 가능성", S["h2"])]
    E += [b(
        "<b>가격·VIX</b>: yfinance API 무료, 일별 자동 갱신 가능 (auto_update.py 구현 완료). "
        "운영 비용 0원, 24/7 자동 수집."
    )]
    E += [b(
        "<b>GPR 지수</b>: Caldara·Iacoviello(FRB)가 매일 갱신해 공개 엑셀로 배포. "
        "주말/공휴일 지연 1~2일 존재 — 실시간성에는 약한 제약. 비용 0원."
    )]
    E += [b(
        "<b>실제 서비스 운영 시 추가 확보 가능 데이터</b>: 거래량, 옵션 IV, 뉴스 sentiment(NewsAPI), "
        "OPEC 이벤트 캘린더 — 모두 확보 경로 명확."
    )]

    E += [Paragraph("5.3. 평가 — 충분한가?", S["h2"])]
    E += [callout(
        "✅ <b>학습용으로는 충분</b>: 1,558일은 분류 모델 학습에 부족하지 않고, "
        "위기·정상·완만 회복 국면을 모두 포함한다. "
        "△ <b>한계</b>: 코로나 이전(2019년 이전) 데이터가 없어 ‘평시 상태’ 학습량이 제한적이고, "
        "은(銀)처럼 변동성이 낮은 자재는 양성표본이 408일로 다른 자재 대비 적어 SMOTE에 의존한다. "
        "<br/><br/>"
        "→ 결론: 현재 데이터로 <b>서비스 출시(MVP) 수준은 가능</b>하나, "
        "장기 안정성 보장을 위해서는 2010년대 데이터까지 백필이 필요하다."
    )]

    E += [PageBreak()]

    # ───────── 6. [질문③] 비용 및 리스크 ─────────
    E += [Paragraph("6. [적용 관점 ③] 비용 및 리스크", S["h1"])]
    E += [p(
        "<b>핵심 질문</b>: 오분류가 발생하면 어떤 위험이 있는가?"
    )]
    E += [Paragraph("6.1. 오분류 유형별 비용 구조", S["h2"])]
    cost_tbl = [
        ["오분류 유형", "발생 상황", "사용자 입장 비용", "심각도"],
        ["<b>FN (제2종 오류)</b><br/>이상변동을 놓침",
         "실제 큰 변동이 있었는데 알림 없음",
         "예측 못한 손실 발생, 서비스 신뢰 직격타",
         "<font color=\"#B71C1C\"><b>치명적</b></font>"],
        ["<b>FP (제1종 오류)</b><br/>헛알람",
         "잔잔한 날 ‘주의’ 알림 발송",
         "불필요한 매매 보류, 알림 피로도",
         "<font color=\"#1565C0\">경미</font>"],
        ["<b>잘못된 매수타이밍</b>",
         "확률 70% 추천 후 실제 하락",
         "직접적인 매매 손실, 책임 소재 발생",
         "<font color=\"#B71C1C\"><b>심각</b></font>"],
        ["<b>매수타이밍 누락</b>",
         "기회였지만 추천 안 나옴",
         "잠재 수익 기회 상실 (실제 손실 X)",
         "<font color=\"#1565C0\">경미</font>"],
    ]
    E += [make_table(cost_tbl, col_widths=[36*mm, 44*mm, 56*mm, 24*mm])]

    E += [Paragraph("6.2. 왜 비용이 비대칭인가", S["h2"])]
    E += [p(
        "행동경제학의 손실회피 효과에 따르면 사람은 같은 금액의 손실을 이익보다 약 <b>2배</b>로 무겁게 느낀다. "
        "여기에 더해 알림 서비스에서 ‘놓침(FN)’은 ‘서비스가 일을 안 했다’는 신뢰 파괴로 직결되는 반면, "
        "‘헛알람(FP)’은 누적되면 피로도를 유발하지만 단건은 대체로 용인된다. "
        "즉 위험경보는 <b>놓치지 않는 것</b>이, 매수타이밍은 <b>헛추천하지 않는 것</b>이 더 큰 가치다."
    )]

    E += [Paragraph("6.3. 본 모델의 리스크 완화 설계", S["h2"])]
    E += [b(
        "<b>투트랙 분리</b>로 위험경보와 매수타이밍의 비용 구조 차이를 임계값으로 반영. "
        "본질적으로 비용 함수가 다른 두 기능을 같은 임계값으로 운영하지 않도록 강제."
    )]
    E += [b(
        "<b>천연가스 위험경보 Recall 75%</b> — 4건 중 3건의 큰 변동을 사전에 알림. "
        "남은 25%의 FN은 ‘예측 불가 사건’으로 면책 범위 설정 권장."
    )]
    E += [b(
        "<b>매수타이밍 Precision 66~67% (금·은)</b> — 추천 시 약 2/3 적중. "
        "단독 신호 대신 ‘참고 지표’로 위치시키고 책임 소재 문서화 필요."
    )]
    E += [b(
        "<b>법적·윤리적 리스크</b> — B2C 서비스 시 ‘투자자문’이 아닌 ‘정보 제공’ 범위 명시 필요 "
        "(자본시장법 투자자문업 회피). 면책 조항·과거 적중률 공시 의무화."
    )]

    E += [PageBreak()]

    # ───────── 7. [질문④] 모델 한계와 일반화 ─────────
    E += [Paragraph("7. [적용 관점 ④] 모델 한계와 일반화", S["h1"])]
    E += [p(
        "<b>핵심 질문</b>: 다른 환경에서도 일반화할 수 있는가?"
    )]

    E += [Paragraph("7.1. 자재 간 일반화 — 같은 코드, 다른 결과", S["h2"])]
    E += [p(
        "본 주차 통합 실험은 그 자체로 일반화 테스트다. <b>동일한 파이프라인</b>으로 4개 자재를 학습한 결과:"
    )]
    E += [b("은: AUC 0.757 (최고) — 변동성이 낮고 패턴이 안정적인 자재에 모델이 잘 맞음")]
    E += [b("금: AUC 0.726 — VIX·GPR과의 상관이 강해 모델 적합")]
    E += [b("원유: AUC 0.613 — OPEC 정책 등 모델이 보지 못한 외부 충격이 잦음")]
    E += [b("천연가스: AUC 0.571 — 계절성·날씨 등 미반영 요인이 커 가장 낮음")]
    E += [callout(
        "📌 <b>관찰</b>: 같은 카테고리(원자재)라도 자재마다 성능 격차가 매우 크다. "
        "이는 <b>‘모든 자재에 한 모델’ 방식의 한계</b>를 보여준다. "
        "자재별로 임계값·feature 가중치를 분리 튜닝하면 추가 개선이 가능하다."
    )]

    E += [Paragraph("7.2. 시기 일반화 — 학습 기간 밖에서도 통할까", S["h2"])]
    E += [b(
        "학습 데이터(2020~2026) 안에는 코로나 폭락, 우-러 전쟁, 급격한 금리 인상이 모두 포함되어 "
        "‘위기 패턴 학습량’은 풍부. → 유사 위기 재발 시 일반화 가능성 높음."
    )]
    E += [b(
        "그러나 <b>완전히 새로운 패턴</b>(예: 양적완화 시기 장기 저변동성, 디지털 자산 연동 등)에 대해서는 "
        "학습되지 않아 일반화 어려움. 모델 재학습 주기(권장: 분기 1회) 필요."
    )]
    E += [b(
        "TimeSeriesSplit으로 마지막 fold(가장 최근)를 평가한 결과가 평균과 크게 다르지 않아 "
        "<b>최근 시기에도 성능 유지</b> 확인됨."
    )]

    E += [Paragraph("7.3. 도메인 일반화 — 다른 시장으로 확장?", S["h2"])]
    gen_tbl = [
        ["대상", "일반화 가능성", "근거"],
        ["다른 원자재(구리·옥수수·커피)",
         "<font color=\"#1565C0\"><b>높음</b></font>",
         "동일 구조(가격+VIX+거시지표) 적용 가능. 자재별 이상기준만 재설정"],
        ["주식 개별 종목",
         "<font color=\"#888\">중간</font>",
         "VIX는 적용 가능하지만 GPR 영향은 약함. 종목 고유 이벤트가 더 크게 작용"],
        ["환율 (KRW/USD 등)",
         "<font color=\"#888\">중간</font>",
         "GPR은 강하게 작동하나 중앙은행 정책 변수 추가 필요"],
        ["암호화폐",
         "<font color=\"#B71C1C\">낮음</font>",
         "시장 메커니즘 자체가 달라 별도 모델링 권장. 24시간 거래·고변동성"],
    ]
    E += [make_table(gen_tbl, col_widths=[50*mm, 30*mm, 80*mm])]

    E += [PageBreak(), Paragraph("7.4. 알려진 모델 한계 (재정리)", S["h2"])]
    E += [b("이상기준(±%)이 고정값 — 시장 국면별 동적 임계값 미적용")]
    E += [b("백테스트 기반 수익률 시뮬레이션 부족 — 1주차 수준에서 정체")]
    E += [b("GPR 갱신 주기가 일별이라 인트라데이 활용 불가")]
    E += [b("뉴스 sentiment, 옵션 IV 등 ‘선행 정보’ 미반영")]
    E += [b("4개 자재 동시 발생 위기(2020년 3월 형) 별도 학습 필요")]

    E += [PageBreak()]

    # ───────── 8. 자동화 (간단) ─────────
    E += [Paragraph("8. 서비스 운영 인프라 (요약)", S["h1"])]
    E += [Paragraph("8.1. 데이터 자동수집 (auto_update.py)", S["h2"])]
    E += [b("매일 cron으로 실행 → yfinance·GPR 갱신, commodity_vix_gpr_latest.xlsx 갱신")]
    E += [b("update_log.json에 실행 기록 → 운영 모니터링")]
    E += [b("cron 예시: <font face=\"Helvetica\">0 9 * * * python3 week4/code/auto_update.py</font>")]

    E += [Paragraph("8.2. Streamlit 대시보드 (streamlit_app.py)", S["h2"])]
    E += [b("4개 원자재 탭, 임계값 슬라이더로 모드별 효과 실시간 확인")]
    E += [b("ML 용어 대신 일상 언어 (‘맞힌 비율’, ‘놓치지 않은 비율’) — B2C 친화적 UI")]
    E += [b("실행: <font face=\"Helvetica\">streamlit run week4/code/streamlit_app.py</font>")]

    # ───────── 9. 결론 ─────────
    E += [Paragraph("9. 종합 결론", S["h1"])]
    E += [p(
        "본 모델은 ‘하나의 원자재 분류 모델’에서 "
        "‘적용 관점 4대 질문에 모두 답할 수 있는 서비스 후보’로 확장되었다. "
        "사용 시나리오(누가/언제)는 B2C·B2B 3종으로 구체화했고, "
        "데이터는 yfinance+GPR로 무료·자동·지속 가능한 확보 경로를 확인했다. "
        "리스크는 놓침과 헛알람의 비용 차이를 임계값 분리로 직접 흡수했으며, "
        "일반화는 자재 4종 비교로 ‘된다/안 된다’를 실증적으로 보였다."
    )]

    # ───────── 부록 ─────────
    E += [Paragraph("부록. 기술 스택", S["h1"])]
    stk = [["분류", "사용 도구"]]
    stk += [
        ["모델링", "scikit-learn, XGBoost, imbalanced-learn (SMOTE)"],
        ["데이터", "yfinance, GPR Daily (Caldara·Iacoviello), pandas, openpyxl"],
        ["시각화", "matplotlib (AppleGothic)"],
        ["대시보드", "Streamlit"],
        ["문서/리포트", "reportlab (PDF), Markdown"],
        ["자동화", "cron + Python 스크립트, update_log.json"],
    ]
    E += [make_table(stk, col_widths=[40*mm, 130*mm])]

    # ───────── PDF 생성 ─────────
    doc = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=22*mm, bottomMargin=20*mm,
        title="4주차 보고서", author="임태후·유종헌",
    )
    doc.build(E)
    print(f"Saved: {out}")


if __name__ == "__main__":
    build()
