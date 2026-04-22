"""1주차 보고서 PDF 생성 스크립트 - 글씨체 크기 조정 버전"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, KeepTogether
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os

# --- 폰트 등록 ---
font_registered = False
for font_path in [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/AppleGothic.ttf",
]:
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("Korean", font_path))
            font_registered = True
            break
        except Exception:
            continue

if not font_registered:
    # fallback: try AppleSDGothicNeo from ttc
    try:
        from reportlab.pdfbase.ttfonts import TTFont
        pdfmetrics.registerFont(TTFont("Korean", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"))
        font_registered = True
    except Exception:
        print("WARNING: Korean font not found. Using Helvetica.")

FONT = "Korean" if font_registered else "Helvetica"

# --- 스타일 정의 (크기 조정) ---
styles = {
    "title": ParagraphStyle(
        "title", fontName=FONT, fontSize=26, leading=34,
        alignment=TA_CENTER, spaceAfter=6*mm, textColor=colors.HexColor("#1a1a2e")
    ),
    "subtitle": ParagraphStyle(
        "subtitle", fontName=FONT, fontSize=14, leading=20,
        alignment=TA_CENTER, spaceAfter=3*mm, textColor=colors.HexColor("#444444")
    ),
    "h1": ParagraphStyle(
        "h1", fontName=FONT, fontSize=18, leading=24,
        spaceBefore=8*mm, spaceAfter=4*mm, textColor=colors.HexColor("#16213e")
    ),
    "h2": ParagraphStyle(
        "h2", fontName=FONT, fontSize=15, leading=20,
        spaceBefore=5*mm, spaceAfter=3*mm, textColor=colors.HexColor("#0f3460")
    ),
    "body": ParagraphStyle(
        "body", fontName=FONT, fontSize=12, leading=18,
        spaceAfter=2*mm, textColor=colors.HexColor("#333333")
    ),
    "body_indent": ParagraphStyle(
        "body_indent", fontName=FONT, fontSize=12, leading=18,
        spaceAfter=2*mm, leftIndent=8*mm, textColor=colors.HexColor("#333333")
    ),
    "bullet": ParagraphStyle(
        "bullet", fontName=FONT, fontSize=12, leading=18,
        spaceAfter=1.5*mm, leftIndent=8*mm, bulletIndent=3*mm,
        textColor=colors.HexColor("#333333")
    ),
    "caption": ParagraphStyle(
        "caption", fontName=FONT, fontSize=10, leading=14,
        alignment=TA_CENTER, spaceAfter=3*mm, textColor=colors.HexColor("#666666")
    ),
}

# --- 표 스타일 ---
def make_table(data, col_widths=None):
    """표를 생성하는 헬퍼 함수"""
    # 셀 내용을 Paragraph로 변환
    table_style_body = ParagraphStyle(
        "tcell", fontName=FONT, fontSize=11, leading=15, textColor=colors.HexColor("#333333")
    )
    table_style_header = ParagraphStyle(
        "thcell", fontName=FONT, fontSize=11, leading=15, textColor=colors.white
    )

    formatted = []
    for i, row in enumerate(data):
        fmt_row = []
        for cell in row:
            s = table_style_header if i == 0 else table_style_body
            fmt_row.append(Paragraph(str(cell), s))
        formatted.append(fmt_row)

    t = Table(formatted, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t

def add_image(elements, path, caption_text=None, max_width=150*mm, max_height=180*mm):
    """이미지 추가 헬퍼 - 비율 유지하면서 페이지에 맞게 조정"""
    if os.path.exists(path):
        from reportlab.lib.utils import ImageReader
        ir = ImageReader(path)
        iw, ih = ir.getSize()
        ratio = iw / ih
        # max_width 기준으로 높이 계산
        w = max_width
        h = w / ratio
        # 높이가 max_height 초과하면 높이 기준으로 재계산
        if h > max_height:
            h = max_height
            w = h * ratio
        img = Image(path, width=w, height=h)
        img.hAlign = "CENTER"
        elements.append(img)
        if caption_text:
            elements.append(Spacer(1, 2*mm))
            elements.append(Paragraph(caption_text, styles["caption"]))
    else:
        elements.append(Paragraph(f"[이미지 없음: {path}]", styles["body"]))

# --- PDF 생성 ---
def build_pdf():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    charts_dir = os.path.join(os.path.dirname(base_dir), "charts")
    output_path = os.path.join(base_dir, "1주차_보고서.pdf")

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm
    )

    W = doc.width
    elements = []

    # ===== 표지 =====
    elements.append(Spacer(1, 40*mm))
    elements.append(Paragraph(
        "지정학적 리스크를 반영한<br/>원자재 이상 변동 감지 및 가격 예측",
        styles["title"]
    ))
    elements.append(Spacer(1, 6*mm))
    elements.append(Paragraph("제출 1 - 초기 설계 &amp; 1차 실행", styles["subtitle"]))
    elements.append(Paragraph("머신러닝 학기 프로젝트", styles["subtitle"]))
    elements.append(Spacer(1, 10*mm))
    elements.append(Paragraph("조장 : 20232501 임태후&nbsp;&nbsp;&nbsp;&nbsp;조원 : 20232514 유종헌", styles["subtitle"]))
    elements.append(Spacer(1, 6*mm))
    elements.append(Paragraph("2026년 3월", styles["subtitle"]))
    elements.append(PageBreak())

    # ===== 1. 문제 정의서 =====
    elements.append(Paragraph("1. 문제 정의서", styles["h1"]))

    elements.append(Paragraph("1.1 해결할 문제", styles["h2"]))
    elements.append(Paragraph(
        "국제 원자재 시장은 지정학적 리스크(전쟁, 군사적 긴장, 테러 등)에 민감하게 반응한다. "
        "2020년 코로나 팬데믹 당시 원유 가격이 사상 최초로 -37달러까지 폭락했고, "
        "2022년 러시아-우크라이나 전쟁 발발 시 원유와 천연가스 가격이 급등했다.",
        styles["body"]
    ))
    elements.append(Paragraph(
        "이 프로젝트는 지정학적 리스크 지표(GPR Index, VIX)를 활용하여 "
        "원자재 가격의 이상 변동을 감지하고, 가격 방향을 예측하는 머신러닝 모델을 개발한다.",
        styles["body"]
    ))

    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph("1.2 예측 목표", styles["h2"]))
    elements.append(make_table([
        ["목표", "유형", "설명"],
        ["이상 변동 감지", "이진 분류", "원유 일별 수익률 ±2% 이상을 이상변동으로 감지"],
        ["가격 방향 예측", "이진 분류", "내일 원유 가격의 상승/하락 예측"],
    ], col_widths=[W*0.22, W*0.18, W*0.60]))

    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph("1.3 기대 효과", styles["h2"]))

    effects = [
        ("1) 투자 손실 최소화",
         "지정학적 위기 발생 시 원자재 가격 급락을 사전 감지하여 포지션을 정리함으로써, "
         "대규모 손실을 방지할 수 있다. 본 모델의 매도 신호 적중률(71.4%)은 하락 회피 전략에 활용 가능성을 보여준다."),
        ("2) 매매 타이밍 최적화를 통한 수익 극대화",
         "리스크 지표 기반의 매수/매도 신호를 활용하여, 단순 보유(Buy &amp; Hold) 대비 더 높은 수익률을 목표로 한다. "
         "현재 모델은 매수 신호의 정확도가 부족하나, 모델 개선을 통해 수익화 가능성이 있다."),
        ("3) 리스크 경보 구독 서비스화",
         "원자재 트레이더, 에너지 기업, 항공사 등 원자재 가격에 민감한 기업/개인에게 "
         "실시간 이상변동 경보를 제공하는 유료 구독 서비스로 확장 가능하다."),
        ("4) 원자재 의존 기업의 원가 관리",
         "제조업, 항공, 물류 등 원자재를 대량 구매하는 기업이 매입 시점을 최적화하여 원가를 절감할 수 있다."),
    ]
    for title, desc in effects:
        elements.append(Paragraph(f"<b>{title}</b>", styles["body"]))
        elements.append(Paragraph(desc, styles["body_indent"]))

    elements.append(PageBreak())

    # ===== 2. 데이터 설명 =====
    elements.append(Paragraph("2. 데이터 설명", styles["h1"]))

    elements.append(Paragraph("2.1 데이터 출처", styles["h2"]))
    elements.append(make_table([
        ["데이터", "출처", "수집 방법"],
        ["원자재 가격", "Yahoo Finance", "yfinance 라이브러리"],
        ["VIX 지수", "CBOE (Yahoo Finance)", "yfinance 라이브러리"],
        ["GPR Index", "Caldara &amp; Iacoviello (FRB)", "공개 엑셀 파일"],
    ], col_widths=[W*0.25, W*0.40, W*0.35]))

    elements.append(Paragraph("2.2 데이터 크기", styles["h2"]))
    elements.append(Paragraph("- 기간: 2020-01-02 ~ 2026-03-16", styles["bullet"]))
    elements.append(Paragraph("- 행 수: 1,558일 (거래일 기준) / 변수 수: 8개", styles["bullet"]))

    elements.append(Paragraph("2.3 주요 변수", styles["h2"]))
    elements.append(make_table([
        ["변수", "설명", "평균", "최소", "최대"],
        ["원유(WTI)", "원유 선물 종가 (USD)", "69.91", "-37.63", "123.70"],
        ["금", "금 선물 종가 (USD)", "2,283", "1,477", "5,318"],
        ["천연가스", "천연가스 선물 (USD)", "3.51", "1.48", "9.68"],
        ["은", "은 선물 종가 (USD)", "28.54", "11.73", "115.08"],
        ["VIX", "시장 공포지수", "20.95", "11.86", "82.69"],
        ["GPR지수", "지정학적 리스크 종합", "134.44", "22.26", "564.93"],
        ["GPR 실제행동", "실제 지정학적 사건", "115.53", "0.00", "653.93"],
        ["GPR 위협", "지정학적 위협 수준", "156.74", "21.69", "809.49"],
    ], col_widths=[W*0.15, W*0.30, W*0.15, W*0.15, W*0.15]))

    elements.append(PageBreak())
    elements.append(Paragraph("2.4 원자재 가격 추이", styles["h2"]))
    add_image(elements, os.path.join(charts_dir, "chart1_prices.png"))

    elements.append(Paragraph("2.5 리스크 지표 추이", styles["h2"]))
    add_image(elements, os.path.join(charts_dir, "chart2_risk.png"))

    elements.append(PageBreak())
    elements.append(Paragraph("2.6 변수 간 상관관계", styles["h2"]))
    add_image(elements, os.path.join(charts_dir, "chart3_corr.png"))

    elements.append(Paragraph("2.7 주요 상관관계 해석", styles["h2"]))
    elements.append(make_table([
        ["관계", "상관계수", "해석"],
        ["금 - 은", "+0.922", "매우 강한 양의 관계 (같이 움직임)"],
        ["원유 - 천연가스", "+0.589", "보통 양의 관계 (에너지 자원)"],
        ["GPR - 원유", "+0.404", "리스크 상승시 원유 가격 상승"],
        ["VIX - 원유", "-0.320", "공포 상승시 원유 가격 하락"],
    ], col_widths=[W*0.25, W*0.18, W*0.57]))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        "* 일별 변동률 기준 상관관계는 대부분 0에 가까움 → 당일 리스크 지표만으로는 즉각적 가격 변동 설명이 어려움",
        styles["body_indent"]
    ))

    elements.append(PageBreak())

    # ===== 3. 기본 모델 실행 =====
    elements.append(Paragraph("3. 기본 모델 실행", styles["h1"]))

    elements.append(Paragraph("3.1 모델 A: 이상 변동 감지 (Random Forest)", styles["h2"]))
    elements.append(Paragraph(
        "<b>[이상 변동 정의]</b> 원유 일별 수익률의 절대값이 ±2% 이상인 날을 이상 변동으로 정의. "
        "전체 1,557일 중 정상 1,004일(64.5%), 이상변동 553일(35.5%)",
        styles["body"]
    ))
    add_image(elements, os.path.join(charts_dir, "chart4_anomaly.png"))

    elements.append(make_table([
        ["항목", "내용"],
        ["알고리즘", "Random Forest Classifier"],
        ["나무 수 (n_estimators)", "100"],
        ["최대 깊이 (max_depth)", "10"],
        ["학습 / 테스트 비율", "80% / 20% (Random Split)"],
        ["Feature (7개)", "VIX, GPR지수, GPR실제행동, GPR위협, 금/천연가스/은 수익률"],
    ], col_widths=[W*0.35, W*0.65]))

    elements.append(Paragraph("3.2 이상 변동 감지 결과", styles["h2"]))
    elements.append(make_table([
        ["성능 지표", "값", "설명"],
        ["Accuracy (정확도)", "68.27%", "전체 중 맞춘 비율"],
        ["Precision (정밀도)", "64.00%", "이상변동 예측 중 실제 이상변동 비율"],
        ["Recall (재현율)", "28.32%", "실제 이상변동 중 감지한 비율"],
        ["F1-Score", "39.26%", "Precision과 Recall의 균형 점수"],
    ], col_widths=[W*0.25, W*0.15, W*0.60]))

    elements.append(PageBreak())
    elements.append(Paragraph("혼동 행렬 (Confusion Matrix)", styles["h2"]))
    add_image(elements, os.path.join(charts_dir, "chart5_cm.png"), None, max_width=100*mm, max_height=130*mm)
    elements.append(Paragraph("- 정상 → 정상 (TN): 181건 / 정상 → 이상 (FP): 18건 [거짓 경보]", styles["bullet"]))
    elements.append(Paragraph("- 이상 → 정상 (FN): 81건 [놓친 이상변동] / 이상 → 이상 (TP): 32건 [성공적 감지]", styles["bullet"]))

    elements.append(Spacer(1, 3*mm))
    elements.append(Paragraph("Feature 중요도", styles["h2"]))
    add_image(elements, os.path.join(charts_dir, "chart6_importance.png"), None, max_width=150*mm, max_height=100*mm)
    elements.append(Paragraph("→ VIX(공포지수)가 이상 변동 예측에 가장 중요한 변수", styles["body_indent"]))

    elements.append(PageBreak())
    elements.append(Paragraph("3.3 모델 B: 가격 방향 예측 &amp; 매매 타이밍", styles["h2"]))
    elements.append(make_table([
        ["항목", "내용"],
        ["방향 예측 모델", "Random Forest Classifier (n=200, depth=10)"],
        ["수익률 예측 모델", "Random Forest Regressor (n=200, depth=10)"],
        ["데이터 분할", "시간순 80% / 20% 분할"],
        ["매수 조건", "방향=상승 AND 예측 수익률 > 0.3%"],
        ["매도 조건", "방향=하락 AND 예측 수익률 < -0.3%"],
    ], col_widths=[W*0.30, W*0.70]))

    elements.append(Spacer(1, 3*mm))
    elements.append(make_table([
        ["지표", "값"],
        ["방향 예측 정확도", "47.4%"],
        ["매수 신호 횟수 / 적중률", "9회 / 33.3%"],
        ["매도 신호 횟수 / 적중률", "14회 / 71.4%"],
    ], col_widths=[W*0.45, W*0.55]))

    elements.append(Spacer(1, 3*mm))
    elements.append(Paragraph("<b>투자 시뮬레이션 (100만원 기준)</b>", styles["body"]))
    elements.append(make_table([
        ["전략", "최종 자산", "수익률"],
        ["그냥 보유 (Buy &amp; Hold)", "1,335,333원", "+33.5%"],
        ["모델 신호 매수", "904,128원", "-9.6%"],
    ], col_widths=[W*0.35, W*0.30, W*0.20]))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        "→ 모델의 매수 신호를 따르면 오히려 손실 발생. 다만 매도 신호는 71.4% 적중률로 하락 회피에 부분적 유용.",
        styles["body_indent"]
    ))

    elements.append(PageBreak())

    # ===== 4. 성능 지표 해석 =====
    elements.append(Paragraph("4. 성능 지표 해석", styles["h1"]))

    elements.append(Paragraph("4.1 이상 변동 감지 모델 (모델 A)", styles["h2"]))
    metrics_a = [
        ("Accuracy 68.3%",
         "전체 예측 중 약 2/3를 맞추었으나, \"모두 정상\"으로 예측해도 64.5%를 달성할 수 있는 수치이므로 "
         "모델의 실질적 기여는 제한적이다."),
        ("Precision 64.0%",
         "모델이 \"이상변동\"이라고 판단한 건 중 64%가 실제 이상변동. 거짓 경보율 36%."),
        ("Recall 28.3%",
         "가장 큰 문제점. 실제 이상변동 113건 중 32건(28%)만 감지. 72%의 이상변동을 놓치고 있다."),
        ("F1-Score 39.3%",
         "Precision과 Recall의 불균형으로 인해 낮은 종합 점수를 기록했다."),
    ]
    for metric, desc in metrics_a:
        elements.append(Paragraph(f"<b>{metric}</b>: {desc}", styles["body"]))

    elements.append(Paragraph("4.2 가격 방향 예측 모델 (모델 B)", styles["h2"]))
    metrics_b = [
        ("방향 정확도 47.4%", "동전 던지기(50%)보다 낮은 수치. 현재 Feature만으로는 가격 방향 예측이 어렵다."),
        ("매수 적중률 33.3%", "매수 타이밍 포착에 실패했다."),
        ("매도 적중률 71.4%", "하락 신호는 비교적 잘 감지하여, 위험 회피 관점에서는 부분적 활용이 가능하다."),
    ]
    for metric, desc in metrics_b:
        elements.append(Paragraph(f"<b>{metric}</b>: {desc}", styles["body"]))

    elements.append(Paragraph("4.3 종합 평가", styles["h2"]))
    elements.append(Paragraph(
        "현재 모델은 \"위험 경보\" 용도로는 부분적으로 작동하나, \"매수 타이밍 추천\" 용도로는 부적합하다. "
        "1주차 기본 모델의 한계를 명확히 보여주는 결과이며, 이후 단계에서 개선이 필요하다.",
        styles["body"]
    ))

    elements.append(PageBreak())

    # ===== 5. 현재 한계 및 개선 방향 =====
    elements.append(Paragraph("5. 현재 한계 및 개선 방향", styles["h1"]))

    elements.append(Paragraph("5.1 현재 한계", styles["h2"]))
    elements.append(make_table([
        ["#", "한계점", "상세 설명"],
        ["1", "시차(Lag) 미반영", "오늘의 GPR/VIX만 사용. 며칠 전 리스크 변화가 반영되지 않음"],
        ["2", "이상변동 기준 단순", "±2% 고정 임계값. 변동성이 큰 시기엔 부적합"],
        ["3", "시간 순서 무시", "Random Split 사용. 시계열 분할이 더 적절"],
        ["4", "원유만 대상", "금, 천연가스, 은 개별 모델 미구축"],
        ["5", "Feature 부족", "이동평균, 거래량, 기술적 추세 지표 미포함"],
        ["6", "Recall 부족", "실제 이상변동의 72%를 놓치고 있음"],
    ], col_widths=[W*0.06, W*0.22, W*0.72]))

    elements.append(Paragraph("5.2 개선 방향", styles["h2"]))
    improvements = [
        "여러 모델 비교 (Logistic Regression, SVM, XGBoost, KNN)",
        "시계열 분할 (TimeSeriesSplit) 적용",
        "시차 Feature 추가 (전일/전주 GPR, VIX 변화량)",
        "이동평균, 볼린저밴드 등 기술적 지표 추가",
        "하이퍼파라미터 튜닝 (GridSearch / RandomSearch)",
        "Class Imbalance 처리 (SMOTE, class_weight 조정)",
        "4개 원자재 통합 예측 시스템으로 확장",
        "실시간 투자 경보 시스템 적용 가능성 분석",
        "API를 이용한 데이터 자동 수집 및 실시간 처리 파이프라인 구축",
    ]
    elements.append(make_table(
        [["개선 내용"]] + [[item] for item in improvements],
        col_widths=[W]
    ))

    elements.append(PageBreak())

    # ===== 부록 =====
    elements.append(Paragraph("부록: 사용 도구 및 환경", styles["h1"]))
    elements.append(make_table([
        ["항목", "내용"],
        ["프로그래밍 언어", "Python 3.9"],
        ["실행 환경", "Google Colab"],
        ["데이터 수집", "yfinance, pandas"],
        ["데이터 분석", "numpy, pandas"],
        ["머신러닝", "scikit-learn (RandomForest)"],
        ["시각화", "matplotlib"],
    ], col_widths=[W*0.30, W*0.70]))

    # --- 빌드 ---
    doc.build(elements)
    print(f"PDF 생성 완료: {output_path}")

if __name__ == "__main__":
    build_pdf()
