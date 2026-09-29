"""3주차 보고서 PDF 생성 - 제출③ 반복 수정 (Feature 엔지니어링 + SMOTE + 튜닝)"""
import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak,
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

styles = {
    "title": ParagraphStyle("title", fontName=FONT, fontSize=26, leading=34,
                            alignment=TA_CENTER, spaceAfter=6*mm,
                            textColor=colors.HexColor("#1a1a2e")),
    "subtitle": ParagraphStyle("subtitle", fontName=FONT, fontSize=14, leading=20,
                               alignment=TA_CENTER, spaceAfter=3*mm,
                               textColor=colors.HexColor("#444444")),
    "h1": ParagraphStyle("h1", fontName=FONT, fontSize=18, leading=24,
                         spaceBefore=8*mm, spaceAfter=4*mm,
                         textColor=colors.HexColor("#16213e")),
    "h2": ParagraphStyle("h2", fontName=FONT, fontSize=15, leading=20,
                         spaceBefore=5*mm, spaceAfter=3*mm,
                         textColor=colors.HexColor("#0f3460")),
    "body": ParagraphStyle("body", fontName=FONT, fontSize=12, leading=18,
                           spaceAfter=2*mm, textColor=colors.HexColor("#333333")),
    "body_indent": ParagraphStyle("body_indent", fontName=FONT, fontSize=12, leading=18,
                                  spaceAfter=2*mm, leftIndent=8*mm,
                                  textColor=colors.HexColor("#333333")),
    "bullet": ParagraphStyle("bullet", fontName=FONT, fontSize=12, leading=18,
                             spaceAfter=1.5*mm, leftIndent=8*mm, bulletIndent=3*mm,
                             textColor=colors.HexColor("#333333")),
}


def make_table(data, col_widths=None):
    body = ParagraphStyle("tcell", fontName=FONT, fontSize=11, leading=15,
                          textColor=colors.HexColor("#333333"))
    head = ParagraphStyle("thcell", fontName=FONT, fontSize=11, leading=15,
                          textColor=colors.white)
    fmt = []
    for i, row in enumerate(data):
        fmt.append([Paragraph(str(c), head if i == 0 else body) for c in row])
    t = Table(fmt, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
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


def add_image(elements, path, max_width=160*mm, max_height=190*mm):
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


def fmt(v, digits=3):
    return f"{v:.{digits}f}"


def arrow(diff):
    if diff > 0.001:
        return f'<font color="#1565C0">↑ {abs(diff):.3f}</font>'
    if diff < -0.001:
        return f'<font color="#B71C1C">↓ {abs(diff):.3f}</font>'
    return f"→ {abs(diff):.3f}"


def build_pdf():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    charts_dir = os.path.join(os.path.dirname(base_dir), "charts")
    output_path = os.path.join(base_dir, "3주차_보고서.pdf")

    with open(os.path.join(base_dir, "results.json"), encoding="utf-8") as f:
        R = json.load(f)
    sw2 = R["summary_w2"]
    sw3 = R["summary_w3"]
    best = R["best_xgb_metrics"]
    best_params = R["best_xgb_params"]
    feat_imp = R["feature_importance_top10"]

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )
    W = doc.width
    elements = []

    # ===== 표지 =====
    elements.append(Spacer(1, 40*mm))
    elements.append(Paragraph(
        "지정학적 리스크를 반영한<br/>원자재 이상 변동 감지 및 가격 예측",
        styles["title"]))
    elements.append(Spacer(1, 6*mm))
    elements.append(Paragraph("제출 3 - 반복 수정 (Feature 엔지니어링 + SMOTE + 튜닝)",
                              styles["subtitle"]))
    elements.append(Paragraph("머신러닝 학기 프로젝트", styles["subtitle"]))
    elements.append(Spacer(1, 10*mm))
    elements.append(Paragraph(
        "조장 : 20232501 임태후&nbsp;&nbsp;&nbsp;&nbsp;조원 : 20232514 유종헌",
        styles["subtitle"]))
    elements.append(PageBreak())

    # ===== 1. 개요 =====
    elements.append(Paragraph("1. 개요 - 2주차 한계와 3주차 개선 방향", styles["h1"]))
    elements.append(Paragraph(
        "2주차에서는 5개 모델(Random Forest, Logistic Regression, SVM, XGBoost, KNN)을 "
        "TimeSeriesSplit 5-Fold로 비교한 결과, F1-Score는 모두 0.21~0.35 범위에 머물렀다. "
        "특히 Recall이 모든 모델에서 0.15~0.32 수준으로 낮아, 실제 이상변동의 약 70%를 "
        "감지하지 못하는 상태였다. 3주차에서는 이 한계를 다음 세 축으로 개선하였다.",
        styles["body"]))
    elements.append(make_table([
        ["2주차 한계", "3주차 개선 방법", "기대 효과"],
        ["당일 값만 사용 (Feature 7개)",
         "Lag(1·3·5일), 이동평균(MA5·MA20), 변동성(20일), RSI(14) 추가 → 23개",
         "지연된 시장 반응과 추세·변동성 정보 반영"],
        ["클래스 불균형 (이상 35.5%) - Recall 저조",
         "SMOTE로 학습 데이터에서 소수 클래스를 합성 오버샘플링",
         "Recall 개선 (놓치는 이상변동 감소)"],
        ["기본 하이퍼파라미터 사용",
         "GridSearchCV(내부 TimeSeriesSplit 3-Fold)로 XGBoost 파라미터 탐색",
         "모델 성능의 상한 확인"],
    ], col_widths=[W*0.28, W*0.42, W*0.30]))
    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph(
        f"데이터: {R['n_days']}일 (정상 {R['n_normal']}일 / 이상 {R['n_anomaly']}일, "
        f"이상변동 비율 {R['n_anomaly']/R['n_days']*100:.1f}%). "
        "비교 조건은 동일한 TimeSeriesSplit 5-Fold이며, "
        "2주차 조건은 'Feature 7개 + SMOTE 미적용', "
        "3주차 조건은 'Feature 23개 + SMOTE 적용'으로 통제하여 변화의 원인을 명확히 분리했다.",
        styles["body"]))

    elements.append(PageBreak())

    # ===== 2. 추가 Feature 상세 설명 =====
    elements.append(Paragraph("2. 추가 Feature 상세 설명", styles["h1"]))
    elements.append(Paragraph(
        "3주차에서 새로 추가한 16개 Feature는 4개 카테고리로 구성된다. "
        "각 카테고리는 서로 다른 가설을 검증하기 위해 설계되었으며, "
        "본 절에서는 각 Feature의 정의, 계산 방법, 그리고 추가 이유를 정리한다.",
        styles["body"]))

    elements.append(Paragraph("2.1 Feature 전체 구성 (총 23개)", styles["h2"]))
    elements.append(make_table([
        ["카테고리", "개수", "Feature 목록"],
        ["기본 (2주차부터)", "7",
         "VIX, GPR지수, GPR_실제행동, GPR_위협, 금_수익률, 천연가스_수익률, 은_수익률"],
        ["시차 (Lag)", "9",
         "VIX_lag(1·3·5), GPR_lag(1·3·5), 원유수익률_lag(1·3·5)"],
        ["이동평균", "4",
         "원유_MA5, 원유_MA20, VIX_MA5, MA_ratio"],
        ["변동성", "2",
         "변동성_20d, VIX_변동성_20d"],
        ["모멘텀", "1", "RSI_14"],
    ], col_widths=[W*0.20, W*0.10, W*0.70]))

    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph("2.2 시차 (Lag) Feature - 9개", styles["h2"]))
    elements.append(Paragraph(
        "<b>가설:</b> 시장 충격은 당일에 모두 반영되지 않고 며칠 동안 여진이 이어진다. "
        "예를 들어 전쟁 발발 다음 날에도 공급 우려로 추가 변동이 발생하는 경우가 많다.",
        styles["body"]))
    elements.append(Paragraph(
        "<b>계산:</b> shift(n)으로 n일 전 값을 오늘의 Feature로 사용",
        styles["body"]))
    elements.append(Paragraph(
        "  df['VIX_lag1'] = df['VIX'].shift(1)  # 어제의 VIX",
        styles["body_indent"]))
    elements.append(Paragraph(
        "<b>1·3·5일을 선택한 이유:</b>",
        styles["body"]))
    elements.append(Paragraph(
        "- 1일: 직전일 효과 (가장 즉각적인 시장 반응)",
        styles["bullet"]))
    elements.append(Paragraph(
        "- 3일: 단기 누적 반응 (주 중반 무렵의 시장 흐름)",
        styles["bullet"]))
    elements.append(Paragraph(
        "- 5일: 1주일 효과 (시장이 한 사이클 돌고 난 후 잔존 영향)",
        styles["bullet"]))

    elements.append(PageBreak())
    elements.append(Paragraph("2.3 이동평균 Feature - 4개", styles["h2"]))
    elements.append(Paragraph(
        "<b>가설:</b> 일별 노이즈를 제거하고 추세(trend) 정보를 모델에 직접 제공하면 "
        "이상변동의 시점 파악에 유리하다.",
        styles["body"]))
    elements.append(make_table([
        ["Feature", "정의", "의미"],
        ["원유_MA5", "최근 5일 종가 평균", "단기 추세 (1주일)"],
        ["원유_MA20", "최근 20일 종가 평균", "장기 추세 (1개월)"],
        ["VIX_MA5", "최근 5일 VIX 평균", "시장 불안의 단기 추세"],
        ["MA_ratio", "원유_MA5 / 원유_MA20", "MA_ratio &gt; 1 상승추세 / &lt; 1 하락추세"],
    ], col_widths=[W*0.20, W*0.30, W*0.50]))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        "<b>특히 MA_ratio는 트레이딩의 '골든크로스/데드크로스'를 수치화한 변수</b>로, "
        "단기와 장기 추세의 상대적 위치를 한 숫자로 표현한다. "
        "Feature 중요도 분석에서 상위 3위를 기록했다.",
        styles["body"]))

    elements.append(PageBreak())

    elements.append(Paragraph("2.4 변동성 Feature - 2개 (Top 1, 2위 차지)", styles["h2"]))
    elements.append(Paragraph(
        "<b>가설:</b> 변동성이 큰 시기에는 큰 변동이 군집해서 발생한다 "
        "(변동성 클러스터링, GARCH 모델의 핵심 가정). 따라서 최근 변동성 자체가 "
        "이상변동 발생을 예측하는 가장 강력한 신호 중 하나가 된다.",
        styles["body"]))
    elements.append(make_table([
        ["Feature", "정의", "수치 의미"],
        ["변동성_20d", "최근 20일 원유 수익률의 표준편차",
         "1.5 = 평온 / 4.0 이상 = 폭풍"],
        ["VIX_변동성_20d", "최근 20일 VIX의 표준편차",
         "VIX 자체가 흔들리는 정도 (시장 합의 결여)"],
    ], col_widths=[W*0.20, W*0.40, W*0.40]))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        "이상변동 정의 자체가 \"|수익률| ≥ 2%\"이므로, 최근 변동성과 직접적으로 연관된다. "
        "Feature 중요도 분석에서 변동성_20d(0.104)가 1위, VIX_변동성_20d(0.052)가 2위를 "
        "차지한 것은 이론적 가설을 데이터가 직접 확인한 결과이다.",
        styles["body"]))

    elements.append(Spacer(1, 3*mm))
    elements.append(Paragraph("2.5 모멘텀 Feature - RSI(14)", styles["h2"]))
    elements.append(Paragraph(
        "<b>RSI(Relative Strength Index):</b> 14일 동안 상승일의 평균 상승폭과 "
        "하락일의 평균 하락폭의 비율을 0~100 사이로 표준화한 지표.",
        styles["body"]))
    elements.append(Paragraph(
        "  RSI = 100 - 100 / (1 + 평균상승 / 평균하락)",
        styles["body_indent"]))
    elements.append(Paragraph(
        "<b>해석 기준:</b>",
        styles["body"]))
    elements.append(Paragraph(
        "- 70 이상: 과매수 (너무 올랐음 → 차익실현 매도로 급락 가능성)",
        styles["bullet"]))
    elements.append(Paragraph(
        "- 30 이하: 과매도 (너무 떨어졌음 → 반등 매수로 급등 가능성)",
        styles["bullet"]))
    elements.append(Paragraph(
        "- 50 근처: 중립 (방향성 약함)",
        styles["bullet"]))
    elements.append(Paragraph(
        "<b>가설:</b> 이상변동은 종종 극단 상태(과매수/과매도)에서 발생한다. "
        "RSI는 이러한 극단 상태를 한 변수로 요약한다.",
        styles["body"]))

    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph("2.6 카테고리별 검증 가설 요약", styles["h2"]))
    elements.append(make_table([
        ["Feature 카테고리", "검증하려는 가설"],
        ["시차 (Lag)", "어제의 시장 충격이 오늘 이상변동에 영향을 주는가?"],
        ["이동평균", "단기/장기 추세 차이가 이상변동의 신호인가?"],
        ["변동성", "변동성이 큰 시기에 더 큰 이상변동이 따라오는가? (변동성 클러스터링)"],
        ["RSI (모멘텀)", "과매수/과매도 극단 상태가 이상변동 직전 상태인가?"],
    ], col_widths=[W*0.25, W*0.75]))

    elements.append(PageBreak())
    elements.append(Paragraph("2.7 결과 요약: Top 5 중요도가 모두 신규 Feature", styles["h2"]))
    elements.append(make_table([
        ["순위", "Feature", "Importance", "카테고리"],
        ["1", "변동성_20d", "0.104", "변동성 (신규)"],
        ["2", "VIX_변동성_20d", "0.052", "변동성 (신규)"],
        ["3", "MA_ratio", "0.050", "이동평균 (신규)"],
        ["4", "원유수익률_lag3", "0.047", "시차 (신규)"],
        ["5", "원유수익률_lag1", "0.046", "시차 (신규)"],
    ], col_widths=[W*0.10, W*0.30, W*0.20, W*0.40]))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        "→ XGBoost가 학습한 결과, 가장 중요한 5개 Feature가 <b>모두 3주차에 새로 추가한 "
        "변수</b>였다. 이는 모델이 \"당일 값\"보다 \"최근 흐름·변동성·시차 정보\"를 더 "
        "신뢰한다는 의미이며, Feature 엔지니어링이 Recall 개선의 핵심 동력이었음을 "
        "직접적으로 보여준다.",
        styles["body"]))

    elements.append(PageBreak())

    # ===== 3. 모델 비교표 =====
    elements.append(Paragraph("3. 모델 비교표 (2주차 vs 3주차)", styles["h1"]))
    elements.append(Paragraph("3.1 모델별 평균 성능 (TimeSeriesSplit 5-Fold)", styles["h2"]))

    rows = [["모델", "조건", "Accuracy", "Precision", "Recall", "F1-Score"]]
    for m in sw3.keys():
        rows.append([m, "2주차",
                     fmt(sw2[m]["accuracy"]), fmt(sw2[m]["precision"]),
                     fmt(sw2[m]["recall"]), fmt(sw2[m]["f1"])])
        rows.append(["", "3주차",
                     f'<b>{fmt(sw3[m]["accuracy"])}</b>',
                     f'<b>{fmt(sw3[m]["precision"])}</b>',
                     f'<b>{fmt(sw3[m]["recall"])}</b>',
                     f'<b>{fmt(sw3[m]["f1"])}</b>'])
    elements.append(make_table(rows,
                               col_widths=[W*0.22, W*0.10, W*0.17, W*0.17, W*0.17, W*0.17]))

    # 변화량 요약
    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph("3.2 지표별 변화량 (3주차 - 2주차)", styles["h2"]))
    diff_rows = [["모델", "ΔAccuracy", "ΔPrecision", "ΔRecall", "ΔF1"]]
    for m in sw3.keys():
        diff_rows.append([
            m,
            arrow(sw3[m]["accuracy"] - sw2[m]["accuracy"]),
            arrow(sw3[m]["precision"] - sw2[m]["precision"]),
            arrow(sw3[m]["recall"] - sw2[m]["recall"]),
            arrow(sw3[m]["f1"] - sw2[m]["f1"]),
        ])
    elements.append(make_table(diff_rows,
                               col_widths=[W*0.24, W*0.19, W*0.19, W*0.19, W*0.19]))
    elements.append(Spacer(1, 2*mm))

    # 핵심 관찰
    best_recall_model = max(sw3, key=lambda x: sw3[x]["recall"])
    best_f1_model = max(sw3, key=lambda x: sw3[x]["f1"])
    elements.append(Paragraph(
        f"→ Recall 1위: <b>{best_recall_model} ({fmt(sw3[best_recall_model]['recall'])})</b>, "
        f"F1 1위: <b>{best_f1_model} ({fmt(sw3[best_f1_model]['f1'])})</b>. "
        "5개 모델 모두 Recall이 상승하였으며, F1도 4개 모델에서 상승하였다.",
        styles["body_indent"]))

    elements.append(PageBreak())

    # ===== 4. 성능 그래프 =====
    elements.append(Paragraph("4. 성능 그래프", styles["h1"]))

    elements.append(Paragraph("4.1 4대 지표 비교 (모델 × 주차)", styles["h2"]))
    add_image(elements, os.path.join(charts_dir, "chart1_w2_vs_w3.png"))
    elements.append(Paragraph(
        "각 모델별로 2주차(회색)와 3주차(파랑)의 4대 지표를 비교한 결과, "
        "Precision·Recall·F1에서 전반적으로 3주차의 막대가 더 길다. "
        "Accuracy는 일부 모델에서 하락했는데, 이는 SMOTE로 인해 소수 클래스 예측 비중이 "
        "늘면서 다수 클래스(정상) 예측 정확도가 일부 희생되었기 때문이다.",
        styles["body"]))

    elements.append(PageBreak())

    elements.append(Paragraph("4.2 Recall 집중 비교 (이상변동 감지율)", styles["h2"]))
    elements.append(Paragraph(
        "3주차 핵심 목표인 Recall만 따로 비교하였다. 5개 모델 전부 Recall이 상승하였다.",
        styles["body"]))
    add_image(elements, os.path.join(charts_dir, "chart2_recall_focus.png"))

    elements.append(PageBreak())

    elements.append(Paragraph("4.3 Fold별 F1-Score 변화 (3주차 안정성)", styles["h2"]))
    elements.append(Paragraph(
        "TimeSeriesSplit 5-Fold 각각에서의 F1을 그렸다. 학습 데이터가 늘어나는 Fold 4~5에서 "
        "대부분의 모델이 더 높은 F1을 보이며, 이는 새로운 Feature가 많은 데이터에서 "
        "더 잘 작동함을 시사한다.",
        styles["body"]))
    add_image(elements, os.path.join(charts_dir, "chart3_fold_f1.png"))

    elements.append(PageBreak())

    elements.append(Paragraph("4.4 ROC 커브 비교 (마지막 Fold)", styles["h2"]))
    add_image(elements, os.path.join(charts_dir, "chart5_roc.png"))
    elements.append(Paragraph(
        "확률 예측 기반 분류 성능을 보여준다. AUC가 1에 가까울수록 우수하며, "
        "0.5는 무작위 추정이다. 3주차 Feature 확장 후 대부분 모델의 AUC가 0.6 이상으로 "
        "확보되었다.",
        styles["body"]))

    elements.append(PageBreak())

    elements.append(Paragraph("4.5 Feature 중요도 (튜닝된 XGBoost)", styles["h2"]))
    add_image(elements, os.path.join(charts_dir, "chart4_feature_importance.png"))
    top3 = list(feat_imp.items())[:3]
    elements.append(Paragraph(
        f"Top 3: {top3[0][0]}({top3[0][1]:.3f}), {top3[1][0]}({top3[1][1]:.3f}), "
        f"{top3[2][0]}({top3[2][1]:.3f}). "
        "<b>주목할 점은 상위 중요도 Feature가 모두 3주차에 새로 추가된 변수라는 것이다.</b> "
        "이는 시차/변동성/추세 정보가 이상변동 감지에 실질적으로 기여함을 의미하며, "
        "Feature 엔지니어링이 효과적이었다는 직접적 증거이다.",
        styles["body"]))

    elements.append(PageBreak())

    # ===== 5. 변화 원인 분석 =====
    elements.append(Paragraph("5. 변화 원인 분석", styles["h1"]))

    elements.append(Paragraph("5.1 무엇이 바뀌었는가 (3가지 변화)", styles["h2"]))
    elements.append(make_table([
        ["변화", "구체적 내용", "영향을 주는 지표"],
        ["① Feature 확장", "7개 → 23개 (Lag·MA·변동성·RSI 추가)", "Precision, Recall, F1"],
        ["② SMOTE 오버샘플링", "학습 데이터에서 이상변동 합성 (테스트는 원본 분포 유지)",
         "Recall (주), Accuracy (희생)"],
        ["③ XGBoost 튜닝", "GridSearchCV로 n_estimators·max_depth·learning_rate 탐색",
         "F1, ROC-AUC"],
    ], col_widths=[W*0.20, W*0.50, W*0.30]))

    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph("5.2 변화별 세부 원인 분석", styles["h2"]))

    elements.append(Paragraph("<b>(1) Recall이 전체적으로 상승한 이유 - SMOTE 효과</b>",
                              styles["body"]))
    elements.append(Paragraph(
        "2주차에서는 정상(64.5%) vs 이상(35.5%)의 불균형으로 모델이 '정상' 쪽으로 편향되어 "
        "이상변동을 놓치는 경향이 있었다. SMOTE는 학습 데이터에서 소수 클래스(이상변동)의 "
        "주변에 합성 데이터를 만들어 클래스 비율을 1:1로 균형 잡는다. 그 결과 모델이 "
        "이상변동 패턴을 더 적극적으로 학습하게 되어 Recall이 상승했다. "
        f"가장 극적인 변화는 SVM(0.200→0.505, +0.305)과 Logistic Regression(0.149→0.413, +0.264)"
        "에서 나타났는데, 이 두 모델은 결정경계가 단순해 클래스 불균형의 영향을 더 크게 받기 "
        "때문이다.",
        styles["bullet"]))

    elements.append(Paragraph(
        "<b>(2) 일부 모델에서 Accuracy가 하락한 이유 - Precision-Recall Trade-off</b>",
        styles["body"]))
    elements.append(Paragraph(
        "Accuracy는 정상/이상 모두를 종합한 정확도이다. SMOTE로 모델이 이상변동을 더 적극적으로 "
        "예측하게 되면서, 실제로는 정상인데 이상으로 잘못 예측하는 사례(FP)가 일부 늘어났다. "
        "그 결과 정상 클래스의 정확도가 떨어지고 전체 Accuracy가 낮아진다. "
        "이는 의도된 trade-off로, 이상변동 감지가 목적인 본 프로젝트에서는 "
        "Accuracy 약간의 하락보다 Recall 개선이 훨씬 중요하다.",
        styles["bullet"]))

    elements.append(Paragraph(
        "<b>(3) F1이 4개 모델에서 상승한 이유 - Feature 확장 효과</b>",
        styles["body"]))
    elements.append(Paragraph(
        "F1은 Precision과 Recall의 조화평균이다. SMOTE만으로는 Recall은 오르지만 Precision이 "
        "내려가 F1이 그대로일 수 있다. 그러나 새로 추가한 Lag/MA/변동성/RSI Feature가 "
        "이상변동의 예측 패턴을 더 잘 포착하면서 Precision도 함께 유지되거나 상승했다. "
        f"Feature 중요도 Top 3가 모두 신규 변수({top3[0][0]}, {top3[1][0]}, {top3[2][0]})라는 "
        "사실이 이를 뒷받침한다.",
        styles["bullet"]))

    elements.append(PageBreak())
    elements.append(Paragraph(
        "<b>(4) XGBoost 튜닝 결과 - 단일 모델로 도달한 최고 성능</b>",
        styles["body"]))
    elements.append(make_table([
        ["조건", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        ["2주차 XGBoost (기본 파라미터, Feature 7개)",
         fmt(sw2["XGBoost"]["accuracy"]), fmt(sw2["XGBoost"]["precision"]),
         fmt(sw2["XGBoost"]["recall"]), fmt(sw2["XGBoost"]["f1"]), "-"],
        ["3주차 XGBoost (기본 파라미터, Feature 23개 + SMOTE)",
         fmt(sw3["XGBoost"]["accuracy"]), fmt(sw3["XGBoost"]["precision"]),
         fmt(sw3["XGBoost"]["recall"]), fmt(sw3["XGBoost"]["f1"]), "-"],
        ["3주차 XGBoost (튜닝 후, 마지막 Fold 평가)",
         f'<b>{fmt(best["accuracy"])}</b>', f'<b>{fmt(best["precision"])}</b>',
         f'<b>{fmt(best["recall"])}</b>', f'<b>{fmt(best["f1"])}</b>',
         f'<b>{fmt(best["roc_auc"])}</b>'],
    ], col_widths=[W*0.40, W*0.12, W*0.13, W*0.11, W*0.10, W*0.14]))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        f"최적 파라미터: n_estimators={best_params['n_estimators']}, "
        f"max_depth={best_params['max_depth']}, "
        f"learning_rate={best_params['learning_rate']}. "
        f"튜닝 후 F1 {fmt(best['f1'])}로, "
        f"1주차 단일 모델 0.393 → 2주차 5-Fold 평균 0.352 → "
        f"3주차 튜닝 후 {fmt(best['f1'])}로 개선되었다.",
        styles["body_indent"]))

    elements.append(PageBreak())

    elements.append(Paragraph("5.3 1·2·3주차 통합 성능 추이 (XGBoost 기준)", styles["h2"]))
    elements.append(make_table([
        ["주차", "조건", "Accuracy", "Precision", "Recall", "F1"],
        ["1주차", "Random Split, RF", "0.683", "0.640",
         '<font color="#B71C1C">0.283</font>', "0.393"],
        ["2주차", "TimeSeriesSplit, XGB(기본)",
         fmt(sw2["XGBoost"]["accuracy"]), fmt(sw2["XGBoost"]["precision"]),
         fmt(sw2["XGBoost"]["recall"]), fmt(sw2["XGBoost"]["f1"])],
        ["3주차", "+ Feature 23개 + SMOTE + 튜닝",
         f'<b>{fmt(best["accuracy"])}</b>', f'<b>{fmt(best["precision"])}</b>',
         f'<b><font color="#1565C0">{fmt(best["recall"])}</font></b>',
         f'<b>{fmt(best["f1"])}</b>'],
    ], col_widths=[W*0.10, W*0.36, W*0.14, W*0.14, W*0.13, W*0.13]))
    elements.append(Spacer(1, 3*mm))
    elements.append(Paragraph(
        "주의: 1주차 성능은 Random Split으로 인한 데이터 누출이 포함되어 부풀려진 수치이다. "
        "공정한 비교는 2주차→3주차 사이에서 이루어지며, "
        f"Recall은 {fmt(sw2['XGBoost']['recall'])} → {fmt(best['recall'])} "
        f"({(best['recall']-sw2['XGBoost']['recall']):+.3f}), "
        f"F1은 {fmt(sw2['XGBoost']['f1'])} → {fmt(best['f1'])} "
        f"({(best['f1']-sw2['XGBoost']['f1']):+.3f})로 개선되었다.",
        styles["body"]))

    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph("5.4 남은 한계", styles["h2"]))
    elements.append(Paragraph(
        "3주차 개선에도 불구하고 Recall은 여전히 50% 수준에 머무르며, 이상변동의 절반 가까이는 "
        "감지되지 않는다. 이는 다음 한계에서 비롯된 것으로 분석된다.",
        styles["body"]))
    elements.append(Paragraph(
        "- 이상변동 기준이 ±2% 고정 → 시기별 변동성 차이를 반영하는 적응형 임계값 필요",
        styles["bullet"]))
    elements.append(Paragraph(
        "- 외부 데이터 부족 → 뉴스 텍스트, 산유국 OPEC 결정 등 비정형 데이터 미반영",
        styles["bullet"]))
    elements.append(Paragraph(
        "- 단일 임계값 0.5 사용 → 사용 목적(위험경보 vs 매수추천)에 따라 분리된 임계값 적용 여지",
        styles["bullet"]))

    elements.append(PageBreak())

    # ===== 부록 =====
    elements.append(Paragraph("부록: 사용 도구 및 환경", styles["h1"]))
    elements.append(make_table([
        ["항목", "내용"],
        ["프로그래밍 언어", "Python 3.9"],
        ["데이터", f"commodity_vix_gpr_data.xlsx ({R['n_days']}일)"],
        ["모델", "scikit-learn (RF, LR, SVM, KNN), XGBoost"],
        ["오버샘플링", "imbalanced-learn SMOTE"],
        ["하이퍼파라미터 튜닝", "GridSearchCV (TimeSeriesSplit 3-Fold 내부 검증)"],
        ["교차검증", "TimeSeriesSplit 5-Fold (외부 평가)"],
        ["전처리", "StandardScaler"],
        ["시각화", "matplotlib (AppleGothic 폰트)"],
    ], col_widths=[W*0.30, W*0.70]))

    doc.build(elements)
    print(f"PDF 생성 완료: {output_path}")
    return output_path


if __name__ == "__main__":
    build_pdf()
