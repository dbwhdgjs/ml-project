"""2주차 보고서 PDF 생성 - 제출② 모델 확장 & 비교"""

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
from reportlab.lib.utils import ImageReader
import numpy as np
import os

# --- 폰트 등록 ---
font_registered = False
for font_path in [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]:
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("Korean", font_path))
            font_registered = True
            break
        except Exception:
            continue
FONT = "Korean" if font_registered else "Helvetica"

# --- 스타일 ---
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
}


def make_table(data, col_widths=None):
    ts_body = ParagraphStyle("tcell", fontName=FONT, fontSize=11, leading=15, textColor=colors.HexColor("#333333"))
    ts_header = ParagraphStyle("thcell", fontName=FONT, fontSize=11, leading=15, textColor=colors.white)
    formatted = []
    for i, row in enumerate(data):
        fmt_row = []
        for cell in row:
            s = ts_header if i == 0 else ts_body
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


def add_image(elements, path, max_width=150*mm, max_height=180*mm):
    if os.path.exists(path):
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


def generate_individual_cm_images(charts_dir):
    """혼동행렬 개별 이미지 5개 생성 (2-2-1 레이아웃용)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 모델별 혼동행렬 데이터 (마지막 Fold 기준, 실측값)
    cm_data = {
        'Random Forest': [[161, 24], [55, 19]],
        'Logistic Regression': [[172, 13], [52, 22]],
        'SVM': [[168, 17], [49, 25]],
        'XGBoost': [[161, 24], [48, 26]],
        'KNN': [[144, 41], [50, 24]],
    }

    # 실제 차트에서 데이터 읽기 시도
    cm_chart = os.path.join(charts_dir, "chart3_confusion_matrices.png")
    if not os.path.exists(cm_chart):
        return False

    try:
        font_path_candidates = [
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        ]
        from matplotlib import font_manager
        for fp in font_path_candidates:
            if os.path.exists(fp):
                font_manager.fontManager.addfont(fp)
                plt.rcParams['font.family'] = 'AppleGothic'
                break
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    cmap = plt.cm.Blues
    for name, cm in cm_data.items():
        cm = np.array(cm)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap=cmap, aspect='auto')
        ax.set_title(name, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('예측', fontsize=12)
        ax.set_ylabel('실제', fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['정상', '이상'], fontsize=11)
        ax.set_yticklabels(['정상', '이상'], fontsize=11)
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=16, fontweight='bold', color=color)
        plt.tight_layout()
        safe_name = name.replace(' ', '_')
        plt.savefig(os.path.join(charts_dir, f'cm_{safe_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    return True


def build_pdf():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    charts_dir = os.path.join(os.path.dirname(base_dir), "charts")
    output_path = os.path.join(base_dir, "2주차_보고서.pdf")

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
    elements.append(Paragraph("제출 2 - 모델 확장 &amp; 비교", styles["subtitle"]))
    elements.append(Paragraph("머신러닝 학기 프로젝트", styles["subtitle"]))
    elements.append(Spacer(1, 10*mm))
    elements.append(Paragraph("조장 : 20232501 임태후&nbsp;&nbsp;&nbsp;&nbsp;조원 : 20232514 유종헌", styles["subtitle"]))
    elements.append(PageBreak())

    # ===== 1. 모델 선택 이유 =====
    elements.append(Paragraph("1. 모델 선택 이유", styles["h1"]))

    elements.append(Paragraph("1.1 1주차 한계와 개선 방향", styles["h2"]))
    elements.append(Paragraph(
        "1주차에서는 Random Forest 단일 모델로 이상 변동 감지를 시도했으나, "
        "다음과 같은 한계가 확인되었다.",
        styles["body"]
    ))
    elements.append(make_table([
        ["한계점", "상세 설명", "2주차 대응"],
        ["시간 순서 무시",
         "데이터를 무작위로 섞어 나누었기 때문에 미래 데이터가 학습에 포함되는 "
         "데이터 누출(data leakage)이 발생하였다. "
         "이로 인해 1주차 성능(F1: 0.393)은 현실에서 재현 불가능한 부풀려진 수치였다.",
         "TimeSeriesSplit 5-Fold로 항상 과거→미래 순서 보장"],
        ["단일 모델",
         "Random Forest만 사용하여 해당 모델이 최적인지 판단할 근거가 없었다. "
         "데이터 특성에 따라 다른 알고리즘이 더 좋은 성능을 보일 수 있다.",
         "5개 모델(RF, LR, SVM, XGBoost, KNN) 비교 분석"],
        ["Recall 부족",
         "Recall 28.3%로 실제 이상변동 100건 중 72건을 놓쳤다. "
         "데이터 불균형(정상 64.5% vs 이상 35.5%)으로 모델이 '정상'에 편향되었다.",
         "다양한 모델로 Recall 개선 시도"],
    ], col_widths=[W*0.13, W*0.55, W*0.32]))

    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph("1.2 선택한 5개 모델과 선택 이유", styles["h2"]))
    elements.append(make_table([
        ["모델", "유형", "선택 이유"],
        ["Random Forest",
         "앙상블 (Bagging)",
         "1주차 기준 모델. TimeSeriesSplit 적용 시 성능 변화를 확인하기 위해 재실행"],
        ["Logistic Regression",
         "선형 분류",
         "가장 단순한 분류 모델로서 성능 하한선(baseline)을 설정. 해석이 쉬워 변수별 영향도 분석에 유리"],
        ["SVM",
         "커널 기반 분류",
         "RBF 커널로 비선형 경계를 학습 가능. 고차원 데이터에서 효과적이며, 소규모 데이터에 강점"],
        ["XGBoost",
         "앙상블 (Boosting)",
         "Gradient Boosting 기반으로 일반적으로 분류 성능이 가장 높음. 불균형 데이터 처리에 강점"],
        ["KNN",
         "거리 기반 분류",
         "학습 없이 유사 패턴을 직접 비교. 데이터 분포에 의존하므로 시계열 특성이 반영되는지 확인"],
    ], col_widths=[W*0.17, W*0.18, W*0.65]))

    elements.append(PageBreak())
    elements.append(Paragraph("1.3 사용 도구: scikit-learn &amp; XGBoost", styles["h2"]))
    elements.append(Paragraph(
        "scikit-learn(sklearn)은 Python 환경에서 머신러닝 모델의 학습, 예측, 평가를 "
        "간편하게 수행할 수 있도록 제공되는 라이브러리이다. "
        "본 프로젝트에서는 다음과 같은 기능을 활용하였다.",
        styles["body"]
    ))
    elements.append(Paragraph("- 분류 모델: Random Forest, Logistic Regression, SVM, KNN, XGBoost", styles["bullet"]))
    elements.append(Paragraph("- 전처리: StandardScaler를 이용한 데이터 정규화", styles["bullet"]))
    elements.append(Paragraph("- 평가 지표: Accuracy, Precision, Recall, F1-Score, ROC-AUC", styles["bullet"]))
    elements.append(Paragraph("- 교차검증: TimeSeriesSplit (시계열 전용 교차검증)", styles["bullet"]))

    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph("1.4 평가 방법: TimeSeriesSplit", styles["h2"]))
    elements.append(Paragraph(
        "TimeSeriesSplit은 시계열 데이터 전용 교차검증 방법으로, "
        "항상 과거 데이터로 학습하고 미래 데이터로 테스트한다. "
        "Fold가 진행될수록 학습 데이터가 점점 늘어나며, "
        "미래 데이터가 학습에 절대 포함되지 않으므로 데이터 누출(data leakage)을 방지할 수 있다. "
        "이를 통해 실제 투자 환경과 동일한 조건에서 모델을 평가하였다.",
        styles["body"]
    ))
    elements.append(make_table([
        ["Fold", "학습 기간 (일)", "테스트 기간 (일)", "설명"],
        ["Fold 1", "262", "259", "최소 학습량으로 예측 (가장 어려운 조건)"],
        ["Fold 2", "521", "259", "약 2년치 학습"],
        ["Fold 3", "780", "259", "약 3년치 학습"],
        ["Fold 4", "1,039", "259", "약 4년치 학습"],
        ["Fold 5", "1,298", "259", "최대 학습량 (가장 유리한 조건)"],
    ], col_widths=[W*0.12, W*0.22, W*0.22, W*0.44]))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        "<b>Fold를 1→5로 점진적으로 확장하는 이유:</b>",
        styles["body"]
    ))
    elements.append(Paragraph(
        "TimeSeriesSplit은 Fold가 진행될수록 학습 데이터가 점점 늘어나는 구조이다. "
        "이를 통해 단순히 모델의 평균 성능만 측정하는 것이 아니라, "
        "학습 데이터가 많아질수록 모델의 성능이 개선되는지, "
        "그리고 특정 시기(예: 전쟁, 위기)의 데이터가 포함되었을 때 성능이 어떻게 변하는지를 "
        "추적할 수 있다. 즉, 모델의 <b>안정성(stability)</b>과 "
        "<b>데이터 양에 따른 학습 효과</b>를 동시에 검증하는 것이 핵심 목적이다.",
        styles["body"]
    ))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        "또한 SVM, KNN, Logistic Regression 등 스케일에 민감한 모델을 위해 "
        "StandardScaler를 적용하였다. 학습 데이터를 기준으로 정규화한 뒤, "
        "같은 기준을 테스트 데이터에도 적용하여 데이터 누출을 방지하였다.",
        styles["body"]
    ))

    elements.append(PageBreak())

    # ===== 2. 성능 비교 =====
    elements.append(Paragraph("2. 성능 비교", styles["h1"]))

    elements.append(Paragraph("2.1 모델별 평균 성능 (5-Fold)", styles["h2"]))
    elements.append(make_table([
        ["모델", "Accuracy", "Precision", "Recall", "F1-Score"],
        ["Random Forest",    "0.635", "0.379", "0.245", "0.289"],
        ["Logistic Regression", '<font color="red"><b>0.676</b></font>', "0.435", "0.144", "0.205"],
        ["SVM",              "0.665", '<font color="red"><b>0.461</b></font>', "0.187", "0.258"],
        ["<b>XGBoost</b>",   "0.632", "0.406", '<font color="red"><b>0.286</b></font>', '<font color="red"><b>0.324</b></font>'],
        ["KNN",              "0.602", "0.372", "0.285", "0.317"],
    ], col_widths=[W*0.25, W*0.17, W*0.17, W*0.17, W*0.17]))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        "→ F1-Score 기준 XGBoost가 0.324로 최고 성능. Recall 기준으로도 XGBoost가 0.286으로 1위.",
        styles["body_indent"]
    ))

    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph("2.2 모델별 성능 비교 차트", styles["h2"]))
    add_image(elements, os.path.join(charts_dir, "chart1_model_comparison.png"))

    elements.append(PageBreak())

    elements.append(Paragraph("2.3 ROC 커브 비교", styles["h2"]))
    elements.append(Paragraph(
        "ROC 커브는 임계값(threshold)을 변화시키면서 True Positive Rate와 "
        "False Positive Rate의 관계를 보여준다. AUC가 1에 가까울수록 좋은 모델이다.",
        styles["body"]
    ))
    add_image(elements, os.path.join(charts_dir, "chart2_roc_curves.png"))
    elements.append(Spacer(1, 2*mm))
    elements.append(make_table([
        ["모델", "AUC", "해석"],
        ["SVM", '<font color="red"><b>0.675</b></font>', "확률 기반 판단에서 가장 우수"],
        ["Random Forest", "0.631", "앙상블 효과로 안정적"],
        ["XGBoost", "0.621", "F1은 최고지만 AUC는 중간"],
        ["Logistic Regression", "0.614", "단순 모델 대비 준수"],
        ["KNN", "0.596", "가장 낮은 AUC"],
    ], col_widths=[W*0.25, W*0.12, W*0.63]))

    elements.append(PageBreak())

    # ===== 혼동 행렬 전용 페이지 (2-2-1 레이아웃) =====
    generate_individual_cm_images(charts_dir)

    elements.append(Paragraph("2.4 혼동 행렬 비교 (마지막 Fold)", styles["h2"]))
    elements.append(Paragraph(
        "각 모델의 혼동 행렬을 통해 정상/이상변동 분류 결과를 시각적으로 확인한다. "
        "이상변동 감지가 목적이므로 각 칸의 의미는 다음과 같다: "
        "좌상단(TN)은 정상을 정상으로 분류한 것으로 맞는 예측이지만, 이 값이 크다고 좋은 것이 아니라 "
        "모델이 '정상'만 예측하는 편향의 신호일 수 있다. "
        "우하단(TP)은 이상변동을 이상으로 감지한 것으로 이 값이 클수록 좋으며 Recall과 직결된다. "
        "좌하단(FN)은 실제 이상변동을 정상으로 놓친 것으로 가장 치명적인 오류이다. "
        "우상단(FP)은 정상을 이상으로 오판한 허위 경보이다.",
        styles["body"]
    ))
    elements.append(Spacer(1, 2*mm))

    cm_img_w = W * 0.43
    cm_img_h = cm_img_w * 0.80

    def cm_image(name):
        p = os.path.join(charts_dir, f'cm_{name}.png')
        if os.path.exists(p):
            img = Image(p, width=cm_img_w, height=cm_img_h)
            img.hAlign = "CENTER"
            return img
        return Paragraph("(이미지 없음)", styles["body"])

    # 1행: Random Forest, Logistic Regression
    row1 = Table([[cm_image('Random_Forest'), cm_image('Logistic_Regression')]],
                 colWidths=[W*0.5, W*0.5])
    row1.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(row1)
    elements.append(Spacer(1, 2*mm))

    # 2행: SVM, XGBoost
    row2 = Table([[cm_image('SVM'), cm_image('XGBoost')]],
                 colWidths=[W*0.5, W*0.5])
    row2.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(row2)
    elements.append(Spacer(1, 2*mm))

    # 3행: KNN (가운데 정렬)
    row3 = Table([[cm_image('KNN')]],
                 colWidths=[W])
    row3.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(row3)
    elements.append(Spacer(1, 2*mm))

    elements.append(Paragraph(
        "→ XGBoost가 이상변동(우하단) 26건으로 가장 많이 감지하였으며, "
        "SVM이 25건으로 2위를 기록하였다.",
        styles["body_indent"]
    ))

    elements.append(PageBreak())

    elements.append(Paragraph("2.5 Fold별 F1-Score 변화 (모델 안정성)", styles["h2"]))
    elements.append(Paragraph(
        "Fold가 진행될수록 학습 데이터가 많아진다. 안정적인 모델은 Fold 간 성능 변동이 작아야 한다.",
        styles["body"]
    ))
    add_image(elements, os.path.join(charts_dir, "chart4_fold_f1.png"))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        "→ 모든 모델이 Fold 2에서 급등 후 Fold 3~4에서 하락하는 패턴을 보인다. "
        "이는 2021~2022년 러시아-우크라이나 전쟁 기간의 높은 변동성이 학습에 포함되면서 "
        "이후 안정기 예측이 어려워지는 현상으로 해석된다.",
        styles["body_indent"]
    ))

    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph("2.6 1주차 vs 2주차 비교 (Random Forest 기준)", styles["h2"]))
    elements.append(make_table([
        ["지표", "1주차 (Random Split)", "2주차 (TimeSeriesSplit)", "변화"],
        ["Accuracy",  "0.683", "0.635", "↓ 0.048"],
        ["Precision", "0.640", "0.379", "↓ 0.261"],
        ["Recall",    "0.283", "0.245", "↓ 0.038"],
        ["F1-Score",  "0.393", "0.289", "↓ 0.103"],
    ], col_widths=[W*0.20, W*0.27, W*0.30, W*0.15]))
    elements.append(Spacer(1, 2*mm))
    elements.append(Paragraph(
        "→ 모든 지표가 하락하였다. 이는 TimeSeriesSplit이 더 현실적인 평가 방법이기 때문이며, "
        "1주차의 높은 성능은 Random Split으로 인한 미래 데이터 누출(data leakage)로 "
        "부풀려진 결과였음을 의미한다.",
        styles["body_indent"]
    ))

    elements.append(PageBreak())

    elements.append(Paragraph("2.7 회귀(수익률 예측) 성능 비교", styles["h2"]))
    elements.append(Paragraph(
        "분류(이상변동 감지) 외에, 각 모델의 회귀 버전을 사용하여 원유 수익률 예측 성능도 비교하였다. "
        "XGBRegressor, RandomForestRegressor, SVR, KNeighborsRegressor, LinearRegression을 "
        "동일한 TimeSeriesSplit 5-Fold 조건에서 평가하였다.",
        styles["body"]
    ))
    elements.append(Spacer(1, 2*mm))
    elements.append(make_table([
        ["모델", "MAE", "RMSE", "R²", "해석"],
        ['<font color="red"><b>SVM (SVR)</b></font>',
         '<font color="red"><b>1.770</b></font>',
         '<font color="red"><b>2.354</b></font>',
         '<font color="red"><b>-0.029</b></font>',
         "회귀 성능 1위. R²가 0에 가장 가까움"],
        ["KNN", "1.962", "2.558", "-0.222", "MAE 기준 2위"],
        ["XGBoost", "1.956", "2.648", "-0.303", "분류 1위이나 회귀는 3위"],
        ["Random Forest", "2.067", "3.424", "-1.239", "회귀에서 가장 불안정"],
        ["Linear Regression", "2.376", "3.100", "-0.796", "선형 모델의 한계"],
    ], col_widths=[W*0.15, W*0.10, W*0.10, W*0.13, W*0.52]))
    elements.append(Spacer(1, 3*mm))
    elements.append(Paragraph(
        "→ 회귀 성능은 SVR이 1위이나, 모든 모델의 R²가 음수로 "
        "현재 Feature만으로는 수익률 예측이 충분하지 않다.",
        styles["body_indent"]
    ))
    elements.append(Spacer(1, 3*mm))
    elements.append(Paragraph("<b>회귀 성능의 한계와 XGBoost 선정 이유:</b>", styles["body"]))
    elements.append(Paragraph(
        "- R²가 모두 음수 → 단순 평균값 예측보다 못하며, 시차 Feature와 기술 지표 추가가 필요",
        styles["bullet"]
    ))
    elements.append(Paragraph(
        "- SVR이 회귀 1위이나, 학습 속도가 느리고 대규모 데이터에 불리하며 "
        "분류 성능(Recall)은 XGBoost보다 낮음",
        styles["bullet"]
    ))
    elements.append(Paragraph(
        "- XGBoost는 분류(F1, Recall 1위) + 회귀(3위)를 하나의 프레임워크에서 모두 처리 가능하고, "
        "scale_pos_weight로 클래스 불균형 처리까지 내장되어 있어 종합적으로 최적",
        styles["bullet"]
    ))

    elements.append(PageBreak())

    # ===== 3. 결과 해석 =====
    elements.append(Paragraph("3. 결과 해석", styles["h1"]))

    elements.append(Paragraph("3.1 최적 모델: XGBoost", styles["h2"]))
    elements.append(Paragraph(
        "F1-Score(0.324)와 Recall(0.286) 모두에서 XGBoost가 최고 성능을 기록하였다. "
        "이상 변동 감지라는 목적상, 이상변동을 놓치지 않는 것(Recall)이 가장 중요한 지표이므로 "
        "XGBoost가 현재 데이터에 가장 적합한 모델이다.",
        styles["body"]
    ))
    elements.append(Spacer(1, 3*mm))
    elements.append(Paragraph(
        "<b>XGBoost가 3가지 과제 모두에 강한 이유:</b>",
        styles["body"]
    ))
    elements.append(Spacer(1, 2*mm))

    elements.append(Paragraph("<b>① 이상변동 감지 (Classification: 이상 vs 정상)</b>", styles["body"]))
    elements.append(Paragraph(
        "- 클래스 불균형 처리: scale_pos_weight 파라미터를 통해 소수 클래스(이상변동)를 "
        "틀렸을 때 벌점을 더 크게 부여할 수 있어, 모델이 이상변동을 더 적극적으로 찾게 됨",
        styles["bullet"]
    ))
    elements.append(Paragraph(
        "- 부스팅 방식: 이전 트리가 놓친 이상변동 샘플에 집중하여 다음 트리를 생성하므로, "
        "1주차 핵심 약점이었던 Recall(28.3%) 개선에 유리",
        styles["bullet"]
    ))
    elements.append(Spacer(1, 2*mm))

    elements.append(Paragraph("<b>② 가격 방향 예측 (Classification: 상승 vs 하락)</b>", styles["body"]))
    elements.append(Paragraph(
        "- 비선형 패턴 학습: VIX 급등 + GPR 상승 시 하락, 같은 복합 조건을 트리 구조로 자연스럽게 포착",
        styles["bullet"]
    ))
    elements.append(Paragraph(
        "- 순차 보정(Gradient Boosting): Random Forest가 여러 트리를 독립적으로 만들어 평균내는 방식이라면, "
        "XGBoost는 이전 트리의 오류를 교정하면서 학습하여 방향 예측 정확도 향상에 유리",
        styles["bullet"]
    ))
    elements.append(Spacer(1, 2*mm))

    elements.append(Paragraph("<b>③ 수익률 예측 (Regression: 몇 % 변동)</b>", styles["body"]))
    elements.append(Paragraph(
        "- XGBRegressor로 회귀 분석도 가능하여, 동일한 프레임워크에서 분류와 회귀를 모두 처리",
        styles["bullet"]
    ))
    elements.append(Paragraph(
        "- 정규화(L1/L2) 내장: Random Forest에 없는 과적합 방지 기능이 있어 "
        "노이즈가 많은 수익률 데이터에서도 안정적 예측 가능",
        styles["bullet"]
    ))
    elements.append(Spacer(1, 2*mm))

    elements.append(Paragraph("<b>공통 강점:</b>", styles["body"]))
    elements.append(Paragraph(
        "- 학습률(Learning Rate) 조절로 세밀한 튜닝이 가능하다 (Random Forest에는 없는 기능)",
        styles["bullet"]
    ))
    elements.append(Paragraph(
        "- 조기 종료(Early Stopping) 기능으로 모델이 과적합되기 전에 자동으로 학습을 중단할 수 있다",
        styles["bullet"]
    ))
    elements.append(Paragraph(
        "- 변수 중요도(Feature Importance)를 통해 3가지 과제에서 어떤 변수가 중요한지 일관되게 비교할 수 있다",
        styles["bullet"]
    ))

    elements.append(PageBreak())
    elements.append(Paragraph("3.2 모델별 특성 분석", styles["h2"]))
    elements.append(make_table([
        ["모델", "강점", "약점", "종합 평가"],
        ["XGBoost",
         "F1, Recall 최고. 불균형 데이터에 강함",
         "Accuracy는 RF보다 낮음",
         "★ 이상변동 감지에 가장 적합"],
        ["KNN",
         "Recall 2위(0.285). 구현이 단순",
         "Accuracy 최하위. 안정성 부족",
         "보조 모델로 활용 가능"],
        ["Random Forest",
         "Fold 간 성능이 비교적 안정적",
         "Recall이 낮아 이상변동 놓침",
         "안정성은 좋으나 감지력 부족"],
        ["SVM",
         "AUC 최고(0.675). Precision 1위",
         "Recall 낮음. 학습 속도 느림",
         "확률 기반 경보에 활용 가능"],
        ["Logistic Regression",
         "Accuracy 최고(0.676). 해석 용이",
         "Recall 최하위(0.144)",
         "이상변동 감지에는 부적합"],
    ], col_widths=[W*0.15, W*0.30, W*0.25, W*0.30]))

    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph("3.3 왜 모든 모델의 성능이 낮은가?", styles["h2"]))
    elements.append(Paragraph(
        "5개 모델 모두 F1-Score가 0.2~0.3 수준으로, 이상변동 감지 성능이 충분하지 않다. "
        "이는 모델의 문제가 아니라 현재 Feature의 한계로 판단된다.",
        styles["body"]
    ))
    elements.append(Paragraph("- 당일 GPR/VIX 값만 사용, 시차(Lag) Feature 부재", styles["bullet"]))
    elements.append(Paragraph("- 이동평균, 볼린저밴드 등 기술적 지표 미포함", styles["bullet"]))
    elements.append(Paragraph("- 이상변동 기준이 ±2% 고정, 변동성 적응형 기준 필요", styles["bullet"]))

    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph("3.4 다음 단계 개선 계획", styles["h2"]))
    elements.append(make_table([
        ["개선 항목", "구체적 방법", "기대 효과"],
        ["시차 Feature 추가", "전일/전주 GPR, VIX 변화량을 Feature로 추가", "지연된 시장 반응 포착"],
        ["기술적 지표 추가", "이동평균(MA), 볼린저밴드, RSI 등", "가격 추세 정보 반영"],
        ["하이퍼파라미터 튜닝", "GridSearch / RandomSearch로 최적 파라미터 탐색", "모델 성능 극대화"],
        ["클래스 불균형 처리", "SMOTE, class_weight 조정", "Recall 개선"],
    ], col_widths=[W*0.22, W*0.40, W*0.38]))

    elements.append(PageBreak())

    # ===== 부록 =====
    elements.append(Paragraph("부록: 사용 도구 및 환경", styles["h1"]))
    elements.append(make_table([
        ["항목", "내용"],
        ["프로그래밍 언어", "Python 3.9"],
        ["실행 환경", "Google Colab"],
        ["데이터", "1주차 통합 데이터 (commodity_vix_gpr_data.xlsx, 1,558일)"],
        ["머신러닝", "scikit-learn (RF, LR, SVM, KNN), XGBoost"],
        ["교차검증", "TimeSeriesSplit (5-Fold)"],
        ["전처리", "StandardScaler"],
        ["시각화", "matplotlib"],
    ], col_widths=[W*0.25, W*0.75]))

    # --- 빌드 ---
    doc.build(elements)
    print(f"PDF 생성 완료: {output_path}")


if __name__ == "__main__":
    build_pdf()
