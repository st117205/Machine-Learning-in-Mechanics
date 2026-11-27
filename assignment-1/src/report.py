"""Build a short PDF report from saved results.

Run from assignment-1:  py -m src.report
"""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PDF_PATH = OUT / "report.pdf"

font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVuSans", str(font_dir / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("TitleRU", parent=styles["Title"], fontName="DejaVuSans-Bold",
                          fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle("H", parent=styles["Heading2"], fontName="DejaVuSans-Bold",
                          fontSize=11, leading=14, spaceBefore=9, spaceAfter=4))
styles.add(ParagraphStyle("Body", parent=styles["BodyText"], fontName="DejaVuSans",
                          fontSize=8.6, leading=11.5, spaceAfter=4))
styles.add(ParagraphStyle("Small", parent=styles["BodyText"], fontName="DejaVuSans",
                          fontSize=7.4, leading=9.5, spaceAfter=3))
styles.add(ParagraphStyle("Cell", fontName="DejaVuSans", fontSize=6.9, leading=8.4))
styles.add(ParagraphStyle("CellHead", fontName="DejaVuSans-Bold", fontSize=6.9, leading=8.4,
                          textColor=colors.white))

NOTES = {
    "ridge_poly2": "Лучший перенос, итоговая модель",
    "poly2_physical": "Почти то же, без Ridge",
    "poly3_physical": "Лучший CV и интерполяция; провал на U=71.3",
    "lasso_poly2": "L1 не помог",
    "elasticnet_poly2": "Хуже Ridge и Lasso",
    "ols_raw": "Простая база на сырых признаках",
    "ols_physical": "Логарифмы по отдельности не помогли",
    "ols_strouhal": "То же для f·c/U",
    "ols_physical_u2": "U² не помог",
    "baseline": "Только среднее значение",
}


def p(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, styles[style])


def fmt(value, digits: int = 3) -> str:
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def table(data: pd.DataFrame, widths: list[float], digits: int = 3) -> Table:
    head = [p(html.escape(str(c)), "CellHead") for c in data.columns]
    body = [
        [p(html.escape(fmt(v, digits) if isinstance(v, (int, float)) else str(v)), "Cell")
         for v in row]
        for row in data.itertuples(index=False)
    ]
    t = Table([head] + body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F5597")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7B7B7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def figure(name: str, width: float) -> Image:
    image = Image(str(OUT / name))
    image.drawHeight = image.imageHeight * width / image.imageWidth
    image.drawWidth = width
    return image


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("DejaVuSans", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(18 * mm, 9 * mm, "Assignment 1 — Airfoil Self-Noise")
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"стр. {doc.page}")
    canvas.restoreState()


def build_report() -> Path:
    comparison = pd.read_csv(OUT / "model_comparison.csv")
    selection = pd.read_csv(OUT / "model_selection.csv")
    degree = pd.read_csv(OUT / "cv_degree.csv")
    boot = pd.read_csv(OUT / "bootstrap_coefficients.csv")
    scale = pd.read_csv(OUT / "target_scale_check.csv")

    final_name = selection.loc[0, "model"]
    final = comparison[comparison["model"] == final_name].iloc[0]

    final_metrics = pd.DataFrame([
        {"subset": "train", "MAE": final["train_mae"], "RMSE": final["train_rmse"], "R²": final["train_r2"]},
        {"subset": "CV", "MAE": final["cv_mae"], "RMSE": final["cv_rmse_oof"], "R²": final["cv_r2"]},
        {"subset": "test", "MAE": final["test_mae"], "RMSE": final["test_rmse"], "R²": final["test_r2"]},
    ])

    comparison_view = selection[[
        "model", "cv_rmse", "rmse_U55.5", "rmse_U71.3", "worst_rmse"
    ]].copy()
    comparison_view["note"] = comparison_view["model"].map(NOTES)
    comparison_view.columns = ["модель", "CV RMSE", "U=55.5", "U=71.3", "худший", "где лучше"]

    degree_view = degree.rename(columns={
        "degree": "степень", "cv_rmse": "CV RMSE",
        "train_rmse": "train RMSE", "test_rmse": "test RMSE",
    })
    boot_view = pd.DataFrame({
        "признак": boot["feature"],
        "медиана": boot["median"],
        "5–95%": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(boot["q05"], boot["q95"])],
        "P(>0)": boot["positive_fraction"],
    })
    units = pd.DataFrame([
        {"переменная": "f, Гц", "диапазон": "200–20000"},
        {"переменная": "α, град", "диапазон": "0–22.2"},
        {"переменная": "c, м", "диапазон": "0.0254–0.3048"},
        {"переменная": "U, м/с", "диапазон": "31.7 / 39.6 / 55.5 / 71.3"},
        {"переменная": "δ*, м", "диапазон": "0.00040–0.05841"},
        {"переменная": "SPL, дБ", "диапазон": "103.38–140.99"},
    ])

    story: list = []
    story.append(p("Airfoil Self-Noise: сравнение моделей", "TitleRU"))
    story.append(p(
        f"Итог: <b>{final_name}</b>, alpha = 0.01. Выбрана по худшему RMSE на отложенных скоростях.",
        "Body",
    ))
    story.append(table(final_metrics, [110, 90, 90, 90]))
    story.append(Spacer(1, 6))
    story.append(table(comparison_view, [105, 58, 56, 56, 58, 157]))
    story.append(p(
        "Степень 3 лучше на CV и на интерполяции, но хуже на U=71.3. "
        "Для переноса выбрана более простая модель с Ridge.", "Small"
    ))

    story.append(PageBreak())
    story.append(p("Результаты", "H"))
    story.append(figure("cv_model_comparison.png", 460))
    story.append(p("Рис. 1. CV RMSE на обучающей части.", "Small"))
    story.append(Spacer(1, 4))
    story.append(figure("measured_vs_predicted.png", 350))
    story.append(p("Рис. 2. Измеренный и предсказанный SPL, test.", "Small"))

    story.append(PageBreak())
    story.append(p("Сложность и регуляризация", "H"))
    story.append(table(degree_view, [85, 120, 125, 125], digits=3))
    story.append(p("Степень 1 недообучает; степень 3 лучше на CV, но хуже на переносе.", "Small"))
    story.append(figure("cv_regularization.png", 400))
    story.append(p("Рис. 3. CV RMSE против alpha.", "Small"))
    story.append(Spacer(1, 4))
    story.append(p("Устойчивость коэффициентов, бутстреп 400 раз", "H"))
    story.append(table(boot_view, [95, 80, 130, 75], digits=3))
    story.append(p(
        "В параметризации f·c/U знак log(c) меняется: коэффициенты чувствительны к коррелированным признакам.",
        "Small",
    ))
    story.append(p(
        f"Шкала цели: SPL = 20·log10(p/p0); обучение на SPL/20. "
        f"Проверка: test RMSE {scale.loc[0, 'test_rmse_db']:.4f} дБ в dB и "
        f"{scale.loc[1, 'test_rmse_db']:.4f} дБ в log-pressure.", "Small"
    ))

    story.append(PageBreak())
    story.append(p("Остатки и провал модели", "H"))
    story.append(figure("residual_diagnostics.png", 470))
    story.append(p(
        "На отложенном U=71.3 м/с при α=22.2° и c=0.0254 м модель недооценивает SPL на 16.9 дБ.",
        "Small",
    ))
    story.append(Spacer(1, 5))
    story.append(p("Данные", "H"))
    story.append(table(units, [150, 260], digits=3))
    story.append(p("UCI dataset 291, Brooks et al. (1989), CC BY 4.0.", "Small"))
    story.append(p("Гипотезы до обучения", "H"))
    for text in [
        "H1. Шум растёт с U — направление подтвердилось.",
        "H2. Зависимость от log f нелинейна — подтвердилось.",
        "H3. f·c/U объединяет спектры — частично.",
        "H4. Рост δ* снижает шум — знак подтвердился, но δ* коррелирует с α.",
    ]:
        story.append(p(html.escape(text), "Small"))

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="Airfoil Self-Noise — Assignment 1", author="ST117205",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return PDF_PATH


if __name__ == "__main__":
    print(build_report())
