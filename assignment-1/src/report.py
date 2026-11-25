"""Build a readable PDF report from the saved result tables and figures.

Run from assignment-1:  py -m src.report
The script only formats files already produced by ``src.main``; it does not
fit or select models.
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


def _register_fonts() -> None:
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))


_register_fonts()

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    "RussianTitle", parent=styles["Title"], fontName="DejaVuSans-Bold",
    fontSize=18, leading=23, alignment=TA_CENTER, spaceAfter=10,
))
styles.add(ParagraphStyle(
    "RussianH1", parent=styles["Heading1"], fontName="DejaVuSans-Bold",
    fontSize=13, leading=16, spaceBefore=12, spaceAfter=6,
))
styles.add(ParagraphStyle(
    "RussianH2", parent=styles["Heading2"], fontName="DejaVuSans-Bold",
    fontSize=10.5, leading=13, spaceBefore=9, spaceAfter=4,
))
styles.add(ParagraphStyle(
    "RussianBody", parent=styles["BodyText"], fontName="DejaVuSans",
    fontSize=9, leading=12.5, spaceAfter=5,
))
styles.add(ParagraphStyle(
    "RussianSmall", parent=styles["BodyText"], fontName="DejaVuSans",
    fontSize=7.7, leading=10, spaceAfter=4,
))
styles.add(ParagraphStyle(
    "Cell", fontName="DejaVuSans", fontSize=7.2, leading=8.8,
))
styles.add(ParagraphStyle(
    "CellHeader", fontName="DejaVuSans-Bold", fontSize=7.2, leading=8.8,
    textColor=colors.white,
))


def p(text: str, style: str = "RussianBody") -> Paragraph:
    return Paragraph(text, styles[style])


def fmt(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def table(data: pd.DataFrame, widths: list[float], digits: int = 3) -> Table:
    header = [p(html.escape(str(col)), "CellHeader") for col in data.columns]
    body = [
        [p(html.escape(fmt(value, digits) if isinstance(value, (int, float)) else str(value)), "Cell")
         for value in row]
        for row in data.itertuples(index=False)
    ]
    result = Table([header] + body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F5597")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7B7B7")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return result


def figure(name: str, width: float) -> Image:
    image = Image(str(OUT / name))
    scale = width / image.imageWidth
    image.drawWidth = width
    image.drawHeight = image.imageHeight * scale
    return image


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("DejaVuSans", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(18 * mm, 10 * mm, "Assignment 1 — Airfoil Self-Noise — отчёт")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"стр. {doc.page}")
    canvas.restoreState()


def build_report() -> Path:
    comparison = pd.read_csv(OUT / "model_comparison.csv")
    selected = comparison.iloc[0]
    holdout = pd.read_csv(OUT / "holdout_metrics.csv")
    degree = pd.read_csv(OUT / "cv_degree.csv")
    grouped = pd.read_csv(OUT / "grouped_cv_metrics.csv")
    boot = pd.read_csv(OUT / "bootstrap_coefficients.csv")
    boot_st = pd.read_csv(OUT / "bootstrap_coefficients_strouhal.csv")
    corr = pd.read_csv(OUT / "feature_correlations.csv", index_col=0)

    final_metrics = pd.DataFrame([
        {
            "subset": "train",
            "MAE, dB": selected["train_mae"],
            "RMSE, dB": selected["train_rmse"],
            "R2": selected["train_r2"],
        },
        {
            "subset": "CV (out-of-fold)",
            "MAE, dB": selected["cv_mae"],
            "RMSE, dB": selected["cv_rmse_oof"],
            "R2": selected["cv_r2"],
        },
        {
            "subset": "test",
            "MAE, dB": selected["test_mae"],
            "RMSE, dB": selected["test_rmse"],
            "R2": selected["test_r2"],
        },
    ])

    model_compact = comparison[[
        "model", "cv_rmse", "test_rmse", "test_r2"
    ]].rename(columns={
        "model": "модель", "cv_rmse": "CV RMSE, дБ",
        "test_rmse": "test RMSE, дБ", "test_r2": "test R²",
    })

    hold_compact = holdout[[
        "held_out_U_m_s", "model", "n_test", "mae", "rmse", "r2"
    ]].rename(columns={
        "held_out_U_m_s": "отлож. U, м/с", "model": "модель",
        "n_test": "n", "mae": "MAE, дБ", "rmse": "RMSE, дБ", "r2": "R²",
    })

    boot_view = boot.rename(columns={
        "feature": "признак", "median": "медиана", "q05": "5%", "q95": "95%",
        "positive_fraction": "P(>0)", "sd": "СКО",
    })
    boot_view = boot_view[["признак", "медиана", "5%", "95%", "P(>0)", "СКО"]]

    corr_pairs = pd.DataFrame([
        {
            "пара": "log10(δ*) — угол атаки",
            "r": corr.loc["log_delta", "angle_deg"],
        },
        {
            "пара": "log10(f) — log10(f c/U)",
            "r": corr.loc["log_f", "log_fc_u"],
        },
        {
            "пара": "log10(c) — угол атаки",
            "r": corr.loc["log_c", "angle_deg"],
        },
    ])

    story: list = []
    story.append(p("Простая линейная модель собственного шума аэродинамического профиля", "RussianTitle"))
    story.append(p(
        "Assignment 1. NASA/UCI Airfoil Self-Noise. Отчёт по данным, гипотезам, "
        "сравнению моделей и проверке на отложенных режимах.",
        "RussianSmall",
    ))
    story.append(Spacer(1, 6))

    story.append(p("1. Что сделано коротко", "RussianH1"))
    story.append(p(
        "Изучен набор данных NASA/UCI на 1503 измерения. До обучения сформулированы "
        "гипотезы о влиянии скорости, частоты, хорды и толщины пограничного слоя. "
        "Затем сравнены: константный baseline, МНК/OLS на сырых и физически "
        "преобразованных признаках, полиномы степени 2 и 3, Ridge, Lasso и Elastic Net. "
        "Все преобразования и стандартизация находились внутри конвейера и "
        "пересчитывались на каждом CV-разбиении. Выбор степени и α выполнялся "
        "только по обучающей части; тест использовался один раз в конце."
    ))
    story.append(p(
        "Лучшая по обычной CV модель — полином степени 3 от физических признаков: "
        f"test MAE {selected['test_mae']:.3f} дБ, RMSE {selected['test_rmse']:.3f} дБ, "
        f"R² {selected['test_r2']:.3f}. Модель остаётся линейной по коэффициентам: "
        "нелинейность появилась только из заранее построенных признаков."
    ))
    story.append(p(
        "Главная оговорка: на интерполяции по скорости модель хороша, но при "
        "исключении U = 71.3 м/с её RMSE вырастает до 4.929 дБ, а R² падает до 0.505. "
        "Систематический провал наблюдается при высоком угле атаки и малой хорде. "
        "Если важна передача на новые режимы, более простой полином степени 2 "
        "оказывается устойчивее.",
    ))

    story.append(p("2. Данные и единицы", "RussianH1"))
    story.append(p(
        "Источник: UCI Machine Learning Repository, dataset 291, "
        "https://doi.org/10.24432/C5VW2C. Данные Brooks, Pope & Marcolini (1989), "
        "NASA: аэродинамическая труба с безэховой камерой, профиль NACA 0012, "
        "размах профиля и положение микрофона постоянны. Лицензия CC BY 4.0. "
        "Файл UCI: 1503 строки, 6 колонок, без пропусков и полных дублей."
    ))
    units = pd.DataFrame([
        {"переменная": "f", "ед.": "Гц", "диапазон": "200 … 20000", "смысл": "частота"},
        {"переменная": "α", "ед.": "град", "диапазон": "0 … 22.2", "смысл": "угол атаки"},
        {"переменная": "c", "ед.": "м", "диапазон": "0.0254 … 0.3048", "смысл": "хорда"},
        {"переменная": "U", "ед.": "м/с", "диапазон": "31.7 / 39.6 / 55.5 / 71.3", "смысл": "скорость потока"},
        {"переменная": "δ*", "ед.": "м", "диапазон": "0.00040 … 0.05841", "смысл": "толщина вытеснения"},
        {"переменная": "SPL", "ед.": "дБ", "диапазон": "103.38 … 140.99", "смысл": "цель: scaled SPL"},
    ])
    story.append(table(units, [70, 55, 135, 245]))
    story.append(p(
        "Цель уже дана в децибелах. Поэтому коэффициенты модели — эмпирические "
        "наклоны в дБ на стандартное отклонение признака, а не физические "
        "амплитуды давления. UCI не раскрывает точную нормализацию 'scaled SPL', "
        "поэтому это неизвестное явно оставлено без догадок.",
        "RussianSmall",
    ))
    story.append(p(
        "Структура: 106 физических конфигураций (α, c, U, δ*), внутри каждой "
        "измерено 8–19 частот. Пара (α, c, δ*) не повторяется при разных U, "
        "то есть δ* запутан со скоростью. Обычный KFold разносит частоты одной "
        "конфигурации по разным фолдам; дополнительно проверен GroupKFold, "
        "который держит всю конфигурацию вместе.",
        "RussianSmall",
    ))

    story.append(p("3. Гипотезы, записанные до обучения", "RussianH1"))
    hypotheses = [
        ("H1", "Уровень шума растёт с U; для шума задней кромки ожидается порядка U⁵.",
         "Направление подтвердилось: медиана β(log U) = +1.28 дБ/декада, знак "
         "положителен во всех 400 бутстреп-выборках. Количественный закон U⁵ "
         "не подтверждён в этой параметризации и при запутывании с δ*."),
        ("H2", "Зависимость от log f нелинейна; квадратичный член уменьшит ошибку.",
         "Подтвердилось: CV RMSE 5.161 → 3.172 дБ при переходе от степени 1 к 2. "
         "Степень 3 дала ещё 2.443 дБ, но ухудшила передачу на новые режимы."),
        ("H3", "Кривые лучше совпадают в координатах f c/U, чем в f.",
         "Частично: corr(log(f c/U), SPL) = −0.455 против −0.355 для log f. "
         "Однако простая замена log f и log U на log(f c/U) не улучшила CV."),
        ("H4", "Рост δ* снижает высокочастотный шум.",
         "Знак отрицательный и устойчивый в базовой модели: P(β < 0) = 0.98. "
         "Но δ* коррелирует с углом атаки (r = 0.835), поэтому вклад нельзя "
         "интерпретировать изолированно."),
    ]
    for code, text, verdict in hypotheses:
        story.append(p(f"<b>{code}.</b> {text} <i>Результат:</i> {verdict}", "RussianSmall"))

    story.append(PageBreak())
    story.append(p("4. Метод", "RussianH1"))
    story.append(p(
        "Базовые физические признаки: log10(f), log10(U), log10(δ*), α, log10(c). "
        "Дополнительно рассматривались log10(f c/U) и U². Полиномиальное "
        "расширение степеней 2 и 3 строилось только после выбора базового набора. "
        "Финальная модель имеет вид ŷ = φ(x)ᵀβ + b: линейна по β, нелинейна только "
        "за счёт φ(x). Перед оценкой коэффициентов признаки стандартизованы."
    ))
    story.append(p(
        "Разбиение: 1202 обучающих / 301 тестовых строк, стратификация по U. "
        "Внутри обучающей части — 5-fold GridSearchCV с отрицательным RMSE как "
        "метрикой выбора. Стандартизация, полиномы и выбор признаков входят в "
        "Pipeline и переобучаются в каждом фолде. Отдельно: 5-fold GroupKFold по "
        "конфигурации, бутстреп系数ов (400 повторов) и два полностью отложенных "
        "уровня скорости (55.5 и 71.3 м/с). Тест не участвует ни в одном выборе."
    ))

    story.append(p("5. Результат выбранной модели", "RussianH1"))
    story.append(p(f"Выбрано training CV: <b>{selected['model']}</b>, параметры {selected['best_params']}.", "RussianSmall"))
    story.append(table(final_metrics, [125, 125, 125, 125], digits=3))
    story.append(Spacer(1, 6))
    story.append(p(
        "Train, CV и test близки: зазора типичного переобучения при обычном "
        "случайном разбиении нет. CV-строка получена out-of-fold предсказаниями; "
        "среднее значение GridSearchCV по фолдам отличается лишь на 0.01 дБ."
    ))
    story.append(figure("measured_vs_predicted.png", 365))
    story.append(p("Рис. 1. Измеренный и предсказанный SPL на случайном тесте.", "RussianSmall"))

    story.append(p("6. Сравнение моделей", "RussianH1"))
    story.append(table(model_compact, [150, 115, 120, 110], digits=3))
    story.append(p(
        "Baseline предсказывает только среднее обучающей выборки. Сырые признаки "
        "с МНК дают CV RMSE 4.843 дБ; физические логарифмы по отдельности не "
        "помогают, но полиномы дают большой выигрыш. Ridge почти не отличается "
        "от МНК (3.170 против 3.172 дБ), Lasso также близок (3.178 дБ), а "
        "Elastic Net на этой сетке хуже (3.662 дБ).",
        "RussianSmall",
    ))
    story.append(figure("cv_model_comparison.png", 460))
    story.append(p("Рис. 2. CV RMSE всех кандидатов на обучающей части.", "RussianSmall"))

    story.append(PageBreak())
    story.append(p("7. Сложность и регуляризация", "RussianH1"))
    degree_view = degree.rename(columns={
        "degree": "степень", "cv_rmse": "CV RMSE, дБ",
        "train_rmse": "train RMSE, дБ", "test_rmse": "test RMSE, дБ",
    })
    story.append(table(degree_view, [90, 140, 140, 140], digits=3))
    story.append(p(
        "Степень 1 недообучает: ошибка ~5 дБ. Степень 2 снижает её примерно на "
        "2 дБ, степень 3 — ещё на 0.73 дБ по CV. При 1202 объектах и 55 "
        "полиномиальных членах данных достаточно, чтобы CV выбрал степень 3. "
        "Однако GroupKFold ниже показывает, что часть этого выигрыша связана с "
        "повторными частотами внутри одной конфигурации."
    ))
    story.append(figure("cv_regularization.png", 420))
    story.append(p(
        "Рис. 3. Ошибка CV при росте α. Для Ridge и Lasso минимум лежит у "
        "наименьших α; Regularization почти не помогает, потому что признаков "
        "немного относительно числа наблюдений."
    ))

    story.append(p("8. Переобучение, недообучение и утечка конфигурации", "RussianH1"))
    grouped_view = grouped.rename(columns={
        "model": "модель", "n_groups": "конфигураций",
        "mae": "MAE, дБ", "rmse": "RMSE, дБ", "r2": "R²",
    })
    story.append(table(grouped_view, [150, 100, 90, 90, 80], digits=3))
    story.append(p(
        "Обычная CV выбрала степень 3, но если фолд содержит полностью новую "
        "конфигурацию, её RMSE равен 4.213 дБ против 2.453 дБ при обычной CV. "
        "У степени 2 ухудшение мягче: 3.426 против 3.172 дБ. Это признак того, "
        "что степень 3 частично запоминает индивидуальные спектры конфигураций. "
        "Поэтому 'честная' оценка переноса на новый эксперимент ближе к 3.4–4.2 дБ, "
        "чем к 2.4 дБ."
    ))

    story.append(p("9. Отложенные скорости: интерполяция и передача", "RussianH1"))
    story.append(table(hold_compact, [90, 145, 45, 75, 80, 65], digits=3))
    story.append(p(
        "U = 55.5 м/с лежит между двумя скоростями из обучения: модель переносится "
        "хорошо (RMSE 2.188 дБ, почти нулевое смещение −0.096 дБ). "
        "U = 71.3 м/с — крайняя скорость: у степени 3 RMSE 4.929 дБ и R² 0.505, "
        "хотя среднее смещение всего +0.284 дБ. Провальный режим — угол "
        "12.3–22.2°, где средний остаток +2.38 дБ; на α = 22.2°, c = 0.0254 м "
        "модель недооценивает SPL на 19–32 дБ. Степень 2 на этой же отложенной "
        "скорости даёт RMSE 3.426 дБ и R² 0.761, то есть переносится заметно лучше."
    ))

    story.append(PageBreak())
    story.append(p("10. Остатки", "RussianH1"))
    story.append(figure("residual_diagnostics.png", 470))
    story.append(p(
        "Рис. 4. Остатки теста против предсказания, скорости, log(f c/U) и угла. "
        "В случайном тесте нет большого среднего смещения по скорости, но "
        "наблюдаются структурированные области: остаток +1.20 дБ при предсказанном SPL "
        "130–136 дБ и −0.86 дБ при α > 12.3°. На отложенной скорости 71.3 м/с "
        "эти эффекты усиливаются и превращаются в крупные ошибки."
    ))

    story.append(p("11. Устойчивость коэффициентов", "RussianH1"))
    story.append(p(
        "Бутстреп проведён для линейной модели на базовом физическом наборе: "
        "коэффициенты выражены в дБ на одно стандартное отклонение признака. "
        "Для полинома степени 3 отдельные коэффициенты не имеют простого "
        "физического смысла, поэтому устойчивость проверена на интерпретируемой "
        "базовой модели."
    ))
    story.append(table(boot_view, [95, 85, 80, 80, 75, 80], digits=3))
    story.append(Spacer(1, 5))
    story.append(p(
        "Все знаки, кроме log(c), устойчивы. Чувствительность к коррелированным "
        "признакам видна при замене log f и log U на log(f c/U): коэффициент "
        "log(c) меняет знак с −2.97 на +0.15 и становится неопределённым "
        "(95% интервал −0.55 … +0.79, P(>0) = 0.63). Это важная оговорка: "
        "интерпретацию отдельных коэффициентов нельзя переносить на другую "
        "параметризацию."
    ))
    story.append(table(corr_pairs, [260, 90], digits=3))
    story.append(p(
        "Таблица 5. Корреляции между проблемными признаками в обучающей части.",
        "RussianSmall",
    ))

    story.append(p("12. Вывод", "RussianH1"))
    story.append(p(
        "Устойчивая физическая зависимость: SPL падает с частотой и безразмерной "
        "частотой f c/U, растёт с U и снижается при больших δ* и углах атаки. "
        "Линейная по коэффициентам модель с полиномиальными признаками хорошо "
        "работает внутри измеренного диапазона и при интерполяции по скорости. "
        "Она систематически отказывает на крайнем сочетании высокой скорости, "
        "большого угла атаки и малой хорды, где физический механизм шума, "
        "по-видимому, меняется. Для чистого предсказания передовых режимов "
        "следует предпочесть степень 2 и собрать данные именно в этой области."
    ))
    story.append(Spacer(1, 6))
    story.append(p(
        "Воспроизведение: из папки assignment-1 выполнить py -m src.main и "
        "py -m src.report. Таблицы и рисунки лежат в assignment-1/results.",
        "RussianSmall",
    ))

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="Airfoil Self-Noise — Assignment 1",
        author="ST117205",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return PDF_PATH


if __name__ == "__main__":
    print(build_report())
