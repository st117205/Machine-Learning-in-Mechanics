# Machine Learning in Mechanics

Репозиторий для учебных заданий по теме «Машинное обучение в механике».

## Assignment 0

**Assignment 0** — аудит учебного проекта **Young's Modulus Calculator** и создание проекта-аналога **Poisson Ratio Lab** на Python.

Все материалы задания находятся в каталоге [`assignment-0/`](assignment-0/).

| Путь | Содержание |
|---|---|
| [`assignment-0/Итог_работы.md`](assignment-0/Итог_работы.md) | Краткая сводка работы |
| [`assignment-0/docs/REPORT.md`](assignment-0/docs/REPORT.md) | Полный отчёт: исследование, аудит, новый проект |
| [`assignment-0/docs/CHECKS.md`](assignment-0/docs/CHECKS.md) | Команды и фактические результаты проверок |
| [`assignment-0/young-modulus/`](assignment-0/young-modulus/) | Копия исходного проекта `banitsriram/young-modulus` |
| [`assignment-0/poisson-lab/`](assignment-0/poisson-lab/) | Созданный проект для расчёта коэффициента Пуассона |

### Кратко о задании

1. Исследован исходный проект `banitsriram/young-modulus`: структура, зависимости, точка входа, способ запуска; текущее поведение воспроизведено, файлы не менялись.
2. Проведён независимый научный аудит: проверены физические предположения, размерности, единицы, формулы, модели и обработка данных; найдено 15 замечаний с указанием файла, риска и способа проверки.
3. Создан проект `poisson-lab` — учебный аналог на Python для расчёта коэффициента Пуассона для тех же 13 материалов. Есть 15 unit-тестов, режимы `--list` и `--demo`, краткая документация.

### Запуск созданного проекта

```bash
cd assignment-0/poisson-lab
py -m unittest discover -s tests -v   # 15 тестов, все OK
py main.py --list
py main.py --demo                     # сталь, nu = 0.300, G = 76.9 GPa
py main.py                            # интерактивный ввод
```

Если `python` не является заглушкой Microsoft Store, можно использовать `python` вместо `py`. Внешние пакеты не нужны, достаточно Python 3.8+.

## Assignment 1

**Assignment 1** — линейная по коэффициентам модель собственного шума профиля NASA/UCI Airfoil Self-Noise: физические преобразования, полиномы, МНК/OLS, Ridge, Lasso, Elastic Net, CV и проверка на отложенных скоростях.

Все материалы задания находятся в каталоге [`assignment-1/`](assignment-1/).

| Путь | Содержание |
|---|---|
| [`assignment-1/README.md`](assignment-1/README.md) | Что сделано, как запустить, результаты и ограничения |
| [`assignment-1/report.pdf`](assignment-1/results/report.pdf) | Полный отчёт с таблицами и рисунками |
| [`assignment-1/src/`](assignment-1/src/) | Конвейеры, метрики, графики и сборка отчёта |
| [`assignment-1/results/`](assignment-1/results/) | CSV-таблицы, PNG-рисунки и PDF |

### Кратко о задании

1. Сравнены baseline, OLS, полиномы 2/3, Ridge, Lasso и Elastic Net.
2. Итоговая модель выбрана по худшему RMSE на отложенных скоростях: `ridge_poly2`, test RMSE 3.227 дБ.
3. Полином степени 3 лучше на случайной CV, но хуже на незнакомом режиме; простая модель с Ridge переносится лучше.

### Запуск

```powershell
cd assignment-1
py -m pip install -r requirements.txt
py -m unittest discover -s tests -v
py -m src.main
py -m src.report
```

