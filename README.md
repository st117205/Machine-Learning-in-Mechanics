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
