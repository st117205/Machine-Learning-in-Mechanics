# Assignment 0 — Проверки и фактические результаты

Все команды выполнялись в Windows, Python 3.9.13 через `py`.

## 1. Исходный проект `young-modulus`

### 1.1 Запуск с системной кодировкой CP1251

```powershell
py main.py
```

Ввод: сталь, `L=50`, `b=2`, `h=0.3`, точечная нагрузка, 2 замера, вес 50 г, прогибы 0.15 и 0.30 см.

Результат: падение до вывода итогов.

```text
UnicodeEncodeError: 'charmap' codec can't encode character '\xb3'
in position 20: character maps to <undefined>
```

Причина: не-ASCII символы в выводе (`cm³`, `✓`, `⚠`).

### 1.2 Запуск с UTF-8

```powershell
$env:PYTHONIOENCODING='utf-8'
py main.py
```

Тот же ввод дал:

```text
Calculated Young's Modulus: 18.92 GPa
Expected Young's Modulus:   200 GPa
Percentage Difference:       90.54%
```

Соответствие README (~200 ГПа) нет.

### 1.3 Нулевой прогиб

Ввод: сосна, равномерная нагрузка, два замера с прогибом 0.

```text
Calculated Young's Modulus: 0.00 GPa
Percentage Difference:       100.00%
```

Предупреждения о пропущенных измерениях нет.

### 1.4 Проверка `NaN`

```powershell
py -c "import sys; sys.path.insert(0, '.'); import main as m; print(m.calculate_moment_of_inertia(float('nan'), 1.0)); print(m.calculate_youngs_modulus_nonuniform(50, 2, 0.3, [{'weight':50,'depression':float('nan')}]))"
```

Вывод: `nan` и `nan`. Валидация ввода не отсекает нечисловые значения.

### 1.5 Арифметическая проверка примеров README

| Пример | По коду | В README | Требуемый прогиб для Заявленного E |
|---|---:|---:|---:|
| Сталь, 50 г, δ=0.15 см | 18.92 ГПа | ~200 ГПа | δ≈0.0142 см |
| Сосна, 100 г, δ=0.80 см | 0.327 ГПа | ~9 ГПа | δ≈0.0291 см |

## 2. Созданный проект `poisson-lab`

### 2.1 Unit-тесты

```powershell
cd poisson-lab
py -m unittest discover -s tests -v
```

```text
Ran 15 tests in 0.001s
OK
```

Покрыто: деформации, знаковое соглашение, один и несколько замеров, отказ нулевых/`NaN`/пустых данных, процентное расхождение, справочный диапазон, устойчивость `-1 < nu < 0.5`, оценка модуля сдвига, наличие всех 13 материалов.

### 2.2 Демонстрация

```powershell
py main.py --demo
```

Ключевые строки вывода:

```text
Measured nu (ratio of mean strains): 0.300
Reference nu:                        0.30
Percentage difference:              0.0%
OK: inside the material reference range
OK: inside isotropic stability range (-1 < nu < 0.5)
Estimated shear modulus G from E=2G(1+nu): 76.9 GPa
```

### 2.3 Таблица материалов

```powershell
py main.py --list
```

Выведены все 13 материалов. Для изотропных материалов рассчитан `G = E/(2(1+nu))`, например:

```text
steel      E=200.0  nu=0.30  G est=76.9 GPa
aluminum   E=69.0   nu=0.33  G est=25.9 GPa
```

Для дуба, сосны, тика, бамбука и фанеры указано `n/a (aniso)`.

### 2.4 Интерактивный сценарий

Ввод: сталь, `L0=50 мм`, `w0=10 мм`, 2 замера `(0.10, 0.006)` и `(0.20, 0.012)`.

```text
Measured nu (ratio of mean strains): 0.300
Reference nu:                        0.30
Percentage difference:              0.0%
OK: inside the material reference range
OK: inside isotropic stability range (-1 < nu < 0.5)
Estimated shear modulus G from E=2G(1+nu): 76.9 GPa
```

Сценарий прошёл при обычной кодировке консоли, без установки `PYTHONIOENCODING`.
