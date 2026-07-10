import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ydata_profiling import ProfileReport


pd.set_option('display.max_columns', 150)
pd.set_option('display.max_row', 150)

data = pd.read_csv('../data/application_train.csv')

data.info(verbose=True, show_counts=True)
data.describe().T
data.value_counts()

print("\n=== PUSTOTI ===")
missing = data.isnull().mean() * 100
missing = missing[missing > 0].sort_values(ascending=False)
print(missing.head(50))

# Смотрим на максимальный стаж работы в днях
print("Максимальный стаж в датасете (в днях):", data['DAYS_EMPLOYED'].max())

# Переводим в годы для наглядности
print("Это примерно сколько лет?:", data['DAYS_EMPLOYED'].max() / 365)


numeric_cols = data.select_dtypes(include=[np.number]).columns
cols_to_check = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'TARGET']]

outliers_summary = []

for col in cols_to_check:
    # Удаляем пропуски для честного расчета квантилей
    col_clean = data[col].dropna()
    if col_clean.empty:
        continue
        
    Q1 = col_clean.quantile(0.25)
    Q3 = col_clean.quantile(0.75)
    IQR = Q3 - Q1
    
    # Жесткий фильтр: берем 3 * IQR (все, что за этими границами — жесткая аномалия)
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Считаем количество выбросов
    outliers_count = ((col_clean < lower_bound) | (col_clean > upper_bound)).sum()
    outliers_pct = (outliers_count / len(data)) * 100
    
    if outliers_count > 0:
        outliers_summary.append({
            'Колонка': col,
            'Нижняя граница': lower_bound,
            'Верхняя граница': upper_bound,
            'Кол-во выбросов': outliers_count,
            'Процент выбросов (%)': round(outliers_pct, 2)
        })

# Собираем в красивую таблицу и сортируем по проценту аномалий
outliers_df = pd.DataFrame(outliers_summary).sort_values(by='Процент выбросов (%)', ascending=False)
print("=== ТОП-10 КОЛОНОК С НАИБОЛЬШИМ КОЛИЧЕСТВОМ АНОМАЛИЙ ===")
print(outliers_df.head(10).to_string(index=False))

print("Начинаю генерацию отчета... Это займет меньше минуты.")

# Берём случайные 20 000 строк для ускорения
data_sample = data.sample(n=min(20000, len(data)), random_state=42)

# Генерируем отчет по выборке с минимальными настройками
profile = ProfileReport(
    data_sample, 
    title="Home Credit Default Risk - Быстрый Аудит", 
    minimal=True,
    explorative=False
)

# Сохраняем в файл
profile.to_file("../notebooks/home_credit_report.html")

print("Йоу, бро! Отчет готов. Ищи файл 'home_credit_report.html' в папке notebooks.")


