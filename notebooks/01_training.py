import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

# ==========================================
# 1. ЗАГРУЗКА ДАННЫХ И ВЫДЕЛЕНИЕ ЦЕЛИ
# ==========================================
# Загружаем твой датасет (укажи правильный путь к файлу)
df = pd.read_csv('../data/application_train.csv')

X = df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
y = df['TARGET']

# Делим на тренировочную и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 2. КАСТОМНЫЙ FEATURE ENGINEERING
# ==========================================
def preprocess_home_credit(data):
    X_out = data.copy()
    
    # Исправляем баг 1000 лет стажа: создаем флаг аномалии и зануляем её в днях
    X_out['DAYS_EMPLOYED_ANOM'] = (X_out['DAYS_EMPLOYED'] == 365243).astype(int)
    X_out['DAYS_EMPLOYED'] = X_out['DAYS_EMPLOYED'].replace(365243, 0)
    
    # Переводим отрицательные дни в нормальные положительные годы
    X_out['AGE_YEARS'] = X_out['DAYS_BIRTH'] / -365.0
    X_out['EMPLOYMENT_YEARS'] = X_out['DAYS_EMPLOYED'] / -365.0
    
    # Логарифмируем скошенные доходы и кредиты для защиты от выбросов
    X_out['AMT_INCOME_TOTAL_LOG'] = np.log1p(X_out['AMT_INCOME_TOTAL'])
    X_out['AMT_CREDIT_LOG'] = np.log1p(X_out['AMT_CREDIT'])
    
    # Удаляем старые неоттрансформированные колонки, чтобы не дублировать информацию
    X_out = X_out.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_CREDIT'], errors='ignore')
    
    return X_out

# Оборачиваем нашу функцию в трансформер scikit-learn
feature_transformer = FunctionTransformer(preprocess_home_credit)

# ==========================================
# 3. АВТОМАТИЧЕСКОЕ РАЗДЕЛЕНИЕ ТИПОВ ДАННЫХ
# ==========================================
# Прогоняем одну строку через наш трансформер, чтобы узнать финальные имена колонок
from sklearn.preprocessing import OrdinalEncoder

# Конвейер для чисел: заполняем пропуски медианой
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Конвейер для текста: сначала заполняем пропуски, а потом КОДИРУЕМ в числа (0, 1, 2)
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Объединяем обработку колонок
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)


# ==========================================
# 4. СБОРКА И ОБУЧЕНИЕ ФИНАЛЬНОГО ПАЙПЛАЙНА
# ==========================================
# В LightGBM текстовые колонки будут идти в самом конце матрицы (после preprocessor)
# Передаем модели индексы категориальных признаков
cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

# Собираем сквозной конвейер
home_credit_pipeline = Pipeline(steps=[
    ('feature_eng', feature_transformer),
    ('data_preprocessor', preprocessor),
    ('model', LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        class_weight='balanced', # Мощная балансировка 8% дефолтов
        n_jobs=-1
    ))
])

print("🚀 Запускаю сквозной пайплайн обучения...")
home_credit_pipeline.fit(X_train, y_train)
print("✅ Модель успешно обучена!")

# ==========================================
# 5. ОЦЕНКА КАЧЕСТВА (ROC-AUC)
# ==========================================
# Считаем вероятности дефолта для валидационной выборки
y_pred_proba = home_credit_pipeline.predict_proba(X_val)[:, 1]

# Вычисляем ROC-AUC score
auc_score = roc_auc_score(y_val, y_pred_proba)
print(f"\n📊 Валидационный ROC-AUC Score: {auc_score:.4f}")

# ==========================================
# 6. СОХРАНЕНИЕ ПАЙПЛАЙНА В ФАЙЛ
# ==========================================
import os
os.makedirs('../model', exist_ok=True)
joblib.dump(home_credit_pipeline, '../model/home_credit_lgbm_pipeline.pkl')
print("💾 Пайплайн со всеми медианами сохранен в '../model/home_credit_lgbm_pipeline.pkl'")
