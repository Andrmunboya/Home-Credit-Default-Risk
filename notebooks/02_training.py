import pandas as pd
import numpy as np
import joblib

# 1. Загружаем сырой тестовый датасет с Kaggle
# Укажи правильный путь к твоему файлу application_test.csv
data_test = pd.read_csv('../data/application_test.csv')

# Обязательно объявляем функцию, чтобы joblib её увидел!
def preprocess_home_credit(data):
    X_out = data.copy()
    X_out['DAYS_EMPLOYED_ANOM'] = (X_out['DAYS_EMPLOYED'] == 365243).astype(int)
    X_out['DAYS_EMPLOYED'] = X_out['DAYS_EMPLOYED'].replace(365243, 0)
    X_out['AGE_YEARS'] = X_out['DAYS_BIRTH'] / -365.0
    X_out['EMPLOYMENT_YEARS'] = X_out['DAYS_EMPLOYED'] / -365.0
    X_out['AMT_INCOME_TOTAL_LOG'] = np.log1p(X_out['AMT_INCOME_TOTAL'])
    X_out['AMT_CREDIT_LOG'] = np.log1p(X_out['AMT_CREDIT'])
    X_out = X_out.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_CREDIT'], errors='ignore')
    return X_out

# 2. Загружаем наш сохраненный пайплайн-комбайн
pipeline = joblib.load('../model/home_credit_lgbm_pipeline.pkl')
print("✅ Пайплайн успешно загружен!")

# 3. Делаем предсказание ВЕРОЯТНОСТЕЙ (ВАЖНО: для ROC-AUC на Kaggle нужны именно вероятности!)
# Метод predict_proba выдает два столбца: [вероятность нуля, вероятность единицы]. Нам нужен второй.
print("🚀 Рассчитываю вероятности дефолта для тестовой выборки...")
test_predictions_proba = pipeline.predict_proba(data_test)[:, 1]

# 4. Формируем финальный submission-файл строго по правилам соревнования
submission = pd.DataFrame({
    'SK_ID_CURR': data_test['SK_ID_CURR'],
    'TARGET': test_predictions_proba
})

# 5. Выгружаем в CSV
submission.to_csv("../data/home_credit_submission.csv", index=False)
print("🎉 Идеальный файл 'home_credit_submission.csv' успешно создан и готов к отправке!")
