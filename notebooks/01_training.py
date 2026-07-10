{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting lightgbm\n",
      "  Downloading lightgbm-4.6.0-py3-none-win_amd64.whl (1.5 MB)\n",
      "     ---------------------------------------- 1.5/1.5 MB 2.0 MB/s eta 0:00:00\n",
      "Requirement already satisfied: numpy>=1.17.0 in c:\\users\\andre\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from lightgbm) (2.3.5)\n",
      "Requirement already satisfied: scipy in c:\\users\\andre\\appdata\\local\\programs\\python\\python311\\lib\\site-packages (from lightgbm) (1.16.3)\n",
      "Installing collected packages: lightgbm\n",
      "Successfully installed lightgbm-4.6.0\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "[notice] A new release of pip available: 22.3 -> 26.1.2\n",
      "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
     ]
    }
   ],
   "source": [
    "%pip install lightgbm\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🚀 Запускаю сквозной пайплайн обучения...\n",
      "[LightGBM] [Info] Number of positive: 19860, number of negative: 226148\n",
      "[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.042690 seconds.\n",
      "You can set `force_row_wise=true` to remove the overhead.\n",
      "And if memory is not enough, you can set `force_col_wise=true`.\n",
      "[LightGBM] [Info] Total Bins 11324\n",
      "[LightGBM] [Info] Number of data points in the train set: 246008, number of used features: 117\n",
      "[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=-0.000000\n",
      "[LightGBM] [Info] Start training from score -0.000000\n",
      "✅ Модель успешно обучена!\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\andre\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\sklearn\\utils\\validation.py:2827: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "📊 Валидационный ROC-AUC Score: 0.7614\n",
      "💾 Пайплайн со всеми медианами сохранен в '../model/home_credit_lgbm_pipeline.pkl'\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import FunctionTransformer\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.metrics import roc_auc_score\n",
    "from lightgbm import LGBMClassifier\n",
    "\n",
    "# ==========================================\n",
    "# 1. ЗАГРУЗКА ДАННЫХ И ВЫДЕЛЕНИЕ ЦЕЛИ\n",
    "# ==========================================\n",
    "# Загружаем твой датасет (укажи правильный путь к файлу)\n",
    "df = pd.read_csv('../data/application_train.csv')\n",
    "\n",
    "X = df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')\n",
    "y = df['TARGET']\n",
    "\n",
    "# Делим на тренировочную и валидационную выборки\n",
    "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n",
    "# ==========================================\n",
    "# 2. КАСТОМНЫЙ FEATURE ENGINEERING\n",
    "# ==========================================\n",
    "def preprocess_home_credit(data):\n",
    "    X_out = data.copy()\n",
    "    \n",
    "    # Исправляем баг 1000 лет стажа: создаем флаг аномалии и зануляем её в днях\n",
    "    X_out['DAYS_EMPLOYED_ANOM'] = (X_out['DAYS_EMPLOYED'] == 365243).astype(int)\n",
    "    X_out['DAYS_EMPLOYED'] = X_out['DAYS_EMPLOYED'].replace(365243, 0)\n",
    "    \n",
    "    # Переводим отрицательные дни в нормальные положительные годы\n",
    "    X_out['AGE_YEARS'] = X_out['DAYS_BIRTH'] / -365.0\n",
    "    X_out['EMPLOYMENT_YEARS'] = X_out['DAYS_EMPLOYED'] / -365.0\n",
    "    \n",
    "    # Логарифмируем скошенные доходы и кредиты для защиты от выбросов\n",
    "    X_out['AMT_INCOME_TOTAL_LOG'] = np.log1p(X_out['AMT_INCOME_TOTAL'])\n",
    "    X_out['AMT_CREDIT_LOG'] = np.log1p(X_out['AMT_CREDIT'])\n",
    "    \n",
    "    # Удаляем старые неоттрансформированные колонки, чтобы не дублировать информацию\n",
    "    X_out = X_out.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_CREDIT'], errors='ignore')\n",
    "    \n",
    "    return X_out\n",
    "\n",
    "# Оборачиваем нашу функцию в трансформер scikit-learn\n",
    "feature_transformer = FunctionTransformer(preprocess_home_credit)\n",
    "\n",
    "# ==========================================\n",
    "# 3. АВТОМАТИЧЕСКОЕ РАЗДЕЛЕНИЕ ТИПОВ ДАННЫХ\n",
    "# ==========================================\n",
    "# Прогоняем одну строку через наш трансформер, чтобы узнать финальные имена колонок\n",
    "from sklearn.preprocessing import OrdinalEncoder\n",
    "\n",
    "# Конвейер для чисел: заполняем пропуски медианой\n",
    "num_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='median'))\n",
    "])\n",
    "\n",
    "# Конвейер для текста: сначала заполняем пропуски, а потом КОДИРУЕМ в числа (0, 1, 2)\n",
    "cat_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\n",
    "    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))\n",
    "])\n",
    "\n",
    "# Объединяем обработку колонок\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', num_transformer, num_cols),\n",
    "        ('cat', cat_transformer, cat_cols)\n",
    "    ]\n",
    ")\n",
    "\n",
    "\n",
    "# ==========================================\n",
    "# 4. СБОРКА И ОБУЧЕНИЕ ФИНАЛЬНОГО ПАЙПЛАЙНА\n",
    "# ==========================================\n",
    "# В LightGBM текстовые колонки будут идти в самом конце матрицы (после preprocessor)\n",
    "# Передаем модели индексы категориальных признаков\n",
    "cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))\n",
    "\n",
    "# Собираем сквозной конвейер\n",
    "home_credit_pipeline = Pipeline(steps=[\n",
    "    ('feature_eng', feature_transformer),\n",
    "    ('data_preprocessor', preprocessor),\n",
    "    ('model', LGBMClassifier(\n",
    "        n_estimators=500,\n",
    "        learning_rate=0.03,\n",
    "        num_leaves=31,\n",
    "        random_state=42,\n",
    "        class_weight='balanced', # Мощная балансировка 8% дефолтов\n",
    "        n_jobs=-1\n",
    "    ))\n",
    "])\n",
    "\n",
    "print(\"🚀 Запускаю сквозной пайплайн обучения...\")\n",
    "home_credit_pipeline.fit(X_train, y_train)\n",
    "print(\"✅ Модель успешно обучена!\")\n",
    "\n",
    "# ==========================================\n",
    "# 5. ОЦЕНКА КАЧЕСТВА (ROC-AUC)\n",
    "# ==========================================\n",
    "# Считаем вероятности дефолта для валидационной выборки\n",
    "y_pred_proba = home_credit_pipeline.predict_proba(X_val)[:, 1]\n",
    "\n",
    "# Вычисляем ROC-AUC score\n",
    "auc_score = roc_auc_score(y_val, y_pred_proba)\n",
    "print(f\"\\n📊 Валидационный ROC-AUC Score: {auc_score:.4f}\")\n",
    "\n",
    "# ==========================================\n",
    "# 6. СОХРАНЕНИЕ ПАЙПЛАЙНА В ФАЙЛ\n",
    "# ==========================================\n",
    "import os\n",
    "os.makedirs('../model', exist_ok=True)\n",
    "joblib.dump(home_credit_pipeline, '../model/home_credit_lgbm_pipeline.pkl')\n",
    "print(\"💾 Пайплайн со всеми медианами сохранен в '../model/home_credit_lgbm_pipeline.pkl'\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
