# Проект по предсказанию цен на жилье

## Обзор
Этот проект посвящен предсказанию цен на жилье с использованием машинного обучения. Он включает предварительную обработку данных, исследовательский анализ данных (EDA), выбор признаков, настройку гиперпараметров и оценку модели. Предполагается, что используется датасет [House Prices с Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data), где целевая переменная — `SalePrice` (цена дома).

Проект организован модульно и воспроизводимо, с разделением этапов на отдельные скрипты. В процессе анализа данных генерируется около 30 визуализаций, которые помогают понять данные и взаимосвязи между признаками.

## Структура проекта
```
House-Price-Prediction/
│
├── DATA/
│   ├── train.csv               # Обучающий набор данных
│   └── test.csv                # Тестовый набор данных
│
├── SRC/
│   ├── data_processing.py      # Загрузка, очистка и предварительная обработка данных
│   ├── eda.py                  # Исследовательский анализ данных и визуализации
│   ├── modeling.py             # Обучение и оценка модели
│   └── main.py                 # Главный скрипт для запуска пайплайна
│
├── Notebooks/
│   ├── main.ipynb              # Главный Jupyter Notebook для запуска пайплайна
│   ├── feature_selection.ipynb # Логика выбора признаков (с помощью RFE)
│   └── hyperparameter_tuning.ipynb # Настройка гиперпараметров для Random Forest
│
├── figures/
│   ├── target_histogram.png    # Гистограмма для SalePrice
│   ├── boxplot_MSZoning.png    # Boxplot для MSZoning vs SalePrice
│   ├── ...                     # Другие boxplot'ы для категориальных признаков
│   ├── scatter_OverallQual.png # Scatterplot для OverallQual vs SalePrice
│   ├── ...                     # Другие scatterplot'ы для числовых признаков
│   └── correlation_heatmap.png # Тепловая карта корреляций числовых признаков
│
├── results_tuned.csv           # Результаты оценки модели
│
└── README.md                   # Документация проекта
```

## Возможности
- **Предобработка данных**: Обработка пропущенных значений, удаление дубликатов, кодирование категориальных признаков и масштабирование числовых.
- **EDA**: Генерация ~29 визуализаций, включая:
  - Гистограмму целевой переменной (`SalePrice`).
  - Boxplot'ы для 7 категориальных признаков (например, `KitchenQual`, `MSZoning`).
  - Scatterplot'ы для ~20 числовых признаков (например, `OverallQual`, `GrLivArea`).
  - Тепловую карту корреляций числовых признаков.
- **Выбор признаков**: Использует Recursive Feature Elimination (RFE) с RandomForestRegressor для выбора 10 лучших признаков.
- **Настройка гиперпараметров**: Настройка RandomForestRegressor с помощью GridSearchCV.
- **Оценка модели**: Оценка настроенной модели с использованием метрик RMSE (кросс-валидация), MAE и R².

## Требования
- Python 3.10+
- Google Colab (для работы с Jupyter Notebook)
- Google Drive (для хранения данных и результатов)

### Необходимые библиотеки
Установите необходимые библиотеки с помощью команды:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib importnb
```

## Инструкции по установке
1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/your-username/House-Price-Prediction.git
   cd House-Price-Prediction
   ```

2. **Загрузите данные**:
   - Поместите файлы `train.csv` и `test.csv` в папку `DATA/`. Их можно скачать с [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

3. **Настройте Google Drive**:
   - Подключите Google Drive в Google Colab:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Скопируйте папку проекта в `/content/drive/MyDrive/Final-Folder/`.

4. **Установите зависимости**:
   - В ячейке Colab выполните:
     ```bash
     !pip install pandas numpy scikit-learn seaborn matplotlib importnb
     ```

## Запуск проекта
### Вариант 1: Использование `main.ipynb`
1. Откройте `/content/drive/MyDrive/Final-Folder/Notebooks/main.ipynb` в Google Colab.
2. Убедитесь, что `feature_selection.ipynb` и `hyperparameter_tuning.ipynb` находятся в той же папке (`/content/drive/MyDrive/Final-Folder/Notebooks/`).
3. Выполните все ячейки в `main.ipynb` (Shift + Enter).

### Вариант 2: Использование `main.py`
1. Откройте терминал или ячейку в Colab.
2. Перейдите в директорию проекта:
   ```bash
   cd /content/drive/MyDrive/Final-Folder/SRC
   ```
3. Запустите скрипт:
   ```bash
   python main.py
   ```

**Примечание**: Для запуска `main.py` требуется установленная библиотека `importnb`, так как скрипт импортирует функции из `.ipynb` файлов.

## Ожидаемые результаты
### Визуализации
В папке `/content/drive/MyDrive/Final-Folder/figures/` будет сгенерировано около 29 визуализаций:
- **Гистограмма**: `target_histogram.png` — Распределение `SalePrice` (скорее всего, скошенное вправо).
- **Boxplot'ы** (7 штук): Для каждого категориального признака (например, `boxplot_KitchenQual.png`) — Показывает, как `SalePrice` зависит от категорий.
  - Пример: Дома с `KitchenQual="Ex"` (отличное качество) имеют более высокую медианную цену, чем с `KitchenQual="TA"` (среднее качество).
- **Scatterplot'ы** (~20 штук): Для каждого числового признака (например, `scatter_GrLivArea.png`) — Показывает связь с `SalePrice`.
  - Пример: Положительная корреляция между `GrLivArea` (жилая площадь) и `SalePrice`.
- **Тепловая карта корреляций**: `correlation_heatmap.png` — Матрица корреляций числовых признаков.
  - Пример: Высокая корреляция между `SalePrice` и `OverallQual` (~0.8).

### Выбор признаков
- RFE выбирает 10 наиболее важных признаков, например:
  ```
  Отобранные признаки: ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt', 'MSZoning_RL', 'KitchenQual_Ex']
  ```

### Настройка гиперпараметров
- GridSearchCV настраивает RandomForestRegressor и выводит лучшие параметры:
  ```
  Лучшие параметры: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
  Лучший RMSE: 31500.123
  ```

### Оценка модели
- Результаты сохраняются в `/content/drive/MyDrive/Final-Folder/results_tuned.csv`:
  ```
  Модель              MAE         R²       CV RMSE
  Tuned Random Forest 16500.45   0.8950   31000.32
  ```

## Примечания
- **Предположения о данных**: Проект предполагает, что `train.csv` и `test.csv` имеют формат датасета Kaggle House Prices.
- **Настройка**:
  - Чтобы изменить количество выбираемых признаков, измените параметр `n_features_to_select` в `main.ipynb` или `main.py`.
  - Чтобы настроить гиперпараметры для других моделей, обновите `hyperparameter_tuning.ipynb`.
- **Визуализации**: Если вы хотите добавить другие графики (например, pairplot или violin plot), измените `eda.py`.

## Как внести вклад
Вы можете форкнуть этот репозиторий, внести улучшения и отправить pull request. Для значительных изменений, пожалуйста, сначала откройте issue, чтобы обсудить изменения.