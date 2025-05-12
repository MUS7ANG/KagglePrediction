import sys
import os
sys.path.append(os.path.abspath('/content/drive/MyDrive/Final-Folder/SRC'))

import data_processing
import eda
import modeling
import pandas as pd

def run_pipeline(train_path, test_path, target='SalePrice', cat_cols=None):
    if cat_cols is None:
        cat_cols = ['MSZoning', 'Utilities', 'BldgType', 'Heating', 'KitchenQual', 'SaleCondition', 'LandSlope']
    
    # Загружаем данные
    train_df = data_processing.load_data(train_path)
    test_df = data_processing.load_data(test_path)
    
    # Обрабатываем пропуски
    train_df = data_processing.handle_missing_values(train_df)
    test_df = data_processing.handle_missing_values(test_df)
    
    # Удаляем дубликаты
    train_df = data_processing.remove_duplicates(train_df)
    test_df = data_processing.remove_duplicates(test_df)
    
    # Выбираем признаки
    train_df, important_num_cols = data_processing.select_features(train_df, target, cat_cols=cat_cols)
    test_df = test_df[important_num_cols + cat_cols]
    
    # Визуализация данных (до кодирования категориальных столбцов)
    eda.visualize_data(train_df, target, cat_cols, important_num_cols, save_path='/content/drive/MyDrive/Final-Folder/figures')
    
    # Кодируем категориальные признаки
    train_df = data_processing.encode_categorical(train_df, cat_cols)
    test_df = data_processing.encode_categorical(test_df, cat_cols)
    
    # Масштабируем числовые признаки
    train_df = data_processing.scale_numerical(train_df, important_num_cols)
    test_df = data_processing.scale_numerical(test_df, important_num_cols)
    
    # Разделяем на X и y
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df
    
    # Обучение и оценка моделей
    models = modeling.get_models()
    X_train_split, X_test_split, y_train_split, y_test_split = data_processing.split_data(X_train, y_train)
    results = modeling.train_and_evaluate(models, X_train_split, y_train_split, X_test_split, y_test_split)
    
    # Вывод результатов
    print(results.sort_values(by='CV RMSE'))
    
    # Сохранение результатов
    results.to_csv('/content/drive/MyDrive/Final-Folder/results.csv', index=False)
    print("Результаты сохранены в /content/drive/MyDrive/Final-Folder/results.csv")

if __name__ == "__main__":
    train_path = '/content/drive/MyDrive/Final-Folder/DATA/train.csv'
    test_path = '/content/drive/MyDrive/Final-Folder/DATA/test.csv'
    run_pipeline(train_path, test_path)