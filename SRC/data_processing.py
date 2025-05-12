import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def load_data(file_path):
    """Загружает данные из CSV файла."""
    return pd.read_csv(file_path)

def handle_missing_values(df, num_strategy='median', cat_strategy='most_frequent'):
    """Обрабатывает пропущенные значения в DataFrame."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    num_imputer = SimpleImputer(strategy=num_strategy)
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    cat_imputer = SimpleImputer(strategy=cat_strategy)
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    return df

def remove_duplicates(df):
    """Удаляет дубликаты из DataFrame."""
    return df.drop_duplicates()

def select_features(df, target='SalePrice', corr_threshold=0.5, cat_cols=None):
    """Выбирает числовые признаки с корреляцией > corr_threshold и указанные категориальные признаки."""
    if cat_cols is None:
        cat_cols = []
    
    # Выбираем только числовые столбцы для вычисления корреляции
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_df = df[num_cols]
    
    num_corr = num_df.corr()[target]
    important_num_cols = num_corr[(num_corr > corr_threshold) | (num_corr < -corr_threshold)].index.tolist()
    if target in important_num_cols:
        important_num_cols.remove(target)
    
    # Объединяем выбранные числовые столбцы с категориальными и целевой переменной
    important_cols = important_num_cols + cat_cols
    return df[important_cols + [target]], important_num_cols

def encode_categorical(df, cat_cols):
    """Кодирует категориальные признаки с помощью one-hot encoding."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[cat_cols]))
    encoded_cols.columns = encoder.get_feature_names_out(cat_cols)
    df = df.drop(cat_cols, axis=1)
    df = pd.concat([df, encoded_cols], axis=1)
    return df

def scale_numerical(df, num_cols):
    """Стандартизирует числовые признаки."""
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def split_data(X, y, test_size=0.2, random_state=42):
    """Разделяет данные на обучающую и тестовую выборки."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def process_data(train_path, test_path, target='SalePrice', cat_cols=None):
    """Полный пайплайн обработки данных."""
    if cat_cols is None:
        cat_cols = []
    train_df = load_data(train_path)
    test_df = load_data(test_path)
    
    # Обработка пропусков
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)
    
    # Удаление дубликатов
    train_df = remove_duplicates(train_df)
    test_df = remove_duplicates(test_df)
    
    # Выбор признаков
    train_df, important_num_cols = select_features(train_df, target, cat_cols=cat_cols)
    test_df = test_df[important_num_cols + cat_cols]
    
    # Кодирование категориальных признаков
    train_df = encode_categorical(train_df, cat_cols)
    test_df = encode_categorical(test_df, cat_cols)
    
    # Стандартизация числовых признаков
    train_df = scale_numerical(train_df, important_num_cols)
    test_df = scale_numerical(test_df, important_num_cols)
    
    # Разделение на X и y
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df
    
    return X_train, y_train, X_test