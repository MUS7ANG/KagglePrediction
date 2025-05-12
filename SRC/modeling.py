from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def evaluate_model(y_true, y_pred):
    """Вычисляет метрики MAE, MSE, RMSE, R2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

def cross_val_rmse(model, X, y, cv=5):
    """Вычисляет RMSE с помощью кросс-валидации."""
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    rmse = np.sqrt(-scores.mean())
    return rmse

def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """Обучает и оценивает несколько моделей."""
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae, mse, rmse, r2 = evaluate_model(y_test, y_pred)
        cv_rmse = cross_val_rmse(model, X_train, y_train)
        results.append({
            'Model': name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'CV RMSE': cv_rmse
        })
    return pd.DataFrame(results)

def get_models():
    """Возвращает словарь с моделями."""
    return {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Random Forest': RandomForestRegressor(),
        'XGBoost': XGBRegressor()
    }