import seaborn as sns
import matplotlib.pyplot as plt
import os

def ensure_directory_exists(path):
    """Создает директорию, если она не существует."""
    if not os.path.exists(path):
        os.makedirs(path)

def plot_correlation_matrix(df, save_path='/content/drive/MyDrive/Final-Folder/figures'):
    """Строит корреляционную матрицу."""
    ensure_directory_exists(save_path)
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), cmap="RdBu")
    plt.title("Correlation Matrix", size=15)
    plt.savefig(os.path.join(save_path, 'correlation_matrix.png'))
    plt.show()
    plt.close()

def plot_pairplot(df, cols, save_path='/content/drive/MyDrive/Final-Folder/figures'):
    """Строит парные графики для указанных колонок."""
    ensure_directory_exists(save_path)
    sns.pairplot(df[cols])
    plt.savefig(os.path.join(save_path, 'pairplot.png'))
    plt.show()
    plt.close()

def plot_jointplot(df, x_col, y_col, save_path='/content/drive/MyDrive/Final-Folder/figures'):
    """Строит совместное распределение для двух колонок."""
    ensure_directory_exists(save_path)
    sns.jointplot(x=df[x_col], y=df[y_col], kind="kde")
    plt.savefig(os.path.join(save_path, f'jointplot_{x_col}_vs_{y_col}.png'))
    plt.show()
    plt.close()

def plot_boxplot(df, cat_col, target, save_path='/content/drive/MyDrive/Final-Folder/figures'):
    """Строит boxplot для категориального признака и целевой переменной."""
    ensure_directory_exists(save_path)
    plt.figure(figsize=(10,6))
    sns.boxplot(x=cat_col, y=target, data=df)
    plt.title(f'{target} by {cat_col}')
    plt.savefig(os.path.join(save_path, f'boxplot_{cat_col}.png'))
    plt.show()
    plt.close()

def plot_histogram(df, col, save_path='/content/drive/MyDrive/Final-Folder/figures'):
    """Строит гистограмму для числового признака."""
    ensure_directory_exists(save_path)
    plt.figure(figsize=(10,6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(os.path.join(save_path, f'histogram_{col}.png'))
    plt.show()
    plt.close()

def visualize_data(df, target='SalePrice', cat_cols=None, num_cols=None, save_path='/content/drive/MyDrive/Final-Folder/figures'):
    """Полный пайплайн визуализации данных."""
    if cat_cols is None:
        cat_cols = []
    if num_cols is None:
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    print("Построение корреляционной матрицы...")
    plot_correlation_matrix(df[num_cols], save_path)
    
    print("Построение парных графиков...")
    plot_pairplot(df, num_cols, save_path)
    
    print("Построение совместных распределений...")
    for col in num_cols:
        if col != target:
            print(f"Совместное распределение для {col} и {target}")
            plot_jointplot(df, col, target, save_path)
    
    print("Построение boxplot для категориальных признаков...")
    for col in cat_cols:
        print(f"Boxplot для {col}")
        plot_boxplot(df, col, target, save_path)
    
    print("Построение гистограмм для числовых признаков...")
    for col in num_cols:
        print(f"Гистограмма для {col}")
        plot_histogram(df, col, save_path)
