import pandas as pd
import numpy as np # для генерации данных
from pathlib import Path # для сохранения данных

# Востроизводимость
np.random.seed(73)

# генерация синтетических данных
loc, scale = 20, 3
n_samples = 5000
features = {
    'feature1': np.random.poisson(10, n_samples),
    'feature2': np.random.logistic(loc, scale, n_samples),
    'feature3': np.random.rand(n_samples),
    'feature4': np.random.rand(n_samples),
    'feature5': np.random.rand(n_samples),
    'feature6': np.random.rand(n_samples)
}

# Целевая переменная
features['target'] = (
        2 * features['feature1']
        + 0.3 * features['feature2']
        + 2.5 * features['feature4']
        + 0.75 * features['feature6']
        + np.random.normal(loc, scale, n_samples)
)

# Coздаем DataFrame
df_train = pd.DataFrame(features).sample(frac=0.8, random_state=73)
df_test = pd.DataFrame(features).drop(df_train.index)

# Путь для сохранения файлов train
path_train = Path('/home/olga/MLOps/train/train_data.csv')
path_train.parent.mkdir(parents=True, exist_ok=True)

# Путь для сохранения данных test
path_test = Path('/home/olga/MLOps/test/test_data.csv')
path_test.parent.mkdir(parents=True, exist_ok=True)

# сохранение данных в csv
df_train.to_csv(path_train, index=False)
df_test.to_csv(path_test, index=False)


print('Tренировочные данные: ', path_train)
print('Тестовые данные: ', path_test)
