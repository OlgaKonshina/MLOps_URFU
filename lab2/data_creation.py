import pandas as pd
from sklearn.model_selection import train_test_split
import opendatasets as od

od.download(
    "https://www.kaggle.com/datasets/mexwell/heart-disease-dataset/data?select=heart_statlog_cleveland_hungary_final.csv")
df = pd.read_csv('heart-disease-dataset/heart_statlog_cleveland_hungary_final.csv')
# разбиваем на тренировочную и валидационную
x_train, x_val, = train_test_split(df, test_size=0.2,random_state=42)
# сохранение данных в csv
x_train.to_csv('train.csv', index=False)
x_val.to_csv('test.csv', index=False)

print('Tренировочные данные: сохранены')
print('Тестовые данные: сохранены')
