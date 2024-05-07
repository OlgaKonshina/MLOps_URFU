import pandas as pd
from sklearn.preprocessing import StandardScaler  # для стандартизации данных

df = pd.read_csv('/home/olga/lab4/pythonProject/datasets/heart_statlog_cleveland_hungary_final.csv')
data = df.drop('target', axis=1)  # удаляем целевую переменную
columns = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
       'fasting blood sugar', 'resting ecg', 'max heart rate',
       'exercise angina', 'oldpeak', 'ST slope']
    # проводим стандартизацию данных
st_data = StandardScaler()
st_data.fit(data)
sdata = st_data.transform(data[columns])
data_st = pd.DataFrame(sdata, columns=columns)

df_prep = pd.concat([data_st, df['target']], axis=1)  # объединяем стандартизованные данные и целевую переменную
df_prep.to_csv('/home/olga/lab4/pythonProject/datasets/heart_statlog_cleveland_hungary_final.csv')