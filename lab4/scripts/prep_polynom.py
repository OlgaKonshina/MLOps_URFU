import pandas as pd
from sklearn.preprocessing import PowerTransformer

df = pd.read_csv('/home/olga/op/MLOps_URFU/lab4/datasets/heart_statlog_cleveland_hungary_final.csv')
columns = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
       'fasting blood sugar', 'resting ecg', 'max heart rate',
       'exercise angina', 'oldpeak', 'ST slope']

pt = PowerTransformer()
data = df.drop('target', axis=1)
pt.fit(data)
pt.lambdas_
power = pt.transform(data)
df_power = pd.DataFrame(power, columns=columns)
df_prep = pd.concat([df_power, df['target']], axis=1)
df_prep.to_csv('/home/olga/op/MLOps_URFU/lab4/datasets/heart_statlog_cleveland_hungary_final.csv')
print(df_prep)
