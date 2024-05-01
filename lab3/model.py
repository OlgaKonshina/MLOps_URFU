from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle  # для сохранения модели
import pandas as pd
from sklearn.preprocessing import StandardScaler


def data_preparation(date):
    df = pd.read_csv(date, sep=',')  # импорт данных
    data = df.drop('target', axis=1)  # удаляем целевую переменную
    columns = data.columns
    # проводим стандартизацию данных
    st_data = StandardScaler(copy=True, with_mean=True, with_std=True)
    st_data.fit(data)
    sdata = st_data.transform(data[columns])
    data_st = pd.DataFrame(sdata, columns=columns)

    df_prep = pd.concat([data_st, df['target']], axis=1)  # объединяем стандартизованные данные и целевую переменную
    return df_prep


model = RandomForestClassifier(max_features='log2', n_estimators=300, random_state=73)
path = 'train_data.csv'

df = data_preparation(path)
X, y = df.drop(columns=['target']), df['target']

# разбиваем на тестовую и валидационную
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.2,
                                                  random_state=73)
model.fit(X_train, y_train)
model.fit(X, y)
pickle.dump(model, open('model.pkl', "wb"))  # сохраняем модель

print('Модель сохранена')
prediction = model.predict(X_val)
print("accuracy:", metrics.accuracy_score(y_val, prediction))
