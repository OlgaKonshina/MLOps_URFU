import pandas as pd
from sklearn.preprocessing import StandardScaler  # для стандартизации данных


def data_preparation(path):
    df = pd.read_csv(path, sep=',')  # импорт данных
    data = df.drop('target', axis=1)  # удаляем целевую переменную
    columns = data.columns
    # проводим стандартизацию данных
    st_data = StandardScaler()
    st_data.fit(data)
    sdata = st_data.transform(data[columns])
    data_st = pd.DataFrame(sdata, columns=columns)

    df_prep = pd.concat([data_st, df['target']], axis=1)  # объединяем стандартизованные данные и целевую переменную
    return df_prep
