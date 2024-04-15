import pickle # для сохранения модели
from model_preprocessing import data_preparation  # для обработки данных

model_path = '/home/olga/MLOps/model.pkl'
test_path = '/home/olga/MLOps/test/test_data.csv'
loaded_model = pickle.load(open(model_path, 'rb'))  # загружаем модель

df = data_preparation(test_path)  # загружаем тестовые данные
X, y = df.drop(columns=['target']), df['target']
test_predict = loaded_model.predict(X)

print('Предсказания модели :', test_predict)