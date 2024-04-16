import pickle # для сохранения модели
from model_preprocessing import data_preparation  # для обработки данных
from sklearn.metrics import mean_squared_error

model_path = '/home/olga/MLOps/model.pkl'
test_path = '/home/olga/MLOps/test/test_data.csv'
loaded_model = pickle.load(open(model_path, 'rb'))  # загружаем модель

df = data_preparation(test_path)  # загружаем тестовые данные
X, y = df.drop(columns=['target']), df['target']
test_predict = loaded_model.predict(X)
mse = mean_squared_error(y, test_predict)

print('mse_test', mse)
print('Предсказания модели :', test_predict)
