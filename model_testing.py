import pickle # для сохранения модели
from model_prep import data_preparation  # для обработки данных
from sklearn import metrics
model_path = '/home/olga/MLOps/model.pkl'
test_path = '/home/olga/MLOps/lab2/test/test_data.csv'
loaded_model = pickle.load(open(model_path, 'rb'))  # загружаем модель

df = data_preparation(test_path)  # загружаем тестовые данные
X, y = df.drop(columns=['target']), df['target']
test_predict = loaded_model.predict(X)
print("accuracy:", metrics.accuracy_score(y, test_predict))