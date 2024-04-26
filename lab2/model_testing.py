import pickle # для сохранения модели
from model_prep import data_preparation  # для обработки данных
from sklearn import metrics
model= 'model.pkl'
test = 'test.csv'
loaded_model = pickle.load(open(model, 'rb'))  # загружаем модель

df = data_preparation(test)  # загружаем тестовые данные
X, y = df.drop(columns=['target']), df['target']
test_predict = loaded_model.predict(X)
print("accuracy:", metrics.accuracy_score(y, test_predict))
