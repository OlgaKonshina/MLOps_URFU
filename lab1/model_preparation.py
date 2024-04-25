from sklearn.model_selection import train_test_split
from model_preprocessing import data_preparation
from sklearn.svm import SVR  # Метод опорных векторов для регрессии scikit-learn
import pickle  # для сохранения модели

model = SVR()
path = '/home/olga/MLOps/train/train_data.csv'

df = data_preparation(path)
X, y = df.drop(columns=['target']), df['target']

# разбиваем на тестовую и валидационную
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.3,
                                                  random_state=73)
model.fit(X_train, y_train)

pickle.dump(model, open('/home/olga/MLOps/model.pkl', "wb"))  # сохраняем модель

print('Модель сохранена')