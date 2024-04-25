from sklearn.model_selection import train_test_split
from model_prep import data_preparation
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle  # для сохранения модели

model = RandomForestClassifier(max_features='log2', n_estimators=300, random_state=73)
path = '/home/olga/MLOps/lab2/train/train_data.csv'

df = data_preparation(path)
X, y = df.drop(columns=['target']), df['target']

# разбиваем на тестовую и валидационную
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.3,
                                                  random_state=73)
model.fit(X_train, y_train)

pickle.dump(model, open('/home/olga/MLOps/lab2/model.pkl', "wb"))  # сохраняем модель

print('Модель сохранена')
prediction = model.predict(X_val)
print("accuracy:", metrics.accuracy_score(y_val, prediction))