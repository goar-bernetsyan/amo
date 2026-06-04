import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Загрузка данных
iris = load_iris()
X = iris.data # Признаки (длина и ширина чашелистика и лепестка)
y = iris.target # Целевая переменная (номер вида цветка)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель логистической регрессии
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Оцениваем модель на тестовой выборке
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Сохраняем модель
joblib.dump(model, 'iris_model.pkl')

# Предсказание для тестового примера из задания: [1, 1, 1, 1]
test_example = [[1, 1, 1, 1]]
prediction = model.predict(test_example)
predicted_class_name = iris.target_names[prediction[0]]

print(f"Точность модели на тестовой выборке: {accuracy:.2f}")
print(f"Тестовый пример: {test_example[0]}")
print(f"Предсказанный класс: {predicted_class_name}")














