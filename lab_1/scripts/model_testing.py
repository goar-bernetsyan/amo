import os
import glob
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

SEQ_LENGTH = 30

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Загрузка модели
try:
    model = load_model('trained_model.h5')
except OSError:
    print("Ошибка: Файл модели trained_model.h5 не найден.")
    print("Сначала запустите model_preparation.py для обучения модели.")
    exit()

results = []

# Тестирование (на каждом файле из папки test по отдельности)
for file in glob.glob('data/test/*.csv'):
    df = pd.read_csv(file)
    test_seq = df['temperature_scaled'].values
    X_test, y_test = create_sequences(test_seq, SEQ_LENGTH)

    X_test = np.reshape(X_test, (*X_test.shape, 1))

    predictions = model.predict(X_test)

    # Вычисление ошибки
    mse = mean_squared_error(y_test, predictions)
    results.append((file, mse))
    print(f"Тестирование файла {os.path.basename(file)} завершено. MSE: {mse:.4f}")

# Итоговый отчет
avg_mse = sum(mse for _, mse in results) / len(results)
print("\n=== ОТЧЕТ О ТЕСТИРОВАНИИ ===")
print(f"Средняя ошибка (MSE) на тестовых данных: {avg_mse:.4f}")
print("Чем ниже MSE, тем точнее модель.")