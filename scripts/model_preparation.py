import os
import glob
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

SEQ_LENGTH = 30  # Модель будет смотреть на предыдущие 30 дней, чтобы предсказать следующий

def create_sequences(data, seq_length):
    """Преобразует плоский массив данных в последовательности для RNN."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Загрузка и объединение всех тренировочных данных
all_train_data = []
for file in glob.glob('data/train/*.csv'):
    df = pd.read_csv(file)
    all_train_data.append(df['temperature_scaled'].values)

full_train_seq = np.concatenate(all_train_data)
X, y = create_sequences(full_train_seq, SEQ_LENGTH)

# Разделение на train/validation внутри тренировочного набора
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Построение модели
model = Sequential([LSTM(50, input_shape=(X_train.shape[1], 1)), Dense(1)])

model.compile(optimizer='adam', loss='mse') # MSE - среднеквадратичная ошибка

print("Начало обучения...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# Сохранение обученной модели
model.save('trained_model.h5')
print("Модель сохранена как trained_model.h5")