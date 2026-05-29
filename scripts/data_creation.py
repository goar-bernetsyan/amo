import os
import numpy as np
import pandas as pd

DATASETS_COUNT = 5         # Сколько файлов надо создать
DAYS_IN_YEAR = 365
NOISE_LEVEL = 2.0          # Уровень обычного шума
ANOMALY_CHANCE = 0.05       # Вероятность появления аномалии в день

def generate_dataset(days=DAYS_IN_YEAR):
    """Генерирует один DataFrame с данными о температуре."""
    dates = pd.date_range(start='2026-01-01', periods=days)
    
    # Базовая синусоида имитирует годичный цикл температур
    base_temp = 15 + 7 * np.sin(2 * np.pi * np.arange(days) / days)
    
    # Случайный шум
    noise = np.random.normal(scale=NOISE_LEVEL, size=days)
    
    # Аномалии: резкий скачок на 10 градусов вверх или вниз
    anomalies = np.where(np.random.rand(days) < ANOMALY_CHANCE,
                         np.random.choice([10, -10], size=days), 0)
                         
    temperatures = base_temp + noise + anomalies
    return pd.DataFrame({'date': dates, 'temperature': temperatures})

os.makedirs('data/train', exist_ok=True)
os.makedirs('data/test', exist_ok=True)

# Генерация и сохранение файлов
for i in range(DATASETS_COUNT):
    df = generate_dataset()
    filename = f'data_{i + 1}.csv'
    if i < 4:
        path = os.path.join('data', 'train', filename)
    else:
        path = os.path.join('data', 'test', filename)
    df.to_csv(path, index=False)
    print(f'Создан файл: {path}')