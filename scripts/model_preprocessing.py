import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(folder_path):
    """
    Находит все CSV-файлы в указанной папке, обрабатывает их и сохраняет обратно.
    """
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        
        # Преобразование даты в полезные фичи
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        
        # Нормализация температуры
        scaler = StandardScaler()
        df['temperature_scaled'] = scaler.fit_transform(df[['temperature']])
        
        df.to_csv(file_path, index=False)
        print(f'Обработан файл: {file_path}')

if __name__ == "__main__":
    preprocess_data('data/train')
    preprocess_data('data/test')