#!/bin/bash

echo "=== Запуск проекта ==="

echo "Шаг 1: Генерация данных..."
python scripts/data_creation.py

echo "Шаг 2: Предобработка данных..."
python scripts/model_preprocessing.py

echo "Шаг 3: Обучение модели..."
python scripts/model_preparation.py

echo "Шаг 4: Тестирование модели..."
python scripts/model_testing.py

echo "=== Готово! ==="
