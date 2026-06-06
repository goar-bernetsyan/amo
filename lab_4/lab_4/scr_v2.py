import pandas as pd

df2 = pd.read_csv('titanic_v1.csv')

missing_before = df2['Age'].isna().sum()
print(f"До: {missing_before} пропущенных значений в 'Age'.")

mean_age = df2['Age'].mean()
df2['Age'].fillna(mean_age, inplace=True)

missing_after = df2['Age'].isna().sum()
print(f"После: {missing_after} пропущенных значений в 'Age'.")

# Сохраняем обновлённый датасет
df2.to_csv('titanic_v2.csv', index=False)
