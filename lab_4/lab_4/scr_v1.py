import pandas as pd

df = pd.read_csv('titanic_orig.csv')

print("До:", df.columns.tolist())

df_subset = df[['Pclass', 'Sex', 'Age']]

print("После:", df_subset.columns.tolist())

df_subset.to_csv('titanic_v1.csv', index=False)
