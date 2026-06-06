import pandas as pd

df3 = pd.read_csv('titanic_v2.csv')

sex_dummies = pd.get_dummies(df3['Sex'], prefix='Sex')
df3 = pd.concat([df3, sex_dummies], axis=1)

df3.drop('Sex', axis=1, inplace=True)

df3.to_csv('titanic_v3.csv', index=False)
