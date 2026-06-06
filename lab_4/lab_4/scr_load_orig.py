from catboost.datasets import titanic

train_df, test_df = titanic()
train_df.to_csv('titanic_orig.csv', index=False)
