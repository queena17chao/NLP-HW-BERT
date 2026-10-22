import pandas as pd

df = pd.read_csv("data/data.csv")

print(df.head())
print(df.columns)
print(df.shape)
print(df["emotion"].value_counts())

print("程式執行完成")