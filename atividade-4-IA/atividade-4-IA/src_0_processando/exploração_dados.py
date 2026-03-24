import pandas as pd 

df = pd.read_csv("acessos_sistemas.csv")

print(head())
print("\n")
print(df.describe())
print("\n")
print(df.info())
print("\n")
print(df.isnull().sum()) 