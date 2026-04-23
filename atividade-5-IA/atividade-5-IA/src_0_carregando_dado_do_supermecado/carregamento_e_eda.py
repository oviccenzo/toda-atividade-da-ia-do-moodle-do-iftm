# src_0_carregando_dado_do_supermecado/carregamento_e_eda.py/carregamento_e_eda.py
import pandas as pd

df = pd.read_csv("basket_supermercado_1000.csv")
print(f"São as dimensões do dataset: {df.shape}")

print(df)

print(df.head())

print(f"As primeiras linha dos dataset: \n{df.head()}")

print(f"São os resumos dos dataset: \n{df.describe()}")

print(f"São os valores ausentes por colunas dos dataset: \n{df.isnull().sum()}")
df_produto = df.drop(columns=['transacao'])
print(df_produto)

print(f"São as frequencia de compras dos produtos dos dataset:",
f"{df_produto.sum().sort_values(ascending=False)}")