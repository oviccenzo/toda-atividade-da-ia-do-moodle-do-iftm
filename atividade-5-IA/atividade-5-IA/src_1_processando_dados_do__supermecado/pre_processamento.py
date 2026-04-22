#src_1_processando_dados_do_supermecado/pre_processamento.py
import pandas as pd
df = pd.read_csv('basket_supermecado_1000.csv')

df = df.drop(columns=['transacao'])

df = df.applymap(lambda x: 1 if x > 0 else 0)

print(df.head())
