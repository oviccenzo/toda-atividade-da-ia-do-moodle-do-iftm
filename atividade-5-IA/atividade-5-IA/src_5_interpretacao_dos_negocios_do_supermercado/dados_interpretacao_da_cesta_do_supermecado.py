# src_5_interpretacao_dos_negocios_do_supermercado/dados_interpretacao_da_cesta_do_supermecado.py

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("basket_supermercado_1000.csv")

df = df.drop(columns=['transacao'])

frequent_itemsets = apriori(
    df,
    min_support=0.05,
    use_colnames=True
)

rules = association_rules(
    frequent_itemsets,
    metric='confidence',
    min_threshold=0.6
)

rules = rules.sort_values(
    by='lift',
    ascending=False
)

#Insights para o negocios das compras
print("\nO insights de negocios das compras")

for _, row in rules.head(10).iterrows():
  antecedentes = ', '.join(list(row['antecedents']))
  consequentes = ', '.join(list(row['consequents']))

  print(f'Clientes que compram[{antecedentes}] que tendem a comprar[{consequentes}]')
  print(f"Suporte: {row['support']:.2f}")
  print(f"Confiança: {row['confidence']:.2f}")
  print(f"Lift: {row['lift']:.2f}")
  print("-" * 50)
