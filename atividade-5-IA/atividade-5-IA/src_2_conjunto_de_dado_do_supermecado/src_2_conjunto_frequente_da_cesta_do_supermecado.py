#src_2_conjunto_de_dado_do_supermecado/src_2_conjunto_frequente_da_cesta_do_supermecado.py

import pandas as pd
from mlxtend.frequent_patterns import apriori

df = pd.read_csv("basket_supermercado_1000.csv")
df = df.drop(columns=['transacao'])

frequent_itemsets = apriori(
    df,
    min_support=0.05,
    use_colnames=True
)

print(frequent_itemsets.sort_values(by='support', ascending=False))
