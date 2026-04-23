#src_4_rules_analise

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


df = pd.read_csv("basket_supermercado_1000.csv")
df = df.drop(columns=['transacao'])

frequent_itemsets = apriori(df,min_support=0.05,use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence',min_threshold=0.6)

#Filtragem da regras relevantes
rules_filtred = rules[
    (rules['lift'] > 1.2) &
    (rules['support'] >= 0.05)
]

#Regras relevantes
print("As regras relevantes para o negocio do dataset de supermecado: \n")
print(rules_filtred[[
    'consequent support',
    'confidence',
    'support',
    'lift'
]])