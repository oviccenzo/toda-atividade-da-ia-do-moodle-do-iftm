#src_4_rules_analise

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("basket_supermercado_1000.csv")
df = df.drop(columns=['transacao'])

#Filtragem da regras relevantes
rules_filtred = rules[
    (rules['lift'] > 1) &
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