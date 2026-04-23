# python3.8 src_3_geracao_das_regras/regra_de_associacao_dos_dados_do_supermecado.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("basket_supermercado_1000.csv")
df = df.drop(columns=['transacao'])

df

#Aplicar algoritmo apri
frequent_itemsets = apriori(
    df,
    min_support=0.01,
    use_colnames=True
)

print("Itemsets/Subconjuntos frequentes:")
print(frequent_itemsets.sort_values(by = "support", ascending=True))


# regra de associação
rules = association_rules(
    frequent_itemsets,
    metric = 'confidence',
    min_threshold=0.5
)

print("Regras de associação:")
print(rules.sort_values(by='confidence', ascending=False))

print(rules[[
    'consequent support',
    'confidence',
    'support',
    'lift'
]].sort_values(by='confidence', ascending=False))