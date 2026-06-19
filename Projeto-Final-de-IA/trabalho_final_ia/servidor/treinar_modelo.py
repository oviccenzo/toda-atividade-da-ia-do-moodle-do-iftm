import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

dados_ecommerce = [
    'Excelente produto, chegou antes do prazo e funciona muito bem!.', 
    'Muito bem superou minhas expectativas, recomendo a todos.' ,
    'Sensacional! O melhor investimento que fiz este ano, perfeito.',
    'Adorei o design e a qualidade do material é impecável.',
    'Enrega rápido e produto de altíssima qualidade.',
    'Péssimo produto, veio quebrado e a entregou atrasou muito.',
    'Não gostei. O material é muito frágil e parou de funcionar em dois dias.',
    'Muito ruim odiei.Não condiz com a descrição do anúncio.',
    'Dinheiro jogado fora, o aparelho trava o tempo todo e esquenta.',
    'Defeituoso e fraco.Solicitei a devolução imediatamente'
]


