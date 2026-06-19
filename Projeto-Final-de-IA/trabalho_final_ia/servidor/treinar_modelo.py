import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

dados_ecommerce = {
    'texto': [
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
    ],
    'sentimento': [
        'Positivo', 'Positivo', 'Positivo', 'Positivo', 'Positivo',
        'Negativo', 'Negativo', 'Negativo', 'Negativo', 'Negativo'
    ]
}

df = pd.DataFrame(dados_ecommerce)
print(f"Base de dados carregada com {len(df)} avaliações de exemplo.")

print("Transformando palavras em números...")
vectorizar = CountVectorizer()
X = vectorizar.fit_transform(df['texto'])
y = df['sentimento']

print("Treinando o algoritmo de Regressão Logística...")
modelo = LogisticRegression()
modelo.fit(X, y)

print("Salvando o cérebro da IA para o flask...")
joblib.dump(modelo, 'modelo_sentimento.pkl')
joblib.dump(vectorizar, 'vetorizador.pkl') 

print("\n=======================================================")
print("--> SUCESSO: 'modelo_sentimento.pkl' e 'vetorizador.pkl'")
print("    foram criados com sucesso dentro da pasta servidor!")
print("=======================================================")