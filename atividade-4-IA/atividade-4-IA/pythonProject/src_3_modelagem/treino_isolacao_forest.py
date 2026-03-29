import pandas as pd
from sklearn.ensemble import IsolationForest

def executar():
    df_proc = pd.read_csv("dados_processados.csv")
    # Criando o modelo (Contaminação de 15% para detectar anomalias raras)
    model = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
    model.fit(df_proc)
    print("Modelo Isolation Forest treinado com sucesso!")
    return model

if __name__ == "__main__":
    executar()
