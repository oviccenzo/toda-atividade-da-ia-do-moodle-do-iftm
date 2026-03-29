import pandas as pd
from sklearn.preprocessing import StandardScaler

def executar():
    df = pd.read_csv("acessos_sistema.csv")
    scaler = StandardScaler()
    # Normaliza os dados para o modelo não ser enviesado por números grandes
    df_scaled = scaler.fit_transform(df)
    df_final = pd.DataFrame(df_scaled, columns=df.columns)
    df_final.to_csv("dados_processados.csv", index=False)
    print("Dados normalizados e salvos em 'dados_processados.csv'.")

if __name__ == "__main__":
    executar()
