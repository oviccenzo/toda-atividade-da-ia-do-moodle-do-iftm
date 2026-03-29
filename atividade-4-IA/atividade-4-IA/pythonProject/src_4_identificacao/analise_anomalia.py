import pandas as pd
from sklearn.ensemble import IsolationForest


def executar():
    df_orig = pd.read_csv("acessos_sistema.csv")
    df_proc = pd.read_csv("dados_processados.csv")

    model = IsolationForest(contamination=0.15, random_state=42)
    # AQUI O MODELO CRIA A COLUNA QUE ESTÁ FALTANDO
    df_orig['anomalia'] = model.fit_predict(df_proc)

    # Mapeia -1 para 'Anomalia' e 1 para 'Normal'
    df_orig['resultado'] = df_orig['anomalia'].map({1: 'Normal', -1: 'Anomalia'})

    df_orig.to_csv("resultado_final.csv", index=False)
    print("✅ Coluna de anomalia gerada! Verifique 'resultado_final.csv'.")
    print("\nContagem de resultados:")
    print(df_orig['resultado'].value_counts())


if __name__ == "__main__":
    executar()
