import pandas as pd

def executar():
    df = pd.read_csv("acessos_sistema.csv")
    print(f"Dados carregados: {df.shape[0]} registros encontrados")
    return df

if __name__ == "__main__":
    executar()

