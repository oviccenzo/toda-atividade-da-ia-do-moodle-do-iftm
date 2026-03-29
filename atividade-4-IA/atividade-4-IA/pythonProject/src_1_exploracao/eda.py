import pandas as pd

def executar():
    df = pd.read_csv("acessos_sistema.csv")
    print("Resumo da estatistica: ")
    print(df.describe())
    print("Colunas estatistica: ")
    print(df.columns.tolist())

if __name__ == "__main__":
    executar()