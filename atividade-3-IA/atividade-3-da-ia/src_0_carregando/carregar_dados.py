#src_0_carregando
import pandas as pd

def carregar_dados_dos_titanic():
    df = pd.read_csv("titanic.csv")

    print("Primeira linha: ")
    print(df.head())

    print("\nInfromacao do dataset do titanic: ")
    print(df.info())

    print("Valores nulo do dataset do titanic: ")
    print(df.isnull().sum())

    return df

if __name__ == "__main__":
    carregar_dados_dos_titanic()