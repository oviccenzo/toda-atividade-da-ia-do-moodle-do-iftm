# src_1_processando
import pandas as pd

def processar_dataset_dos_dados(df):
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # As colunas sendo removida
    df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"], inplace=True)

    #Tratamento da categorica
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    #Seperara X e Y
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X.to_csv("X.csv", index=False)
    y.to_csv("y.csv", index=False)

    return X,y

if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("titanic.csv")
    X, y = processar_dataset_dos_dados(df)

    print("X: ")
    print(X.head())