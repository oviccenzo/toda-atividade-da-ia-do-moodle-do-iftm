# src_2_treinando_o_modelo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def treinar_o_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test

if __name__ == "__main__":
    X = pd.read_csv('X.csv')
    y = pd.read_csv('y.csv')

    X_test, X_pred , y_train ,y_test = train_test_split(X,y)

    print("Shape X: ", X.shape)
    print("Shape X_test: ", X_test.shape)

