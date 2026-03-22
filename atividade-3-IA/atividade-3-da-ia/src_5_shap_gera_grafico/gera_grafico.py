# src_6_shap_gera_grafico
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def gerar_grafico(model,X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values[1], X_test)

    return explainer, shap_values

if __name__ == "__main__":
    X = pd.read_csv("X.csv")
    y = pd.read_csv("y.csv")

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y.values.ravel())

    print("Gerando o grafico do modelo do shap global")
    gerar_grafico(model, X)