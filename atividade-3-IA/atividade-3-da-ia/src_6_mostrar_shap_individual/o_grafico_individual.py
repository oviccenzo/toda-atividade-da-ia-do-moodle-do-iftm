# src_6_mostrar_shap_individual
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def mostrar_individual(explainer, shap_values, X_test, index=0):

    if isinstance(shap_values, list):
        v = shap_values[1][index]
        ev = explainer.expected_value[1]
    else:
        v = shap_values
        ev = explainer.expected_value[1]

    exp = shap.Explanation(
        values=v,
        base_values=ev,
        data=X_test.iloc[index],
        feature_names=X_test.columns.tolist()
    )

    shap.plots.waterfall(exp, show=False)

    plt.title(f"A explicação indvidual do dataset do titanic {index}")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    X = pd.read_csv('X.csv')
    y = pd.read_csv('y.csv').values.ravel()

    model = RandomForestClassifier(random_state=42).fit(X,y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mostrar_individual(explainer, shap_values, X, index=0)