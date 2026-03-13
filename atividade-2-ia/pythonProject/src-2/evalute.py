# src-2/evaluate.py

import sys, os
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# ---- IMPORTANTE ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_PATH)

DATA_PATH = os.path.join(BASE_DIR, "data", "titanic.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_titanic.pkl")

def main():
    df = pd.read_csv(DATA_PATH)
    df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    print("Carregando modelo...")
    model = joblib.load(MODEL_PATH)

    print("Gerando predições...")
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    print("\n=== MATRIZ DE CONFUSÃO ===")
    print(confusion_matrix(y, pred))

    print("\n=== RELATÓRIO ===")
    print(classification_report(y, pred))

    print("ROC AUC:", roc_auc_score(y, proba))

if __name__ == "__main__":
    main()