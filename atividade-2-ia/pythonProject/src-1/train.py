# src-1/train.py

import sys, os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_PATH)

# Agora podemos importar preprocess normalmente
from preprocesso import build_preprocessor

DATA_PATH = os.path.join(BASE_DIR, "data", "titanic.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_titanic.pkl")

def main():

    print("Carregando dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    print("Construindo preprocessor...")
    preprocessor = build_preprocessor()

    print("Criando pipeline do modelo...")
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", model)])

    print("Treinando...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

    print("\n==== AVALIAÇÃO ====")
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    print("ROC AUC:", roc_auc_score(y_test, proba))

    print("Cross-validation...")
    cv = cross_val_score(pipe, X, y, cv=5)
    print(f"CV Acc: {cv.mean():.4f} ± {cv.std():.4f}")

    print("Salvando modelo...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    print(f"\nModelo salvo em: {MODEL_PATH}")

if __name__ == "__main__":
    main()