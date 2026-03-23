# src_1_treinando_o_modelo/train.py

import sys, os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

# 1. Ajuste de Caminhos
BASE_DIR = os.getcwd()
SRC_PATH = os.path.join(BASE_DIR, "src_0_processando")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Import do seu arquivo customizado
from preprocesso import build_preprocessor

# DATA_PATH deve apontar para a pasta 'data' na raiz, não dentro de 'src_0_processando'
DATA_PATH = os.path.join(BASE_DIR, "data", "titanic.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_titanic.pkl")

def main(): # Voltamos para o nome main() para bater com a chamada lá embaixo
    if not os.path.exists(DATA_PATH):
        print(f"Erro: arquivo não encontrado em {DATA_PATH}")
        return

    print("Carregando o dataset do arquivo 'titanic.csv'...")
    df_raw = pd.read_csv(DATA_PATH)

    cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    df = df_raw[cols].copy()
    df = df.dropna(subset=["Survived"])

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    print("Construindo preprocessor...")
    preprocessor = build_preprocessor()

    print("Criando o pipeline do modelo...")
    # CORREÇÃO: Pipeline usa steps=[...] com letras minúsculas e tuplas
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", model)])

    print("Treinando o modelo...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

    print("\n==== Avaliacao do modelo ====")
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    print(f"ROC AUC: {roc_auc_score(y_test, proba):.4f}")

    print("\nCross-validation...")
    cv = cross_val_score(pipe, X, y, cv=5)
    print(f"CV ACC: {cv.mean():.4f} +- {cv.std():.4f}")

    print("\nSalvando o modelo...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")

if __name__ == "__main__":
    main()
