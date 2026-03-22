# src_3_avaliacao
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def avaliarModelo(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print(f"Acuracia: {accuracy_score(y_test,y_pred):.4f}")

    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm,annot=True,fmt='d',cmap="Reds")
    plt.title("A matriz da confusão")
    plt.xlabel("Prediccao da avaliação")
    plt.ylabel("Actual da availiação")
    plt.show()

    print("\nO relatorio da classifcacao da avaliacao do modelo da acuracia: ")
    print(classification_report(y_test,y_pred))

if __name__ == "__main__":

    from sklearn.dummy import DummyClassifier
    import pandas as pd
    import numpy as np

    X_test = pd.DataFrame(np.random.rand(20, 3))
    y_test = np.random.randint(0, 2, size=20)

    modelo = DummyClassifier(strategy="most_frequent")
    modelo.fit(X_test, y_test)

    avaliarModelo(modelo, X_test, y_test)
