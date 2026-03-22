# src_4_feature_importacao
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def importacao_da_variavel(model,X):
    importances = model.feature_importances_
    features = X.columns

    df_imp = pd.DataFrame({
        'Feature':features,
        'Importance':importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=df_imp)
    plt.title("Features Importance")
    plt.show()

    print(df_imp)

if __name__ == '__main__':
    print("Importacao dos variaveis")
    df = pd.read_csv('titanic.csv')

    X = df[['Pclass','Age','SibSp','Parch','Fare']].fillna(0)
    y = df['Survived']

    model = RandomForestClassifier()
    model.fit(X,y)

    importacao_da_variavel(model , X)