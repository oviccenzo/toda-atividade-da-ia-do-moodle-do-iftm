import pandas as pd
import sklearn.ensemble import IsolationForest

df = pd.read_csv("acessos_sistemas.csv")

model = IsolationForest(
    n_estimators = 100,
    contamination = 0.2,
    random_states = 42
)

df["anomaly"] = model.fit_predict(df)
df["anomaly"] = df["anomaly"].map({1: "Normal", 0 :"Anonal"})

df.to_csv("acessos_sistemas.csv", index=False)

