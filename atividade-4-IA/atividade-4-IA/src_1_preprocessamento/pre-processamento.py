import pandas as pd
from sklearn.preprocessing import StandarScaler

df = pd.read_csv("acessos_sistemas.csv")

scaler = StandarScaler()
df_scaler = scaler.fit_transform(df)

df_scaler = pd.DataFrame(df_scaler, columns=df.columns)

df_scaler.to_csv("acessos_sistemas.csv", index=False)

