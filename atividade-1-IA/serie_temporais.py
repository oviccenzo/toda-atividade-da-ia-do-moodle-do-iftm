import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


tickers = ["PETR4.SA","VALE3.SA","ITUB4.SA","AAPL","TSLA"]
data_inicio = "2019-01-01"
data_fim = "2025-01-01"

def baixar_e_analisar(ticker):
  df = yf.download(ticker, start=data_inicio, end=data_fim)
  print(f"\nAnalise exploratoria(EDA): {ticker}")
  print(df.tail())
  df["Close"].plot(title=f"Historico do fechamento: {ticker}")
  plt.show()
  return df

dados_acoes = {t: baixar_e_analisar(t) for t in tickers}

LOOKBACK = 60

def preparar_sequencial(dados, lookback):
  X, y = [], []
  for i in range(lookback, len(dados)):
    X.append(dados[i-lookback:i, 0])
    y.append(dados[i, 0])
  return np.array(X), np.array(y)

scaler = MinMaxScaler(feature_range=(0,1))

resultados_metricas = {}

for ticker in tickers:
  df = dados_acoes[ticker][["Close"]].values
  scaled_data = scaler.fit_transform(df)

  tamanho_treino = int(len(scaled_data) * 0.8)
  train_data = scaled_data[:tamanho_treino]
  test_data = scaled_data[tamanho_treino:]

  X_train, y_train = preparar_sequencial(train_data, LOOKBACK)
  X_test, y_test = preparar_sequencial(test_data, LOOKBACK)

  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

  model = Sequential([
      LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
      Dropout(0.2),
      LSTM(50, return_sequences=False),
      Dropout(0.2),
      Dense(25),
      Dense(1)
  ])
  model.compile(optimizer="adam", loss='mean_squared_error')
  model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)

  predictions = model.predict(X_test)
  predictions = scaler.inverse_transform(predictions)
  y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))

  mae = mean_absolute_error(y_test_real, predictions)
  rmse = np.sqrt(mean_absolute_error(y_test_real, predictions))
  resultados_metricas[ticker] = {"MAE": mae, "RMSE" : rmse}

  plt.figure(figsize=(10,5))
  plt.plot(y_test_real, label="Real")
  plt.plot(predictions, label="Previsto")
  plt.title(f"Previsão vs Real: {ticker}")
  plt.legend()
  plt.show()

print("\n--- Métricas Finais para o meu README do---")
for t, m in resultados_metricas.items():
    print(f"{t}: MAE = {m['MAE']:.2f}, RMSE = {m['RMSE']:.2f}")


