import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def executar():
    df = pd.read_csv("resultado_final.csv")

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='hora_acesso', y='duracao_sessao_min',
                    hue='resultado', palette={'Normal': 'blue', 'Anomalia': 'red'}, s=100)

    plt.title("Detecção de Anomalias: Hora vs Duração (Identificadas pela IA)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("src_5_visualizacao/grafico_anomalias.png")
    print("✅ Gráfico salvo em 'src_5_visualizacao/grafico_anomalias.png'.")
    plt.show()


if __name__ == "__main__":
    executar()
