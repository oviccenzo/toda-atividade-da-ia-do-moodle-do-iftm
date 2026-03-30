import pandas as pd
import os

from src_0_carga.carga import carregar_dados
from src_1_exploracao.exploracao import analisar_dados
from src_2_preprocessamento.preparo import escalar_dados
from src_3_modelagem.modelo import treinar_isolation_forest
from src_4_identificacao.rotulagem import identificar_anomalias
from src_5_visualizacao.graficos import gerar_grafico

def main():
    print("🚀 Inciando Pipeline de Inteligência Artificial...")
    
    # Configuração de caminhos
    arquivo_input = "acessos_sistema.csv"
    arquivo_output = "resultado_final_ia.csv"

    # Verificação de segurança
    if not os.path.exists(arquivo_input):
        print(f"❌ Erro: O arquivo {arquivo_input} não foi encontrado na raiz!")
        return

    # --- ETAPA 0: CARGA ---
    print("\n[Etapa 0] Carregando dados...")
    df = carregar_dados(arquivo_input)

    # --- ETAPA 1: EXPLORAÇÃO ---
    print("\n[Etapa 1] Analisando estrutura dos dados...")
    analisar_dados(df)

    # --- ETAPA 2: PREPROCESSAMENTO ---
    print("\n[Etapa 2] Normalizando dados (Scaling)...")
    df_scaled, scaler = escalar_dados(df)

    # --- ETAPA 3: MODELAGEM ---
    print("\n[Etapa 3] Treinando modelo Isolation Forest (Contaminação=0.15)...")
    modelo = treinar_isolation_forest(df_scaled, contaminacao=0.15)

    # --- ETAPA 4: IDENTIFICAÇÃO ---
    print("\n[Etapa 4] Rotulando anomalias e salvando resultados...")
    df_final = identificar_anomalias(modelo, df_scaled, df)
    df_final.to_csv(arquivo_output, index=False)
    print(f"✅ Arquivo '{arquivo_output}' gerado com sucesso!")

    # --- ETAPA 5: VISUALIZAÇÃO ---
    print("\n[Etapa 5] Gerando visualização gráfica...")
    gerar_grafico(df_final)
    
    print("\n✨ Processo concluído com sucesso!")

if __name__ == "__main__":
    main()
