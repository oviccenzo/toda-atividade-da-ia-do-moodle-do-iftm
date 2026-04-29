atividade 1 
Previsão de Séries Temporais Financeiras com Redes Neurais

1)O objetivo do trabalho é sobre previsão de serie temporal financeira com rede neural para previsão 
  de series temporais envolvendo no minimo 5 ações e aplicar 
  a rede neural para recorrer e prever o preço do fechamento de ativos 
  financeiro utilizando dados reais e biblioteca de aprendizado profundo

2)Ações utilizada:
  As ações foi selecionado cinco de grande relevancia ao mercado financeiro:

  PETR4.SA (Petroleo brasileiro S.A - Petroleio)
  VALE3.SA (Vale S.A)
  ITUB4.SA (ITAU unibanco Holding S.A)
  AAPL (Apple Inc.)
  TSLA (tesla, inc).

3)O periodo historico utilizado é o:
  Inicio: 01 de janeiro de 2019
  fim: 01 de janeiro de 2025

4)Descrição do modelo de rede neural:
  Foi adotado a uma arquitetura de rede neural recorrente do tipo LSTM(Long Short-Term Memory).
  Essa capcidade justifica a escolha do LSTM em aprender dependencia de longo prazo em sequencia temporais,
  superando ate a rede neural densa tradicionais no tratamento financeiro de dados.

5)A analise de exploratorio consiste em:
  visualização do preço do fechamento
  comparação historico ativos
  calculo de retorno diarios
  A identificação das tendêcias, volatibilidade e padrões

  Esses graficos permitem compreender a dinamica de cada serie individual ante da modelagem

6)Metricas utilizada para avaliação foi:
  O desempenho foi mensurado por:
  mae(mean absolute error): Trata do erro média absoluto em termos monetários
  rmse (root mean squared error): Trata da raiz do erro do rmse quadratico medio, que penaliza erros maiores

7) A intrução para execução:

  git clone https: //github.com

  !pip install yfinance 
  
  !pip install pandas
  
  !pip install matplotlib
  
  !pip install scikit-learn
  
  !pip tensorflow

8) A avaliação do modelo
   A avaliação do desempenho do modelo mostrou ativo mais estéveis do que esperado o exemplo: ITUB4, PETR4 tendem a mostrar o
   erros menores.

   A avalição do desempenho do modelo mostrou ativo voláteis tipo TSLA, AAPL mostram o erro maiores.
   As previsões seguiram preço de padrões coerentes com os preços reais.
  
