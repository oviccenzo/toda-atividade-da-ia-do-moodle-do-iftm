Essa atividade e sobre analisar a comprar e ver a taxa de variação de comprar ao utilizar a tecnica de cesta de compra analise ultilizando o algoritmo apriori para analisar os comportamento das compras dos clientes de um supermecadp de médio porte.A seguir os padrão de dados do dataset a partir dos padrões de identidificados são gerado pela insights de estratégico de comprar dos basket_supermercado_1000.csv e para apioar as decisões gerenciais relacionadas a layiut , promoções e aumento do médio ticket.

Autor desse trabalho:
Nome: Viccenzo de Oliveira Nunes Resende
Cursos: Ciências da Computação
Professor: André luiz franca batista

O Objetivo principal do trabalho
O objetivo principal é identificar as associações relevantes entre os prudotos do cliente e transformar tais, como:
- O planejamento de produto combinado a promoções
- Organizar e planeja as estratégias das gôndolas
- Identificar os produtos em âncoras
- Aumentar ticket médio por clientes

## Dataset
** 'basket_supermecado_1000.csv' **

As característica do dataset:
- 1000 transações
- A cada linha representar uma compra
- A cada coluna representar um produto
- Cada Valores binários:
- '1' - produto comprado
- '0' - não comprado

Os produtos analisado incluem itens básicos, alimentos, bebidas, hortifruti, doces e produto de limpezas


As metodologias minhas sobre a cesta de supermercado 1000 compra segue:
1. O carregamento e exploração dos dados
2. O pré-processamento dos dados
3. A extração dos dados de conjuntos frequentes que é p algoritmo(apriori)
4. A geração de insights para gerar regras de associação
5. A analises dos dados métricas sobre os dados seria (suporte, confiança e lift)
6. A a interpretação dos dados dos resultado sob a ótica dos negócios

A descrição do repositórios:
[acessar a pasta src_0_carregando_dado_do_supermecado](carregamento_e_eda.py)
