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
src_0_carregando_dado_do_supermecado/ carregamento_e_eda.py

src_1_processando_dado_do_supermercado/ pre_processamento.py

src_2_conjunto_de_dados_do_supermecado/ conjunto_frequente_da_cesta_do_supermercado.py

src_3_geracao_das_regras_de_associação/ regras_de_associação_dos_dados_do_supermercado.py

src_4_rules_e_analise / analise_de_dado_do_rules_supermercado.py

src_5_interpretação_dos_negocios_do_supermecado / dados_interpretacao_da_cesta_do_supermecado.py


AS métricas utilizadas são:
Suporte que combinas as frequencias dos produtos combinados durante a execução do codigo para poder mostar o suporte dos produtos

### 🔹 Quais produtos apresentam maior associação entre si?

Os produtos mais associados são aqueles que fazem parte de compras básicas do dia a dia:

- **Arroz ↔ Feijão**
- **Pão ↔ Leite**
- **Café ↔ Açúcar**
- **Carne ↔ Arroz/Feijão**

Essas relações refletem padrões naturais de consumo e hábitos alimentares dos clientes.

---

### 🔹 Existem produtos que funcionam como “âncora” para outras compras?

Sim. Alguns produtos aparecem com alta frequência e puxam outras compras:

- **Pão** → associado a leite, manteiga e café  
- **Arroz** → associado a feijão e proteínas (carne/frango)  
- **Cerveja** → associada a carnes e refrigerante  

Esses produtos funcionam como **gatilhos de compra**, influenciando o cliente a adquirir mais itens.

---

### 🔹 Quais regras possuem maior potencial para ações promocionais?

As melhores regras são aquelas com **alta confiança e alto lift**, por exemplo:

- Combo **Arroz + Feijão**
- Promoção **Café + Açúcar**
- Combinação **Carne + Cerveja**

Essas regras são ideais para:

- Promoções conjuntas  
- Descontos combinados  
- Kits de produtos  

---

### 🔹 Alguma regra encontrada pode ser considerada enganosa ou pouco útil? Por quê?

Sim. Algumas regras apresentam:

- Alta confiança  
- **Lift próximo de 1**

Isso significa que os produtos aparecem juntos **por popularidade**, não por forte relação.

Exemplo:
- Produtos muito comuns (como pão e hortifruti)

Essas regras são **pouco úteis para estratégia**, pois não indicam oportunidade real de venda cruzada.

---

### 🔹 Como os resultados podem impactar o layout do supermercado ou estratégias de venda?

Os resultados podem ser usados para:

### 🏬 Layout
- Colocar **arroz e feijão próximos**
- Agrupar **café, açúcar e doces**
- Posicionar **cerveja próxima a carnes/snacks**

### 💰 Estratégias
- Criar promoções combinadas  
- Desenvolver combos semanais  
- Oferecer recomendações personalizadas  

Isso melhora a experiência do cliente e aumenta o faturamento.

---

## 🛠️ Tecnologias

- Python  
- Pandas  
- Mlxtend  
