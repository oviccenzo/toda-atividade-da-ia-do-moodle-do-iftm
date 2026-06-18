# Analisador de Sentimentos em Avaliações de E-commerce com IA 🤖

**Trabalho Final da disciplina de Inteligência Artificial** **Instituição:** IFTM - Campus Ituiutaba  
**Curso:** Ciência da Computação (7º Período)  
**Autor:** Viccenzo Oliveira  

---

## 1. Descrição do Problema
Com o alto volume de vendas no e-commerce moderno, ler e classificar manualmente o feedback dos clientes é um processo inviável, lento e custoso. Identificar rapidamente se uma avaliação de produto é positiva ou negativa é crucial para a tomada de decisão estratégica das empresas. O problema consiste em automatizar a triagem e a interpretação desses textos em larga escala utilizando computação inteligente.

## 2. Descrição da Solução Proposta
Este projeto utiliza Inteligência Artificial, especificamente técnicas de **Processamento de Linguagem Natural (PLN)** e **Machine Learning**, para analisar textos de avaliações e classificá-los automaticamente. 

O modelo foi treinado com uma base de dados real do e-commerce brasileiro (dataset B2W-Reviews01, disponibilizado pela Americanas) utilizando vetorização (*Bag of Words*) e o algoritmo de Regressão Logística. 

Para a interação com o usuário final, foi desenvolvida uma **Single Page Application (SPA)**. O backend foi construído em Flask atuando como uma API REST, enquanto o frontend utiliza requisições assíncronas (Fetch API em JavaScript) para entregar os resultados da IA em tempo real na tela, de forma suave e sem recarregar a página.

---

## 3. Explicação Teórica do Exercício (Como a IA funciona)

A máquina não lê textos como os humanos; ela processa matrizes matemáticas. O fluxo deste exercício baseia-se em duas etapas principais da IA clássica:

### A. Vetorização (Bag of Words com CountVectorizer)
Para transformar as palavras das avaliações em números, o modelo cria um dicionário estatístico com as 5.000 palavras mais frequentes do dataset da Americanas. Cada frase vira um vetor numérico que contabiliza a presença e a frequência dessas palavras. Se a palavra "excelente" aparece em uma avaliação, a coluna correspondente a essa palavra recebe o valor `1`, caso contrário, recebe `0`.

### B. O Algoritmo de Regressão Logística
A Regressão Logística é um algoritmo de aprendizado supervisionado ideal para **classificação binária** (onde o resultado é mapeado estritamente entre `0` e `1`, ou seja, Negativo ou Positivo). 

Durante a fase de treinamento (`modelo.fit`), o algoritmo analisa os vetores gerados e aprende o peso matemático de cada palavra. Palavras como *"ótimo"*, *"recomendo"* e *"lindo"* recebem pesos positivos elevados. Palavras como *"ruim"*, *"quebrou"* e *"péssimo"* recebem pesos negativos. 

Ao receber uma frase inédita, o modelo aplica a função sigmoide à soma ponderada dos pesos das palavras encontradas, gerando a probabilidade estatística (Confiança) e definindo o veredito final do sentimento.

---

## 4. Estrutura do Projeto
O projeto está organizado na seguinte estrutura de diretórios:

```text
trabalho_ia/
│
├── app.py                   # Servidor web (Backend em Flask)
├── treinar_modelo.py        # Script para baixar os dados e treinar a IA
├── modelo_sentimento.pkl    # Modelo matemático treinado (Gerado pelo script)
├── vetorizador.pkl          # Vocabulário treinado (Gerado pelo script)
├── requirements.txt         # Lista de dependências do projeto
│
├── static/                  # Arquivos estáticos do Frontend
│   ├── style.css            # Estilização visual (CSS)
│   └── script.js            # Lógica de requisições assíncronas (JS)
│
└── templates/               # Arquivos de renderização
    └── index.html           # Interface visual principal (HTML)

5. Tecnologias e Bibliotecas Utilizadas

    Linguagem Principal: Python 3.14

    Machine Learning & PLN: scikit-learn (LogisticRegression, CountVectorizer)

    Manipulação de Dados: pandas

    Backend Web: Flask

    Persistência do Modelo: joblib

    Frontend: HTML5, CSS3 e JavaScript Vanilla (Fetch API)

6. Instruções de Execução (Passo a Passo no Terminal)

Se você estiver utilizando o ambiente do PyCharm no Windows 11, abra a aba Terminal na barra inferior e execute os comandos exatamente nesta sequência cronológica:

Passo 1: Criar o Ambiente Virtual (Venv)
Cria uma bolha isolada para instalar as dependências sem precisar de permissões de administrador no computador:
Bash

python -m venv venv

Passo 2: Ativar o Ambiente Virtual
Ativa o ambiente isolado. Note que a palavra (venv) aparecerá no início da linha do terminal:
Bash

venv\Scripts\activate

Passo 3: Instalar as dependências do projeto
Instala todos os pacotes necessários listados no gerenciador de requisitos:
Bash

pip install -r requirements.txt

Passo 4: Compilar e Treinar a Inteligência Artificial
Roda o script que faz o download do dataset real, treina os algoritmos e exporta os arquivos binários .pkl:
Bash

python treinar_modelo.py

(Aguarde até que a mensagem de SUCESSO apareça no console indicando a criação do modelo).

Passo 5: Executar a Aplicação Web
Inicia o servidor local do Flask para integrar a interface gráfica à IA treinada:
Bash

python app.py

Passo 6: Acessar no Navegador
O terminal exibirá a linha de execução local. Segure a tecla Ctrl e clique no link gerado ou digite o endereço abaixo no seu navegador:
Plaintext

[http://127.0.0.1:5000](http://127.0.0.1:5000)

Para encerrar o servidor web após os testes, basta clicar no terminal do PyCharm e pressionar a combinação de teclas Ctrl + C.
Lembrete Importante

O arquivo requirements.txt precisa existir na pasta raiz para o comando do Passo 3 funcionar. Crie ele com as seguintes linhas dentro:
Plaintext

flask
pandas
scikit-learn
joblib
