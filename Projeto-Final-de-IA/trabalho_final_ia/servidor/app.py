from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# ==============================================================================
# CARREGAMENTO DA IA
# ==============================================================================
# Aqui o Flask carrega a inteligência que você treinou com os 10.000 dados da Americanas
try:
    modelo = joblib.load('modelo_sentimento.pkl')
    vetorizador = joblib.load('vetorizador.pkl')
    print("IA carregada com sucesso! Servidor pronto.")
except FileNotFoundError:
    print("ERRO: Arquivos .pkl não encontrados. Rode o treinar_modelo.py primeiro!")

# ==============================================================================
# ROTAS DO SITE
# ==============================================================================
# Rota 1: Carrega a página inicial vazia
@app.route('/')
def home():
    return render_template('index.html')

# Rota 2: Recebe o texto digitado, passa pela IA e devolve o resultado
@app.route('/analisar', methods=['POST'])
def analisar():
    # 1. Pega o que o usuário digitou na caixa de texto
    texto_usuario = request.form['avaliacao']
    
    # 2. Transforma o texto em matemática usando o vocabulário da Americanas
    vetor = vetorizador.transform([texto_usuario])
    
    # 3. A IA faz a previsão (1 = Positivo, 0 = Negativo)
    predicao = modelo.predict(vetor)[0]
    
    # 4. Calcula a % de certeza da IA
    confianca = modelo.predict_proba(vetor)[0][predicao] * 100
    
    # 5. Formata a resposta
    sentimento = "Positivo" if predicao == 1 else "Negativo"
    
    # 6. Atualiza a página web com o veredito
    return render_template('index.html', 
                           texto_analisado=texto_usuario, 
                           resultado=sentimento, 
                           confianca=round(confianca, 1))

if __name__ == '__main__':
    # Inicia o servidor local na porta 5000
    app.run(debug=True)