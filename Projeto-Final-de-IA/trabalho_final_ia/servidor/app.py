import os
from flask import Flask, request, render_template
import joblib

pasta_atual = os.path.abspath(os.path.dirname(__file__))

pasta_templates = os.path.join(pasta_atual, 'templates')
pasta_static = os.path.join(pasta_atual, 'static')
caminho_modelo = os.path.join(pasta_atual, 'modelo_sentimento.pkl')
caminho_vetorizador = os.path.join(pasta_atual, 'vetorizador.pkl')

app = Flask(__name__, template_folder=pasta_templates, static_folder=pasta_static)

try:
    modelo = joblib.load(caminho_modelo)
    vetorizador = joblib.load(caminho_vetorizador)
    print("IA carregada com sucesso! Servidor pronto.")
except FileNotFoundError:
    print("ERRO: Arquivos .pkl não encontrados. Rode o treinar_modelo.py primeiro!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analisar', methods=['POST'])
def analisar():
    # 1. Pega o que o usuário digitou
    texto_usuario = request.form.get('avaliacao', '')
    
    if not texto_usuario:
        return render_template('index.html', erro="Por favor, digite uma avaliação.")
    
    # 2. Transforma o texto em matemática
    vetor = vetorizador.transform([texto_usuario])
    
    # 3. A IA faz a previsão (Ex: "Positivo" ou "Negativo")
    predicao = modelo.predict(vetor)[0]
    
    # 4. Calcula a % de certeza da IA (pegando o maior valor de probabilidade)
    confianca = max(modelo.predict_proba(vetor)[0]) * 100

    return render_template('index.html', 
                           texto_analisado=texto_usuario, 
                           resultado=predicao, 
                           confianca=round(confianca, 1))

if __name__ == '__main__':
    app.run(debug=True)