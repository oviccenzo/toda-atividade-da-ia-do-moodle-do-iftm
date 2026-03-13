O trabalho 2 foi sobre o dataset titanic predição de sobrevivencia

Este projeto consiste em uma implementação do modelo machine learning para poder prever a sobrevivencia de passageiros de naufrágio do titanic usando o dataset fornecido na aula
Incluiu o pré processamento treinamento do modelo e avaliação e organização do modelo de aprendizado de maquina para o conforme os requisitos da aula de 02 ia iftm ciência da computação

O dataset de titanic: (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked etc).

A estrutura do projeto: 
pythonProject
data/titanic.csv
src/preprocess.py
src-1/train.py
src-2/evalute.py

A tecnologia utilizado que é o:
python3.8
scikit-learn
Pandas
Joblib

O pipelina de ia 
O pre-processamento
Modelo: RandomForestClassifier
Metrica utilizado
matrix da confusão
precision/recall/f1
ROC AUC
cross-validation

O modelo final foi salvo em : 
models/model_titanic.pkl

Como executar o comando no terminal do pycharm do window 10 ou mac os catalina versão 10.15.7:
1) criar um ambiente virtual do pycharm:
python3.8 -m venv venv
 source venv/bin/activate

2) Instalar a depedencia do terminal pycharm
pip install pandas sciki-learn joblib flask

3) treinar o modelo 
python src-1/train.py

4) avaliar o modelo
python src-2/evalute.py

O meu nome completo: viccenzo de oliveira nunes resende
O nome do professor: andré luiz frança baptista
Universidade federal: iftm

