import pandas as pd

# importa a base
base = pd.read_csv('census.csv')
# divide a base entre os previsores (colunas com os atributos de previsão) e classes (classificação que o conjunto de atributos retorna)
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
                
# transformar variáveis categóricas em discretas, ou seja, transforma textos em valores, para que oalgorítimo entenda
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

# evitar problemas de peso na coluna, transformo cada valor da coluna em uma nova coluna com o onehotencoder
# PARA ESSE MODELO, AO FINAL, DESCOBRIMOS QUE O ONEHOTENCODER REDUZIA A ACERTIVIDADE DO MODELO, POR ISSO ESTÁ COMENTADO
# onehotencoder = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
# previsores = onehotencoder.fit_transform(previsores).toarray()

# transformar variáveis categóricas em discretas para as classes
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

# Escalonamento das variáveis previsoras, dando pesos similares aos valores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisão da base em base de treinamento e base de teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# importando naive bayes do sklearn
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
# fazendo o classificador gerar a tabela de probabilidades baseado nos previsores e classes
classificador.fit(previsores_treinamento, classe_treinamento)
# aplicando previsões na base de teste
previsoes = classificador.predict(previsores_teste)
# após aplicar previsões na base de teste, eu posso comparar abrindo a tabela previsões e comparando com a tabela classe_teste 
# pois essa tabela classe_teste eu sei o resultado. Treinei com tabelas treinamento e testo com tabelas de teste.

# para testar a acurácia do algorítimo podemo usar o metodo abaixo ele vai criar uma variável com a precisão de acerto, messe caso foi 0.938, 94%
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
# outra validação, é a matriz de confusão parar ver qual classe obteve mais acertos
matriz = confusion_matrix(classe_teste, previsoes)

# PARA ESSA BASE, IDENTIFICAMOS QUE APLICAR ESCALONAMENTO EM TODOS PREVISORES NÃO FOI EFICIENTE. PARA CHEGAR NO MELHOR RESULTADO, FOMOS COMENTANDO PARTES
# DO CODIGO ATÉ TERMOS A MELHOR ACURÁCIA
