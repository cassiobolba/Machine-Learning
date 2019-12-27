# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:20:33 2019

@author: cassio.bolba
"""

import pandas as pd
base = pd.read_csv('census.csv')

# separação do previsores e classes
# quando tenrat ver essa base, ele dara erro, pois até o momento
#não aceita objetos tipo object
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

# labelencoder é usado para numerar variáveis categóricas
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# crira a variável que chama a função LabelEncoder
labelencoder_previsores = LabelEncoder()
# variável que armazena variável cat. da coluna 1 transformada para numero
#labels = labelencoder_previsores.fit_transform(previsores[:,1])
# trecho acima cometado, pq foi teste, agora usamo aquele resultado pra
# trocar os valores categóricos pelos respectivos numéricos
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelencoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelencoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelencoder_previsores.fit_transform(previsores[:,13])

# usando onehotencoder para todos os valores categóricos, até o ordinal
onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()

# parar transformar a classe, como temos apenas 2, usamos o label encoder
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

# escalonamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# dividir as bases
from sklearn.cross_validation import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,classe, test_size = 0.25, random_state = 0)
