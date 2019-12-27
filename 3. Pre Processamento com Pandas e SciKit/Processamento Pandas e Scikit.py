# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 22:00:11 2019

@author: cassio.bolba
"""

import pandas as pd
base = pd.read_csv('credit_data.csv')
base.describe()

# localizar na base, na coluna age, campos menores que 0
base.loc[base['age']<0]
base.loc[base['clientid']==(16) ]


# Apagar uma coluna (não é muito usado, pois apaga a coluna inteira)
base.drop('age',1,inplace = True)

# Apagar somentes os registros com problema
base.drop(base[base.age < 0].index, inplace= True)

# Preencher os valore manualmente
# Preencher os valore com a média das idades
base['age'].mean() #considera os valores negativos
base['age'][base.age >0].mean()
base.loc[base.age < 0, 'age'] = 40.92

# achar nulos - método 1
pd.isnull(base['age'])
# achar nulos - método 2
base.loc[pd.isnull(base['age'])]

#criando variaveis previsoras
#iloc para fazer a divisão das bases
#iloc[linhas , colunas ]
previsores = base.iloc[:, 1:4].values

#criando classes para classificação
classe = base.iloc[:,4].values

# trazendo pacote que corrige os nulos Imputer
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0 )
imputer = imputer.fit(previsores[:,0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

#Escalonamento de dados:
# é dizer que colunas tem o mesmo peso para os cálculos dos algorítimos
# KNN por exemplo, considera a coluna que tem maior diferença
#como a coluna mais importa, dando mais peso a ela. Mas se quero
#que tenha o mesmo peso, devo escalonar.
#KNN usa distÂncia euclidiana para os cálculos

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
