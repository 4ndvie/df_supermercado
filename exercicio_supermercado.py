#%%Criacao de um dataset aleatório para realizar o exercício

import pandas as pd
import numpy as np

# Definir número de funcionários
num_funcionarios = 200

# Criar dataset simulado
np.random.seed(42)
df = pd.DataFrame({
    "ID_Funcionario": np.arange(1, num_funcionarios + 1),
    "Idade": np.random.randint(22, 60, num_funcionarios),
    "Tempo_na_Empresa": np.random.randint(1, 20, num_funcionarios),
    "Salario": np.random.randint(3000, 15000, num_funcionarios),
    "Satisfacao_no_Trabalho": np.random.randint(1, 6, num_funcionarios),
    "Horas_extras": np.random.choice(["Sim", "Não"], num_funcionarios, p=[0.3, 0.7]),
    "Saiu_da_Empresa": np.random.choice(["Sim", "Não"], num_funcionarios, p=[0.2, 0.8])
})


#%% Passo 2: Fazer uma análise exploratória para entender se funcionários com menor satisfação tendem a sair mais
import seaborn as sns
import matplotlib.pyplot as plt

# Contar quantos saíram por nível de satisfação
sns.countplot(data=df, x="Satisfacao_no_Trabalho", hue="Saiu_da_Empresa")
plt.title("Satisfação no Trabalho x Rotatividade")
plt.show()

#%% Passo 3:  Identificar se o tempo de empresa influencia na saída de funcionários

sns.boxplot(data=df, x="Saiu_da_Empresa", y="Tempo_na_Empresa")
plt.title("Tempo na Empresa x Rotatividade")
plt.show()

#%% Identificar se o salário influência na saída de funcionários

sns.boxplot(data=df, x="Saiu_da_Empresa", y="Salario")
plt.title("Salário x Rotatividade")
plt.show()


#%% Passo 3: Testar a correlação de Person entre satisfação no trabalho e rotatividade para definir estatisticamente 
#%% se os valores estão relacionados
correlacao = df_encoded["Satisfacao_no_Trabalho"].corr(df_encoded["Saiu_da_Empresa"])
print("Correlação entre Satisfação e Rotatividade:", correlacao)


#%%Modelo preditivo baseado em classificação

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Converter variáveis categóricas para numéricas
df_encoded = df.copy()
df_encoded["Horas_extras"] = df_encoded["Horas_extras"].map({"Sim": 1, "Não": 0})
df_encoded["Saiu_da_Empresa"] = df_encoded["Saiu_da_Empresa"].map({"Sim": 1, "Não": 0})

# Definir variáveis preditoras e alvo
X = df_encoded.drop(columns=["ID_Funcionario", "Saiu_da_Empresa"])
y = df_encoded["Saiu_da_Empresa"]

# Separar dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar modelo de classificação
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliar modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

df_encoded['Previsao_Modelo'] = modelo.predict(X)

#%%somando resultados 
df_encoded.query('Previsao_Modelo == 1 & Saiu_da_Empresa == 1')
df_soma = df_encoded.query('Previsao_Modelo == 1 & Saiu_da_Empresa == 0')
df_encoded.query('Previsao_Modelo == 0 & Saiu_da_Empresa == 0')