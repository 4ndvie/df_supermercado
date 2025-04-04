#%%Criacao de um dataset aleatório para realizar o exercício

import pandas as pd
import numpy as np

# Definir número de funcionários
num_clientes = 200

# Criar dataset simulado
np.random.seed(42)
df = pd.DataFrame({
    "ID_Cliente": np.arange(1, num_clientes + 1),
    "Idade": np.random.randint(22, 60, num_clientes),
    "Renda_Mensal": np.random.randint(3000, 20000, num_clientes),
    "Historico_Credito": np.random.choice(["Ruim", "Medio", "Bom"], num_clientes, p=[0.2, 0.2, 0.6]),
    "Dividas_Atuais": np.random.randint(1000, 20000, num_clientes),
    "Numero_Cartoes": np.random.randint(1, 10, num_clientes), 
    "Aprovado": np.random.choice(["Sim", "Não"], num_clientes, p=[0.7, 0.3])
})


#%% Passo 2: Fazer uma análise exploratória entre dividas atuais e número de cartões
import seaborn as sns
import matplotlib.pyplot as plt

# Contar quantos clientes tem dividas de acordo com o número de cartões
sns.countplot(data=df, x="Dividas_Atuais", hue="Numero_Cartoes")
plt.title("Dividas Atuais x Numero de Cartoes")
plt.show()

#%% Passo 3: Fazer uma análise exploratória utilizando scatterplot

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Dividas_Atuais"], y=df["Numero_Cartoes"], alpha=0.5)
plt.title("Relação entre Dívidas Atuais e Número de Cartões")
plt.xlabel("Dívidas Atuais")
plt.ylabel("Número de Cartões")
plt.show()

#%% Passo 3.1: Correlação entre dividas atuais e numero de cartões
    
correlacao = df["Renda_Mensal"].corr(df["Aprovado"])
print(f"Correlação entre Renda mensal e status de crédito: {correlacao:.2f}")


#%% Passo 3.2: Fazer uma análise exploratória utilizando scatterplot

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df_encoded["Renda_Mensal"], y=df_encoded["Historico_Credito"], alpha=0.5)
plt.title("Relação entre renda mensal e histórico de crédito")
plt.xlabel("Renda Mensal")
plt.ylabel("Histórico de Crédito")
plt.show()

#%% 3.2.1 Convertendo variáveis categóricas 

df_encoded = df.copy()
df_encoded["Historico_Credito"] = df_encoded["Historico_Credito"].map({"Ruim": 0, "Medio": 1, "Bom": 2})
df_encoded["Aprovado"] = df_encoded["Aprovado"].map({"Sim": 1, "Não": 0})

#%% 3.2.2 Correlações 
correlacao = df_encoded["Renda_Mensal"].corr(df_encoded["Historico_Credito"])
print(f"Correlação entre renda mensal e histórico de crédito: {correlacao:.2f}")


#%%Modelo preditivo baseado em classificação

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Definir variáveis preditoras e alvo
X = df_encoded.drop(columns=["ID_Cliente", "Aprovado"])
y = df_encoded["Aprovado"]

# Separar dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Criar modelo de classificação
modelo = RandomForestClassifier(n_estimators= 100, max_depth=30, bootstrap=True, random_state=42)
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliar modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

df_encoded['Previsao_Modelo'] = modelo.predict(X)

#%%somando resultados 
df_encoded.query('Previsao_Modelo == 1 & Aprovado == 1')
df_encoded.query('Previsao_Modelo == 1 & Aprovado == 0')
df_encoded.query('Previsao_Modelo == 0 & Aprovado == 1')
df_encoded.query('Previsao_Modelo == 0 & Aprovado == 0')


#%% Analise entre tamanho do teste e acurácia

import matplotlib.pyplot as plt

test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
accuracies = []

for size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)
    
    modelo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.plot(test_sizes, accuracies, marker='o')
plt.xlabel("Proporção do Teste")
plt.ylabel("Acurácia")
plt.title("Impacto do Tamanho do Conjunto de Teste na Acurácia")
plt.show()
