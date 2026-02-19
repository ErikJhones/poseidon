from tabicl import TabICLClassifier

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Carregar o dataset
# Certifique-se de que o arquivo 'titanic.csv' esteja na mesma pasta do script
df = pd.read_csv('titanic.csv')

# 2. Definir as variáveis (X = características, y = alvo)
# No Titanic, o objetivo geralmente é prever a coluna 'Survived'
X = df.drop('Survived', axis=1) 
y = df['Survived']

# 3. Dividir em Treino e Teste
# test_size=0.2 significa que 20% dos dados serão para teste e 80% para treino
# random_state garante que a divisão seja a mesma toda vez que você rodar o código
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = TabICLClassifier()
clf.fit(X_train, y_train)  # downloads checkpoint on first use, otherwise cheap
print(clf.predict(X_test))  # in-context learning happens here

# reg = TabICLRegressor()
# reg.fit(X_train, y_train)
# reg.predict(X_test)