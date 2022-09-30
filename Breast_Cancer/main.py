#imports
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#read dataset
df = pd.read_csv('breast-cancer-data.csv')
print(df.head())
print(df.dtypes)

# Cleaning and Preparing the Data

#Binary values
df['class'] = np.where((df['class'] == 'no-recurrence-events'), 0, 1)
df['breast'] = np.where((df['breast'] == 'left'), 1, 2)
df['irradiat'] = np.where((df['irradiat'] == 'no'), 0, 1)
df['node-caps'] = np.where((df['node-caps'] == 'no'), 0, 1)

# testar a função np.select(condicão, novo_valor)
menopause_target = [(df['menopause'] == "premeno"),(df['menopause'] == 'ge40'),(df['menopause'] == 'it40')]
menopause_values = [1, 2, 3]
df['menopause'] = np.select(menopause_target, menopause_values)

# para poupar tempo de escrita, sacrifico desempenho
quadrante = {'left_up':1, 'left_low': 2, 'right_up':3, 'right_low':4, 'central':5} 
df = df.replace({'breast-quad': quadrante})
df['breast-quad'] = df['breast-quad'].apply(pd.to_numeric, downcast='float', errors='coerce')
df[df.isnull().any(axis = 1)]
df = df.dropna()

age = {'20-29':24.5, '30-39':34.5,'40-49':44.5,'50-59':54.5, '60-69':64.5,'70-79':74.5,'80-89':84.5,'90-99':94.5}
df = df.replace({'age': age})

nodes = {'0-2':1, '3-5':4,'6-8':7,'9-11':10, '12-14':13,'15-17':16,'18-20':19,'21-23':22,'24-26':25,'27-29':28,'30-32':31,'33-35':34,
        '36-38':37,'39':39}
df = df.replace({'inv-nodes': nodes})

tumor = {'0-4':2, '5-9':7,'10-14':12,'15-19':17, '20-24':22,'25-29':27,'30-34':32,'35-39':37,'40-44':42,'45-49':47,'50-54':52}
df = df.replace({'tumor-size': tumor})

print(df.head())
print("Shape do dataset", df.shape)

# analyse correlation between features
print(df.corr())
f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)

#De acordo com a matriz de correlação, as features (tumor-size, inv-nodes, node-caps, deg-malig, irradiat)
#apresentam os melhores índices de correlação com a class. Logo, essas serão as features utilizadas no treinamento.

x = df[['tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'irradiat']].copy()
Y = df[['class']].copy()

print(x.head())
print(Y.head())

# Train/test Split
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.20, random_state = 171)
print("Shape train", x_train.shape)
print("Shape test", x_test.shape)

# Decision tree classifier
decisionTree_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=4)
decisionTree_classifier.fit(x_train, Y_train)

Y_pred_tree = decisionTree_classifier.predict(x_test)

print('Accuracy Score: ', accuracy_score(Y_test, Y_pred_tree))
print('Confusion Matrix: ', confusion_matrix(Y_test, Y_pred_tree))

mpl.rcParams['figure.dpi'] = 200

plt.figure()
plot_tree(decisionTree_classifier, class_names=True, filled=True)
plt.show()

# Gaussian Naive Bayes
gaussian_classifier = GaussianNB()
gaussian_classifier.fit(x_train, Y_train)

Y_pred_bayes = gaussian_classifier.predict(x_test)

print("Accuracy Score for bayes method: ", accuracy_score(Y_test, Y_pred_bayes))
print("Confusion matrix for bayes method: ", confusion_matrix(Y_test, Y_pred_bayes))

