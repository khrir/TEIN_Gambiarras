#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#read dataset
df = pd.read_csv('breast-cancer-data.csv')
# print(df.head())
print(df.dtypes)

#pre-processing
 

# analyse correlation between features