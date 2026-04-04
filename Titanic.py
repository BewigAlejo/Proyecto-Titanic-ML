import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Lectura de los datos
df = pd.read_csv("Data/train.csv")

#Limpeza de dados
df_modelo = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].copy()
print(df_modelo.isnull().sum())

# limpiar bien
df_modelo["Age"] = df_modelo["Age"].fillna(df_modelo["Age"].median())
df_modelo["Fare"] = df_modelo["Fare"].fillna(df_modelo["Fare"].median())
df_modelo["Embarked"] = df_modelo["Embarked"].fillna(df_modelo["Embarked"].mode()[0])

# verificar
print(df_modelo.isnull().sum())

df_modelo = pd.get_dummies(df_modelo, columns=["Sex", "Embarked"], drop_first=True)

#División de los datos en características (X) y variable objetivo (y)
X = df_modelo.drop("Survived", axis=1)
y = df_modelo["Survived"]

#División de los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entrenamiento del modelo de regresión logística
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#ver la precisión del modelo
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))