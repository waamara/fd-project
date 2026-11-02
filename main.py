import pandas as pd
from scipy.io import arff

data, meta = arff.loadarff("diabetes.arff")
df = pd.DataFrame(data)
df["class"] = df["class"].str.decode("utf-8")


# 1) Compter les zéros impossibles
cols_zero = ["plas", "pres", "skin", "insu", "mass"]
print((df[cols_zero] == 0).sum())

# 2) Remplacer ces zéros par NaN (= valeur manquante)
df[cols_zero] = df[cols_zero].replace(0, float("nan"))

# 3) Remplacer les NaN par la médiane de chaque colonne
df[cols_zero] = df[cols_zero].fillna(df[cols_zero].median())

# 4) Transformer la classe en 0 / 1
df["class"] = df["class"].map({"tested_negative": 0, "tested_positive": 1})

print(df.isnull().sum())
print(df["class"].value_counts())
print(df.describe())


print(df.shape)
print(df.head()) 

# Séparer les attributs (X) et la classe à prédire (y)
X = df.drop(columns=["class"]).values
y = df["class"].values

print(X.shape)
print(y.shape)