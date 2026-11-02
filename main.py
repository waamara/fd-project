import pandas as pd
from scipy.io import arff

data, meta = arff.loadarff("diabetes.arff")
df = pd.DataFrame(data)

# La colonne texte est lue en bytes (b'tested_positive'), on la décode
df["class"] = df["class"].str.decode("utf-8")

print(df.shape)
print(df.head())
print(df.dtypes)
print(df.isnull().sum())
print(df["class"].value_counts())
print(df.describe())