# =====================================================================
# TP1 - FD2 - M2 Bioinfo : classification du diabète avec k-NN
# Aymen : préparation des données + application de k-NN
# Ines  : partitionnement + matrice de confusion + mesures
# Ensemble : courbe de précision + meilleur k
# =====================================================================

import pandas as pd                 # tableaux
import numpy as np                  # calculs
import matplotlib.pyplot as plt     # dessin de la courbe
from scipy.io import arff           # lecture du fichier .arff

GRAINE = 7                          # graine du hasard (à citer dans le rapport)


# =====================================================================
# 1) PRÉPARATION DES DONNÉES (Aymen)
# =====================================================================

# Chargement du benchmark Diabète (Pima)
data, meta = arff.loadarff("diabetes.arff")
df = pd.DataFrame(data)
df["class"] = df["class"].str.decode("utf-8")   # bytes -> texte normal

# Les 0 sont impossibles pour ces colonnes : ce sont des valeurs manquantes cachées
cols_zero = ["plas", "pres", "skin", "insu", "mass"]
print("Nombre de 0 impossibles par colonne :")
print((df[cols_zero] == 0).sum())

# 0 -> NaN, puis NaN -> médiane de la colonne
df[cols_zero] = df[cols_zero].replace(0, float("nan"))
df[cols_zero] = df[cols_zero].fillna(df[cols_zero].median())

# Classe : tested_negative -> 0 (sain), tested_positive -> 1 (diabétique)
df["class"] = df["class"].map({"tested_negative": 0, "tested_positive": 1})

print("\nValeurs manquantes restantes :", int(df.isnull().sum().sum()))
print("Patients par classe :")
print(df["class"].value_counts())

# X = les 8 attributs, y = la classe à prédire
X = df.drop(columns=["class"]).values
y = df["class"].values
print("\nX :", X.shape, "| y :", y.shape)


# =====================================================================
# 2) PARTITIONNEMENT 80% / 20% (Ines)
# =====================================================================

def partitionner(X, y, ratio_train=0.8, graine=GRAINE):
    """Partition aléatoire et SANS remise en apprentissage / test."""
    rng = np.random.default_rng(graine)
    indices = rng.permutation(len(X))         # chaque patient apparaît une seule fois
    n_train = int(ratio_train * len(X))
    tr, te = indices[:n_train], indices[n_train:]
    return X[tr], y[tr], X[te], y[te]


X_train, y_train, X_test, y_test = partitionner(X, y)
print("Apprentissage :", X_train.shape, "| Test :", X_test.shape)

# Normalisation (z-score) : statistiques de l'APPRENTISSAGE seulement
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
sigma[sigma == 0] = 1.0                       # sécurité : évite une division par 0
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma


# =====================================================================
# 3) ALGORITHME k-NN (Aymen)
# =====================================================================

def knn_predict(X_train, y_train, X_test, k):
    """Prédit la classe de chaque patient du test avec k voisins."""
    predictions = []
    for x in X_test:
        # distance euclidienne à tous les patients d'apprentissage
        distances = np.sqrt(((X_train - x) ** 2).sum(axis=1))
        # numéros des k plus proches voisins
        voisins = np.argsort(distances)[:k]
        # vote majoritaire (égalité -> classe 0)
        vote = np.bincount(y_train[voisins]).argmax()
        predictions.append(vote)
    return np.array(predictions)


# k-NN pour k = 1, 2, ..., 10
preds = {}
for k in range(1, 11):
    preds[k] = knn_predict(X_train, y_train, X_test, k)


# =====================================================================
# 4) ÉVALUATION : MATRICE DE CONFUSION + MESURES (Ines)
# =====================================================================

def matrice_confusion(y_vrai, y_pred):
    """Retourne TP, TN, FP, FN (classe positive = 1 = diabétique)."""
    TP = int(((y_vrai == 1) & (y_pred == 1)).sum())
    TN = int(((y_vrai == 0) & (y_pred == 0)).sum())
    FP = int(((y_vrai == 0) & (y_pred == 1)).sum())
    FN = int(((y_vrai == 1) & (y_pred == 0)).sum())
    return TP, TN, FP, FN


def mesures(TP, TN, FP, FN):
    """Précision, rappel et F-mesure (0 si division par 0)."""
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rappel = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if (precision + rappel) > 0:
        f_mesure = 2 * precision * rappel / (precision + rappel)
    else:
        f_mesure = 0.0
    return precision, rappel, f_mesure


def afficher_matrice(k, TP, TN, FP, FN):
    print("\n===== Matrice de confusion pour k =", k, "=====")
    print("                       Prédit: 1 (diabétique)   Prédit: 0 (sain)")
    print("Réel: 1 (diabétique)   TP =", TP, "                FN =", FN)
    print("Réel: 0 (sain)         FP =", FP, "                TN =", TN)


resultats = []          # une ligne par k : (k, TP, TN, FP, FN, précision, rappel, F)
for k in range(1, 11):
    TP, TN, FP, FN = matrice_confusion(y_test, preds[k])
    precision, rappel, f_mesure = mesures(TP, TN, FP, FN)
    resultats.append((k, TP, TN, FP, FN, precision, rappel, f_mesure))
    afficher_matrice(k, TP, TN, FP, FN)

# Tableau récapitulatif pour k = 1 à 10
print("\n" + "=" * 62)
print(f"{'k':>3} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} "
      f"{'Précision':>10} {'Rappel':>8} {'F-mesure':>9}")
for (k, TP, TN, FP, FN, p, r, f) in resultats:
    print(f"{k:>3} {TP:>4} {TN:>4} {FP:>4} {FN:>4} {p:>10.3f} {r:>8.3f} {f:>9.3f}")


# =====================================================================
# 5) COURBE DE PRÉCISION + MEILLEUR k (Aymen et Ines)
# =====================================================================

ks = [r[0] for r in resultats]
precisions = [r[5] for r in resultats]

meilleur = max(resultats, key=lambda r: r[5])    # plus haute précision
print("\nMeilleur k =", meilleur[0], "| précision =", round(meilleur[5], 3))

plt.figure(figsize=(7, 4))
plt.plot(ks, precisions, marker="o")
plt.plot(meilleur[0], meilleur[5], "r*", markersize=15,
         label=f"Meilleur k = {meilleur[0]}")
plt.xlabel("k (nombre de voisins)")
plt.ylabel("Précision")
plt.title("Précision du k-NN selon k")
plt.xticks(ks)
plt.grid(True)
plt.legend()
plt.savefig("courbe_precision.png", dpi=150, bbox_inches="tight")
plt.show()