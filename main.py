# =====================================================================
# TP1 - FD2 - M2 Bioinfo : classification du diabète avec k-NN
# Aymen : préparation des données + application de k-NN
# Ines  : partitionnement + matrice de confusion + mesures
# Ensemble : courbe de précision + meilleur k
# =====================================================================

import pandas as pd                
import numpy as np                  
import matplotlib.pyplot as plt     
from scipy.io import arff           # lecture du fichier .arff

<<<<<<< HEAD
GRAINE = 7                          
=======
GRAINE = 7                          # graine du hasard 

>>>>>>> 5f62e5a162a1b875bd0c2afacd3a3228d35aa8b7


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

    # Générateur aléatoire initialisé avec la graine (résultats reproductibles)
    rng = np.random.default_rng(graine)

    # Mélange aléatoire des numéros de lignes 0..n-1.
    # Chaque indice apparaît exactement une fois => tirage SANS remise,
    # donc aucune ligne ne peut être à la fois dans train et dans test.
    indices = rng.permutation(len(X))

    # Nombre de lignes d'apprentissage : 80 % du total, arrondi à l'entier inférieur
    n_train = int(ratio_train * len(X))

    # Les n_train premiers indices -> apprentissage, les autres -> test
    # (80 % / 20 %, ensembles disjoints)
    tr, te = indices[:n_train], indices[n_train:]

    # On sélectionne les lignes de X et y avec les MÊMES indices,
    # pour que chaque exemple reste associé à sa bonne classe
    return X[tr], y[tr], X[te], y[te]



X_train, y_train, X_test, y_test = partitionner(X, y)
print("Apprentissage :", X_train.shape, "| Test :", X_test.shape)

# ---------- Normalisation (z-score) ----------
# Les statistiques sont calculées sur l'APPRENTISSAGE seulement,
# pour que le test reste réellement "jamais vu" (pas de data leakage)

mu = X_train.mean(axis=0)      # moyenne de chaque colonne (axis=0 : sur les lignes)
sigma = X_train.std(axis=0)    # écart-type de chaque colonne

# Si une colonne est constante, son écart-type vaut 0 : on le remplace par 1
# pour éviter une division par zéro (la colonne devient alors simplement 0)
sigma[sigma == 0] = 1.0

# Formule du z-score : (valeur - moyenne) / écart-type
X_train = (X_train - mu) / sigma

# Le test est normalisé avec les mêmes mu et sigma que le train
# (on ne recalcule JAMAIS de statistiques sur le test)
X_test = (X_test - mu) / sigma

# On standardise les variables pour que celles à grande valeur (ex. insuline) ne dominent pas les autres.


# =====================================================================
# 3) ALGORITHME k-NN (Aymen)
# =====================================================================

def knn_predict(X_train, y_train, X_test, k):
    """Prédit la classe de chaque patient du test avec k voisins."""
    predictions = []
    for x in X_test:
        # distance euclidienne à tous les patients d'apprentissage
        distances = np.sqrt(((X_train - x) ** 2).sum(axis=1))
         # np.argsort trie les distances et renvoie les INDICES (pas les valeurs) ;
        # [:k] garde les k plus petits = les k plus proches voisins
        voisins = np.argsort(distances)[:k]
        # vote majoritaire (égalité -> classe 0)
         # np.bincount compte les occurrences de chaque classe 
        # .argmax() renvoie la classe la plus fréquente
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
    # (y_vrai == 1) donne un tableau de booléens ; & fait un "ET" élément par élément
    # .sum() compte les True, donc le nombre de cas qui vérifient les deux conditions
    TP = int(((y_vrai == 1) & (y_pred == 1)).sum())  # vraiment 1 ET prédit 1
    TN = int(((y_vrai == 0) & (y_pred == 0)).sum())  # vraiment 0 ET prédit 0
    FP = int(((y_vrai == 0) & (y_pred == 1)).sum())  # vraiment 0 mais prédit 1
    FN = int(((y_vrai == 1) & (y_pred == 0)).sum())  # vraiment 1 mais prédit 0
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