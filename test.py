
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# =====================================================
# 1. CHARGEMENT DU DATASET
# =====================================================

df = pd.read_csv('data/diabetes_preprocessed.csv')


# =====================================================
# 2. SEPARATION DES CARACTERISTIQUES ET DE LA CLASSE
# =====================================================

X = df.drop('class', axis=1)
y = df['class']


# =====================================================
# 3. PARTITIONNEMENT DU DATASET (80% / 20%)
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y
)

# Affichage des dimensions
print("Taille du dataset :", df.shape)
print("X_train :", X_train.shape)
print("X_test :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test :", y_test.shape)


# =====================================================
# 4. CONVERSION EN TABLEAUX NUMPY
# =====================================================

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# =====================================================
# 5. CALCUL DE LA DISTANCE EUCLIDIENNE
# =====================================================

def dist_euc(x1, x2):

    return np.sqrt(np.sum((x1 - x2) ** 2))


# =====================================================
# 6. CALCUL DES DISTANCES
# =====================================================

def calculer_distances(point_test, X_train):

    distances = []

    for point_train in X_train:

        d = dist_euc(point_test, point_train)

        distances.append(d)

    return distances


# =====================================================
# 7. RECHERCHE DES K PLUS PROCHES VOISINS
# =====================================================

def trouver_voisins(distances, k):

    indices_tries = np.argsort(distances)

    voisins = indices_tries[:k]

    return voisins


# =====================================================
# 8. VOTE MAJORITAIRE
# =====================================================

def vote_majoritaire(voisins, y_train):

    classes_voisins = y_train[voisins]

    classes, comptes = np.unique(
        classes_voisins,
        return_counts=True
    )

    classe_predite = classes[np.argmax(comptes)]

    return classe_predite


# =====================================================
# 9. FONCTION PRINCIPALE KNN
# =====================================================

def knn(point_test, X_train, y_train, k):

    distances = calculer_distances(point_test, X_train)

    voisins = trouver_voisins(distances, k)

    prediction = vote_majoritaire(voisins, y_train)

    return prediction


# =====================================================
# 10. APPLICATION DE KNN POUR K = 1 A 10
# =====================================================

for k in range(1, 11):

    predictions = []

    # Parcourir toutes les observations de test
    for point_test in X_test:

        # Prédire la classe
        prediction = knn(point_test, X_train, y_train, k)

        # Stocker la prédiction
        predictions.append(prediction)

    print("\nPour k =", k)
    print("Nombre de prédictions :", len(predictions))

    # Afficher quelques prédictions
    print("Premières prédictions :", predictions[:5])
