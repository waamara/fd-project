import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# =====================================================
# 1. CHARGEMENT DU DATASET
# =====================================================

# Lecture du fichier CSV contenant le dataset déjà prétraité
df = pd.read_csv('data/diabetes_preprocessed.csv')


# =====================================================
# 2. SEPARATION DES CARACTERISTIQUES ET DE LA CLASSE
# =====================================================

# X : les variables explicatives (toutes les colonnes sauf 'class')
X = df.drop('class', axis=1)

# y : la variable à prédire (diabétique ou non)
y = df['class']


# =====================================================
# 3. PARTITIONNEMENT DU DATASET (80% / 20%)
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,    # 20% pour le test, donc 80% pour l'apprentissage
    random_state=42,  # graine fixe : on obtient toujours la même partition
    shuffle=True,     # mélange aléatoire des lignes avant la découpe
    stratify=y        # garde la même proportion de classes dans train et test
)

# Affichage des dimensions pour vérifier la partition
print("Taille du dataset :", df.shape)
print("X_train :", X_train.shape)
print("X_test :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test :", y_test.shape)


# =====================================================
# 4. CONVERSION EN TABLEAUX NUMPY
# =====================================================

# On passe de pandas à numpy pour faire les calculs de distances
# plus simplement (indexation par position, opérations vectorisées)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


# =====================================================
# 5. CALCUL DE LA DISTANCE EUCLIDIENNE
# =====================================================

def dist_euc(x1, x2):
    # Distance euclidienne entre deux points :
    # racine de la somme des carrés des différences, variable par variable
    return np.sqrt(np.sum((x1 - x2) ** 2))


# =====================================================
# 6. CALCUL DES DISTANCES
# =====================================================

def calculer_distances(point_test, X_train):
    # Liste qui contiendra la distance entre le point de test
    # et chacun des points d'apprentissage
    distances = []

    # Parcourir tous les points de l'ensemble d'apprentissage
    for point_train in X_train:

        # Distance entre le point de test et ce point d'apprentissage
        d = dist_euc(point_test, point_train)

        distances.append(d)

    return distances


# =====================================================
# 7. RECHERCHE DES K PLUS PROCHES VOISINS
# =====================================================

def trouver_voisins(distances, k):
    # argsort renvoie les indices des points classés
    # de la plus petite distance à la plus grande
    indices_tries = np.argsort(distances)

    # On garde les k premiers : ce sont les k plus proches voisins
    voisins = indices_tries[:k]

    return voisins


# =====================================================
# 8. VOTE MAJORITAIRE
# =====================================================

def vote_majoritaire(voisins, y_train):
    # Classes des k voisins (on récupère leurs étiquettes dans y_train)
    classes_voisins = y_train[voisins]

    # Compter combien de fois chaque classe apparaît parmi les voisins
    classes, comptes = np.unique(
        classes_voisins,
        return_counts=True
    )

    # La classe prédite est celle qui a le plus de votes
    # (en cas d'égalité, argmax prend la première dans l'ordre alphabétique)
    classe_predite = classes[np.argmax(comptes)]

    return classe_predite


# =====================================================
# 9. FONCTION PRINCIPALE KNN
# =====================================================

def knn(point_test, X_train, y_train, k):
    # Étape 1 : distance entre le point de test et tous les points d'apprentissage
    distances = calculer_distances(point_test, X_train)

    # Étape 2 : indices des k points les plus proches
    voisins = trouver_voisins(distances, k)

    # Étape 3 : la classe majoritaire parmi ces voisins est la prédiction
    prediction = vote_majoritaire(voisins, y_train)

    return prediction


# =====================================================
# 10. APPLICATION DE KNN POUR K = 1 A 10
# =====================================================

# Répéter tout le processus pour chaque valeur de k de 1 à 10
for k in range(1, 11):

    # Liste des prédictions pour cette valeur de k
    predictions = []

    # Parcourir toutes les observations de test
    for point_test in X_test:

        # Prédire la classe en utilisant l'ensemble d'apprentissage comme référence
        prediction = knn(point_test, X_train, y_train, k)

        # Stocker la prédiction
        predictions.append(prediction)

    # Affichage des résultats pour ce k
    print("\nPour k =", k)
    print("Nombre de prédictions :", len(predictions))

    # Afficher quelques prédictions seulement (les 5 premières)
    print("Premières prédictions :", predictions[:5])
    
    