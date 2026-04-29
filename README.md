# PCA Parallélisée pour la Reconnaissance d'Activité Humaine (HAR)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Implémentation from scratch de l'Analyse en Composantes Principales (ACP) avec parallélisation du calcul de la matrice de covariance, appliquée au dataset *Human Activity Recognition with Smartphones* (UCI). Visualisation de la séparabilité des classes et classification SVM.**

---

## 📋 Table des matières

- [Aperçu du projet](#aperçu-du-projet)
- [Dataset](#dataset)
- [Méthodologie](#méthodologie)
  - [ACP from scratch](#acp-from-scratch)
  - [Parallélisation du calcul de covariance](#parallélisation-du-calcul-de-covariance)
  - [Classification](#classification)
- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats attendus](#résultats-attendus)
- [Visualisations](#visualisations)
- [Architecture technique](#architecture-technique)
- [Troubleshooting](#troubleshooting)
- [Crédits & Références](#crédits--références)

---

## 🎯 Aperçu du projet

Ce projet démontre comment **réduire drastiquement la dimension d'un problème de classification** (561 attributs de capteurs smartphones) tout en conservant l'essentiel de la variance, grâce à une ACP construite sans bibliothèque dédiée. Le calcul de la matrice de covariance, étape la plus coûteuse, est **parallélisé avec `multiprocessing`** pour exploiter les architectures multicœurs.

Les objectifs professionnels :
- Maîtrise du **calcul scientifique** et du **parallélisme en Python**.
- **Visualisation de la séparabilité** des classes après projection.
- **Évaluation comparative** (SVM linéaire) entre données brutes et réduites.

---

## 📊 Dataset

Nous utilisons le dataset **Human Activity Recognition Using Smartphones** (UCI Machine Learning Repository), mis à disposition sur [Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones).

- **Participants** : 30 volontaires (19-48 ans)
- **Activités** : 6 classes
  - WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS
  - SITTING, STANDING, LAYING
- **Capteurs** : accéléromètre et gyroscope (50 Hz)
- **Taille** : 7 352 exemples d'entraînement, 2 947 de test
- **Features** : 561 attributs temporels et fréquentiels

---

## 🔧 Méthodologie

### ACP from scratch

1. **Standardisation** : centrage-réduction de chaque feature.
2. **Matrice de covariance** : \( C = \frac{1}{n-1} X^T X \)  
   où \( X \) est la matrice centrée-réduite.
3. **Décomposition spectrale** : `np.linalg.eigh` (matrice symétrique).
4. **Projection** : \( X_{proj} = X V_k \) avec \( V_k \) les \( k \) premiers vecteurs propres (triés par valeur propre décroissante).

### Parallélisation du calcul de covariance

- La matrice \( X \) est découpée en **blocs de lignes**.
- Chaque bloc est envoyé à un processus du pool `multiprocessing.Pool` pour calculer \( X_{bloc}^T X_{bloc} \) (produit local).
- Les résultats partiels sont **sommés sur le processus maître**, puis divisés par \( n-1 \).
- La taille des blocs s'adapte automatiquement au nombre de cœurs (`cpu_count()`).

### Classification

Un **SVM linéaire** (`LinearSVC`, sklearn) est entraîné sur :
- Les 561 features brutes
- Les 50 premières composantes principales (captant ~95 % de variance)

Métriques mesurées : **accuracy**, **temps d'entraînement**, **rapport de classification**.

---

## 📁 Structure du projet

```
HAR_PCA_Project/
├── data/
│   ├── train.csv                # Données d'entraînement
│   └── test.csv                 # Données de test
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Chargement et prétraitement
│   ├── pca_parallel.py          # Implémentation parallèle de l'ACP
│   ├── classifier.py            # SVM et évaluation
│   └── visualization.py         # Graphiques 2D et variance expliquée
├── main.py                      # Script principal
├── requirements.txt
└── README.md                    # Ce fichier
```

---

## ⚙️ Installation

**Prérequis** : Python ≥ 3.8

1. Cloner ce dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/HAR_PCA_Project.git
   cd HAR_PCA_Project
   ```
2. Créer un environnement virtuel (recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```
3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
4. Placer les fichiers `train.csv` et `test.csv` dans le dossier `data/` (téléchargeables depuis [Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)).

---

## 🚀 Utilisation

Lancez le script principal depuis VS Code (ou tout terminal) :

```bash
python main.py
```

Le programme :
- Charge et standardise les données.
- Parallélise le calcul de la covariance (utilisation de tous les cœurs disponibles).
- Affiche le temps de calcul et la variance expliquée.
- Génère deux visualisations :
  1. Projection 2D des données d'entraînement sur les deux premières composantes principales.
  2. Courbe de variance expliquée par composante.
- Entraîne un SVM sur les données brutes et sur 50 composantes PCA, puis compare les performances.

---

## 📈 Résultats attendus

### Variance expliquée

| Nombre de composantes | Variance individuelle (%) | Variance cumulée (%) |
|-----------------------|---------------------------|----------------------|
| 1                     | ~25 %                     | ~25 %                |
| 10                    | ~3 % (moyenne)            | ~80 %                |
| 50                    | <1 % (moyenne)            | **>95 %**            |
| 100                   | <0.5 % (moyenne)          | ~99 %                |

### Séparabilité des classes (projection 2D)

Les activités statiques (`LAYING`, `SITTING`, `STANDING`) apparaissent très bien séparées sur les deux premières composantes principales, tandis que les activités dynamiques (`WALKING`, `UPSTAIRS`, `DOWNSTAIRS`) se chevauchent partiellement. Ceci est cohérent avec la littérature sur ce dataset car les capteurs détectent mieux les différences statiques (position du téléphone).

### Performances de classification (SVM linéaire)

| Modèle               | Accuracy (test) | Temps entraînement | Nombre de features | Réduction |
|----------------------|-----------------|-------------------|--------------------|-----------|
| SVM (données brutes) | ~96.2 %          | ~2.1 s            | 561                | 1x        |
| SVM (PCA 50)         | ~95.8 %          | ~0.4 s            | 50                 | **11x**   |

**Observations clés :**
- La réduction de dimension par un facteur 11 n'entraîne qu'une perte de précision marginale (~0.4 %).
- Le temps d'apprentissage est divisé par **plus de 5**, ce qui est crucial pour les applications temps réel.
- L'approche parallèle du calcul de covariance exploite 100% des cœurs CPU disponibles.

---

## 📊 Architecture technique

### Workflow d'exécution

```
1. Chargement des données (train.csv, test.csv)
   ↓
2. Standardisation (centrage/réduction)
   ↓
3. Parallélisation du calcul de covariance
   - Division en blocs → Pool de workers → Somme des résultats partiels
   ↓
4. Décomposition spectrale (np.linalg.eigh)
   ↓
5. Projection des données brutes sur les composantes principales
   ↓
6. Visualisation 2D et courbe de variance
   ↓
7. Classification SVM (brute vs PCA) + Comparaison
```

### Complexité computationnelle

- **Calcul de covariance** : O(n × p²) où n = nb d'échantillons, p = nb de features
  - Avec parallélisation : O(n × p² / num_cpus) en pratique
- **Décomposition spectrale** : O(p³) (standard numpy)
- **Projection** : O(n × p × k) où k = nb de composantes retenues
- **SVM** : O(n²) (cas linéaire sur données denses)

---

## 📊 Visualisations

Les graphiques générés à chaque exécution :

1. **`pca_2d.png`** : Scatter plot interactif des deux premières composantes principales
   - Axe X : 1ère composante principale
   - Axe Y : 2e composante principale
   - Couleurs : différentes activités (6 classes)
   
2. **Courbe de variance expliquée** : Affichée en temps réel
   - Barre d'erreur ou histogramme montrant variance individuelle
   - Courbe cumulative

### Exemple de sortie console

```
Dimensions train: (7352, 561), test: (2947, 561)
[PCA] Covariance calculée en 0.145 s sur 8 cœurs, 23 blocs.
Transformation totale ACP: 0.156s
Variance cumulée (50 premières): 96.2%

--- SVM sur données brutes (561 features) ---
Entraînement en 2.143s
Accuracy: 0.9623 | F1-score: 0.9620

--- SVM sur 50 premières composantes PCA ---
Entraînement en 0.387s
Accuracy: 0.9582 | F1-score: 0.9579
```

---

## 🔧 Troubleshooting

### Problème : Les fichiers CSV ne sont pas trouvés

**Solution** :
1. Vérifier que `train.csv` et `test.csv` sont bien dans le dossier `data/`
2. Télécharger depuis [Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)
3. S'assurer que les chemins dans `main.py` sont corrects :
   ```bash
   ls -la data/  # Linux/Mac
   dir data\    # Windows
   ```

### Problème : Erreur "ModuleNotFoundError"

**Solution** :
```bash
# Réinstaller les dépendances
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# Ou utiliser conda si vous utilisez Anaconda
conda install numpy pandas matplotlib scikit-learn
```

### Problème : Utilisation mémoire très élevée

**Solution** :
- Réduire la taille des blocs manuellement dans `src/pca_parallel.py`
- Augmenter le nombre de blocs pour une meilleure parallélisation
- Limiter le nombre de composantes principales via `n_components` dans `main.py`

### Problème : La parallélisation ne fonctionne pas bien

**Vérifier** :
- Nombre de cœurs : `python -c "from multiprocessing import cpu_count; print(cpu_count())"`
- Si la machine n'a qu'1 cœur, la parallélisation n'apportera pas de gains
- Sur Windows, utiliser `if __name__ == "__main__":` pour protéger le code (voir `main.py`)

### Problème : Figures matplotlib ne s'affichent pas

**Solution** :
- Vérifier que matplotlib est en mode compatible :
  ```bash
  python -c "import matplotlib; print(matplotlib.get_backend())"
  ```
- Les graphiques sont toujours sauvegardés en fichiers PNG même si l'affichage échoue

---

## 📚 Crédits & Références

- **Dataset** :  
  [UCI Machine Learning Repository - Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)  
  Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra et Jorge L. Reyes-Ortiz.

- **Citations académiques** :
  - Anguita et al., *A Public Domain Dataset for Human Activity Recognition Using Smartphones*, ESANN 2013.
  - Anguita et al., *Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine*, IWAAL 2012.

- **Librairies open source** : NumPy, pandas, Matplotlib, scikit-learn.

---

## 📖 À savoir

- Ce projet est une **implémentation pédagogique** de la PCA à partir de zéro (sans sklearn.PCA).
- L'accent est mis sur le **calcul haute performance** via la parallélisation multiprocessus.
- Les résultats sont entièrement **reproductibles** : graines aléatoires fixées.
- **Extension possible** : ajouter PCA incrémentale (streaming data) ou kernel-PCA.

---

**Auteur** : [Ouael Djeddi]  
**Licence** : Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

*Projet réalisé dans le cadre d'une démonstration de compétences en calcul haute performance et analyse de données.*
