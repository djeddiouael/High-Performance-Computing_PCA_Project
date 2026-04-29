import numpy as np
from multiprocessing import Pool, cpu_count
import time

def _centered_block_product(args):
    """Calcule X_block.T @ X_block pour un bloc centré-réduit."""
    X_block = args
    return X_block.T @ X_block

class ParallelPCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.std_ = None
        self.components_ = None  # V_k
        self.explained_variance_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        # Centrage / réduction
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Éviter division par zéro
        self.std_[self.std_ == 0] = 1.0
        X_scaled = (X - self.mean_) / self.std_

        # Parallélisation du calcul de covariance
        t0 = time.time()
        n_cpus = cpu_count()
        # Découpage en blocs de lignes (taille ~10 % des données, mini 1)
        chunk_size = max(1, n_samples // (n_cpus * 4))
        blocks = [X_scaled[i:i+chunk_size] for i in range(0, n_samples, chunk_size)]

        with Pool(processes=n_cpus) as pool:
            partial_mats = pool.map(_centered_block_product, blocks)

        # Somme des produits partiels
        sum_cov = sum(partial_mats)
        cov = sum_cov / (n_samples - 1)
        t_cov = time.time() - t0
        print(f"[PCA] Covariance calculée en {t_cov:.3f} s sur {n_cpus} cœurs, {len(blocks)} blocs.")

        # Décomposition spectrale
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Tri décroissant
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        if self.n_components is not None:
            eigvals = eigvals[:self.n_components]
            eigvecs = eigvecs[:, :self.n_components]

        self.components_ = eigvecs
        self.explained_variance_ = eigvals
        return self

    def transform(self, X):
        X_scaled = (X - self.mean_) / self.std_
        return X_scaled @ self.components_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_proj):
        return X_proj @ self.components_.T * self.std_ + self.mean_