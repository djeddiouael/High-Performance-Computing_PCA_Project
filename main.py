import numpy as np
import time
from src.data_loader import load_har_data
from src.pca_parallel import ParallelPCA
from src.visualization import plot_pca_2d, plot_variance_explained
from src.classifier import train_and_evaluate

def main():
    # Chargement
    X_train, y_train, X_test, y_test, _ = load_har_data('data/train.csv', 'data/test.csv')
    print(f"Dimensions train: {X_train.shape}, test: {X_test.shape}")

    # ACP parallèle
    pca = ParallelPCA(n_components=100)  # On garde 100 composantes pour la classification
    t0 = time.time()
    X_train_proj = pca.fit_transform(X_train)
    X_test_proj = pca.transform(X_test)
    print(f"Transformation totale ACP: {time.time()-t0:.2f}s")
    print(f"Variance cumulée (50 premières): {np.sum(pca.explained_variance_[:50]) / np.sum(pca.explained_variance_)*100:.1f}%")

    # Visualisation 2D (sur les deux premières composantes)
    plot_pca_2d(X_train_proj, y_train, title="Séparabilité des activités sur PC1/PC2 (HAR)", save_path="pca_2d.png")

    # Variance expliquée
    plot_variance_explained(pca.explained_variance_, n_components=50)

    # Classification brute vs réduite
    print("\n--- SVM sur données brutes (561 features) ---")
    train_and_evaluate(X_train, y_train, X_test, y_test, desc="SVM brut")

    print("\n--- SVM sur 50 premières composantes PCA ---")
    train_and_evaluate(X_train_proj[:, :50], y_train, X_test_proj[:, :50], y_test, desc="SVM PCA(50)")

if __name__ == "__main__":
    main()