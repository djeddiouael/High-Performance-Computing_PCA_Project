import matplotlib.pyplot as plt
import numpy as np

def plot_pca_2d(X_proj, y, title="Projection PCA (2D)", save_path=None):
    activities = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(activities)))
    plt.figure(figsize=(8,6))
    for act, col in zip(activities, colors):
        mask = y == act
        plt.scatter(X_proj[mask, 0], X_proj[mask, 1],
                    c=[col], label=act, alpha=0.6, edgecolors='none', s=10)
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.title(title)
    plt.legend(markerscale=3)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_variance_explained(explained_variance, n_components=50):
    cumsum = np.cumsum(explained_variance) / np.sum(explained_variance) * 100
    plt.figure(figsize=(8,4))
    plt.bar(range(1, min(n_components+1, len(explained_variance)+1)),
            explained_variance[:n_components] / np.sum(explained_variance) * 100,
            alpha=0.7, label='Individuelle')
    plt.step(range(1, n_components+1), cumsum[:n_components], where='mid',
             label='Cumulée', color='red')
    plt.xlabel("Composante")
    plt.ylabel("Variance expliquée (%)")
    plt.title("Variance expliquée par composante (PCA)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()