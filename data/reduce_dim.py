"""
A script to reduce the dimensionality of the CAH sentence embeddings.
"""

import h5py
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA

from humerusbot.data.embed_sentences import PARENT_DIR, load_embeddings

PCA_DIR = PARENT_DIR / 'embeddings-pca.h5'
UMAP_DIR = PARENT_DIR / 'embeddings-umap.h5'


def run_pca(dim):
    embeddings = load_embeddings()
    pca_fit = PCA(n_components=dim)
    components = pca_fit.fit_transform(embeddings)

    with h5py.File(PCA_DIR, 'w') as hf:
        hf.create_dataset('embeddings-pca', data=components)
    return components


def run_umap(dim):
    embeddings = load_embeddings()
    umap_fit = umap.UMAP(n_components=dim, verbose=True)
    components = umap_fit.fit_transform(embeddings)

    with h5py.File(UMAP_DIR, 'w') as hf:
        hf.create_dataset('embeddings-umap', data=components)
    return components


def _plot_2d_embeddings(x):
    x = x.transpose()
    plt.scatter(x[0], x[1], s=2)
    plt.show()


def main():
    x = run_umap(dim=2)
    _plot_2d_embeddings(x)


if __name__ == '__main__':
    main()
