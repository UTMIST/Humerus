"""
A script to reduce the dimensionality of the CAH sentence embeddings.
"""

import h5py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from humerusbot.data.embed_sentences import PARENT_DIR, load_embeddings

PCA_DIR = PARENT_DIR / 'embeddings-pca.h5'


def run_pca(dim):
    embeddings = load_embeddings()
    pca = PCA(n_components=dim)
    low_dim_embeddings = pca.fit_transform(embeddings)

    with h5py.File(PCA_DIR, 'w') as hf:
        hf.create_dataset('embeddings-pca', data=embeddings)
    return low_dim_embeddings


def _plot_2d_embeddings(x):
    x = x.transpose()
    plt.scatter(x[0], x[1], s=2)
    plt.show()


def main():
    x = run_pca(dim=2)
    _plot_2d_embeddings(x[0:1000])


if __name__ == '__main__':
    main()
