import pickle
import numpy as np
'''
The pca class
input matrix should be m x 768 where m is the number of examples.
'''
class pca:
    def __init__(self):
        with open('data/pca_cache_69.pkl', 'rb') as f:
            cache = pickle.load(f)

        self.eigenvector = cache["eigenvectors"]
        self.eigenvalue = cache["eigenvalues"]
        self.mean = cache["mean"]
        self.sd = cache["std"]

    def reduce_kdim(self, X, k):
        X = (X - self.mean) / self.sd
        X_reduced = np.dot(X, self.eigenvector[:, :k])
        return X_reduced

# testing:
'''
arr = np.random.rand(1,768)
print(arr.shape)
p = pca()
arr = p.reduce_kdim(arr, 69)
print(arr.shape)
'''