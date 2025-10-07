import numpy as np

def pca(matrix:np.array,k=2):
    """Снижение размерности. На вход подаётся матрица типа np.array и размерность, до которой нужно снизить"""
    mat_norm=(matrix-matrix.mean(axis=0)/matrix.std(axis=0))
    mat_cov=(mat_norm.T @ mat_norm)/(mat_norm.shape[0]-1)
    eigen_values, eigen_vectors=np.linalg.eigh(mat_cov)
    indices_sort=np.argsort(eigen_values)[::-1]
    eigen_vectors_sort=eigen_vectors[:, indices_sort]
    return mat_norm @ eigen_vectors_sort[:, :2]