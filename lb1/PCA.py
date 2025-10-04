import numpy as np

def PCA(matrix: np.array, k=2):
    """Снижение размерности. На вход подаётся матрица типа np.array и размерность, до которой нужно снизить.
    Вычисляется нормированная матрица, матрица ковариации, а затем собственные вектора матрицы ковариации.
    От собственных векторов берутся соответствующие k наибольшим собственным числам, и нормированная матрица умножается
    на полученные вектора"""
    matrix_norm = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
    matrix_c = (matrix_norm.T @ matrix_norm) / (matrix_norm.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eig(matrix_c)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    max_eigenvectors = eigenvectors_sorted[:, :k]

    return matrix_norm @ max_eigenvectors
