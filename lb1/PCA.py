import numpy as np

def PCA(matrix: np.array, k=2):
    """Снижение размерности. На вход подаётся матрица типа np.array и размерность, до которой нужно снизить"""

    matrix_norm = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0) #Вычисляется нормированная матрица

    matrix_c = (matrix_norm.T @ matrix_norm) / (matrix_norm.shape[0] - 1) #Вычисляется матрица ковариации

    eigenvalues, eigenvectors = np.linalg.eig(matrix_c) #Вычисление собственных векторов матрицы ковариации

    # От собственных векторов берутся соответствующие k наибольшим собственным числам
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    max_eigenvectors = eigenvectors_sorted[:, :k]

    return matrix_norm @ max_eigenvectors #нормированная матрица умножается на полученные вектора и возвращает результат
