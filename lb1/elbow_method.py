import numpy as np
from kmeans_alg import do_kmeans
from plot import plot_elbow_graph


def do_elbow(data: np.array, max_num_of_clusters: int, title: str):
    inertia = []
    for i in range(max_num_of_clusters):
        centers = do_kmeans(data, cluster_capacity=i + 1, max_iterations=15, tol=0.1)
        summ = 0
        for j in range(len(centers)):
            summ += centers[j].wcss()
        inertia.append(summ)

    """Проводим из верхней левой точки графика локтя до нижней правой прямую.
    Самая далёкая точка графика до этой прямой и есть искомый локоть"""
    dists = []
    for x, y in enumerate(inertia):
        y2 = inertia[len(inertia) - 1]
        y1 = inertia[0]
        x2 = len(inertia)
        x1 = 1

        # Формула расстояния от прямой до точки
        dist = (abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)) / (np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
        dists.append(dist)

    best_num_of_clusters = np.argmax(dists) + 1

    plot_elbow_graph(inertia, best_num_of_clusters, title)

    return best_num_of_clusters
