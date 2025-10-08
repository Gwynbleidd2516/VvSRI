import numpy as np
from plot import plot_clasters
from Center import Center

# Улучшенный выбор центров
def greedy_kmeans_plus_plus(data: np.array, cluster_capacity: int):
    centers = []
    n = data.shape[0]
    #Первый центр выбираем случайно
    centers.append(data[np.random.choice(n)])
    # n_local_trials вычисляется именно так, гарантируя, что у нас будет не слишком много попыток (log медленно растёт)
    n_local_trials = 2 + int(np.log(cluster_capacity))

    for i in range(cluster_capacity - 1):
        # Вычисляем расстояние всех точек до предыдущего центра
        dists = np.sum((data - centers[i])**2, axis=1)
        candidates = []
        candidates_dists = []

        # n_local_trials раз выбираем точки с вероятностью квадрата их расстояний до центра
        for _ in range(n_local_trials):
            # Вероятности
            probes = dists / np.sum(dists)

            # Кандидаты на выбор в центры
            candidates.append(data[np.random.choice(n, p=probes)])
            candidates_dists.append(np.sum((candidates[-1] - centers[:i+1]) ** 2))

        # Лучший кандидат - это кандидат с наибольшим расстоянием от остальных центров
        best_candidate = candidates[np.argmax(candidates_dists)]
        # Он и становится новым центром
        centers.append(best_candidate)

    arr_centers = []
    for i in range(cluster_capacity):
        arr_centers.append(Center())
        arr_centers[i].pos = centers[i]

    return arr_centers

def do_kmeans(data: np.array, cluster_capacity: int, max_iterations: int, tol: float, show=False):
    centers = greedy_kmeans_plus_plus(data, cluster_capacity)

    # Распределяем точки по кластерам
    n = 0
    stop = False
    while not(stop):
        n += 1 # Считает количество итераций для остановки

        for center in centers:
            center.points = []

        for point in data:
            # Проходимся по всем центрам, рассчитывая расстояние от них до точки и добавляем точку к ближайшему центру
            dists = [x.dist(point) for x in centers]
            min_ind = np.argmin(dists)

            # Добавляем точку к ближайшему центру, рассчитываем отклонений
            centers[min_ind].points.append(point)

        for i in range(cluster_capacity):
            centers[i].make_new_pos()
        #print([x.pos for x in centers])
        # Если изменение координат центров небольшое или количество итераций слишком много, то остановимся
        stop = all([x.enough(tol) for x in centers]) or n > max_iterations
        if show:
            plot_clasters(centers, f"Итерация {n}")

    return centers