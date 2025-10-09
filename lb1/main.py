import numpy as np

from points_gen import gen_point_clouds
from plot import plot_points, plot_clasters
from PCA import PCA
from kmeans_alg import do_kmeans
from elbow_method import do_elbow

#Пункт 1. Генерация случайных точек
print("Как много точек в каждом из трёх облаков сгенерировать?")
N = int(input())

#Здесь будут генерироваться облака случайных точек, пока человеку не покажется, что они визуально различимы
points = []
flag = True
while flag:
    points = gen_point_clouds(N)
    print("Различимы ли визуально три облака точек? Y/N")
    plot_points(points, 'Generated points')
    answer = input()
    if answer == 'Y':
        flag = False

#Пункт 2. Расширение размерности
extended_points = np.zeros((points.shape[0], 5))
extended_points[:, :2] = points

for i in range(extended_points.shape[0]):
    x1 = extended_points[i, 0]
    x2 = extended_points[i, 1]
    extended_points[i, 2] = x1 + x2
    if x1 == 0 or x1 == 1: #На всякий случай проверка, потому что логарифм от 0 и 1 не определён
        extended_points[i, 3] = np.log(4) + x2
    else:
        extended_points[i, 3] = np.log(abs(x1)) + x2 #В задании без модуля, но логарифм от чисел < 0 не определён
    extended_points[i, 4] = np.sin(x1 * x2)

#Пункт 3. Снижение размерности с PCA
new_points = PCA(extended_points)
plot_points(new_points, 'Points after PCA')

#Пункт 4. Метод локтя
best_num_of_clusters = do_elbow(new_points, 10, "График метода Локтя")
print("Подходящее число кластеров =", best_num_of_clusters)

#Пункт 5. Кластеризация по лучшему числу кластеров
centers = do_kmeans(new_points, cluster_capacity=best_num_of_clusters, max_iterations=15, tol=0.1, show=True)
plot_clasters(centers, "Итоговая кластеризация")
