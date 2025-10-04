import numpy as np
from points_gen import gen_points
from plot_points import plot_points
from PCA import PCA

#Пункт 1. Генерация случайных точек
print("Как много точек в каждом из трёх облаков сгенерировать?")
N = int(input())

#Здесь будут генерироваться облака случайных точек, пока человеку не покажется, что они визуально различимы
flag = True
while flag:
    points = gen_points(N)
    print("Различимы ли визуально три облака точек? Y/N")
    plot_points(points)
    answer = input()
    if answer == 'Y': flag = False

#Пункт 2. Расширение размерности
extended_points = np.zeros((points.shape[0], 5))
extended_points[:, :2] = points

for i in range(extended_points.shape[0]):
    x1 = extended_points[i, 0]
    x2 = extended_points[i, 1]
    extended_points[i, 2] = x1 + x2
    extended_points[i, 3] = np.log(x1) + x2
    extended_points[i, 4] = np.sin(x1 * x2)

#Пункт 3. Снижение размерности с PCA
new_points = PCA(extended_points)
plot_points(new_points)

#Пункт 4. Кластеризация
