import numpy as np
import matplotlib.pyplot as plt
from point_gen import *
from plot_points import *
from PCA import pca
from kmeans_alg import *
from elbow_method import *

p=list()
# Пункт 1. Генерация случайных точек
print("Напишите координаты для ваших 3 точек")
for i in range(3):
    print(str(i+1)+' точка!')
    tmp=[int(input('x: ')), int(input('y: '))]
    p.append(tmp)
print("Сколко точек должно быть в одном облаке?")
cap = int(input())
print("Какой радиус разброса должен быть?")
sct = int(input())
mass=gen_point_clouds(p, cap, sct)
plot_points(mass, 'Сгенерированные точки',3)
# Пункт 2. Обработка PCA
mass=pca(mass)
plot_points(mass, 'После обработки PCA',3)
# Пункт 3. Нахождение идеального количетсва центорв точек
inrt=do_elbow(mass, range(1,10))
plot_elbow_graph([range(1,10), inrt], 'Метод локтя')

cluster_capasity=int(input('Выберите количество кластеров: '))
clst, wcll=do_kmeans(mass, cluster_capasity)

plot_with_centers(mass, clst, ' ')