from math import sqrt
import numpy as np

class Center:

    def __init__(self):
        self.pos = [0.,0.]
        self.points = []
        self.wcss = 0.
        self.prev_pos = [0.,0.]

    # Расчёт новых центров кластеров
    def make_new_pos(self):
        self.prev_pos = self.pos
        if self.points:
            self.pos = np.mean(self.points, axis = 0)
            self.wcss = sum(map(self.dist, self.points))

    # Метод для расчёта расстояния от центра до какой-либо точки
    def dist(self, point: np.array):
        return sqrt((self.pos[0] - point[0])**2 + (self.pos[1] - point[1])**2)

    # Рассчитываем достаточно ли мало изменение центров точек по сравнению с предыдущей итерацией
    def enough(self, tol):
        return sqrt((self.prev_pos[0] - self.pos[0])**2 + (self.prev_pos[1] - self.pos[1])**2) < tol