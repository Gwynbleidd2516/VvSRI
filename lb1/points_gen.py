import numpy as np

def far_point(point, k, low, high):
    """Сгенерировать далёкую точку от заданной. Генерируется k случайных точек-кандидатов и одна с наибольшей квадратной
    дистанцией будет возвращена"""
    q = np.random.default_rng().uniform(size=(k,2), low=low, high=high)
    square_dists = np.array([((point[0] - x[0])**2 + (point[1] - x[1])**2) for x in q])
    return q[square_dists.argmax()]

def point_cloud(point, n, k, low, high):
    """Генерирует облако точек, основываясь на заданной. n раз генерируется k случайных точек-кандидатов
     и одна с наименьшей квадратной дистанцией будет выбрана. В итоге возвращается массив точек"""
    if n <= 1: return point
    else: yield point
    for i in range(1, n):
        q = np.random.default_rng().uniform(size=(k, 2),low=low, high=high)
        square_dists = np.array([((point[0] - x[0]) ** 2 + (point[1] - x[1]) ** 2) for x in q])
        p = q[square_dists.argmin()]
        yield p

def gen_points(n=100, k1=20, k2=40, low=0, high=1000):
    """Генерирует 3 облака точек"""
    point1 = np.random.default_rng().uniform(size=2)
    cloud1 = list(point_cloud(point1, n, k2, low, high))
    point2 = far_point(point1, k1, low, high)
    cloud2 = list(point_cloud(point2, n, k2, low, high))
    point3 = far_point(point2, k1, low, high)
    point4 = far_point(point1,k1, low, high)
    cloud3 = list(point_cloud((point3+point4)/2, n, k2, low, high))
    return np.array(cloud1 + cloud2 + cloud3)
