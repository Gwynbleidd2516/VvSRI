import numpy as np

# Просто генерация точек и ответов выше они или ниже
def generate_points_and_ans(num = 100, low = -10, high = 10, seed = 42):
    np.random.seed(seed)
    points = np.random.uniform(low, high, (num, 2))
    answers = list(map(lambda x: 1 if x[1] > -x[0] else 0, points))
    return (points, answers)