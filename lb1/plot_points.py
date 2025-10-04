import matplotlib.pyplot as plt

def plot_points(points, title):
    """Рисует точки"""
    plt.figure()
    plt.title(title)
    plt.scatter(*zip(*points), s = 5)
    plt.xlabel('X')
    plt.ylabel('Y')
    xmin = min(min(x) for x in points)
    xmax = max(max(x) for x in points)
    plt.xlim(xmin - (xmax - xmin) / 4, xmax + (xmax - xmin) / 4)
    plt.ylim(xmin - (xmax - xmin) / 4, xmax + (xmax - xmin) / 4)
    """(xmax - xmin) / 4 вычитается для того, чтобы было свободное пространство на графике
     для более удобного просмотра"""
    plt.grid()
    plt.show()
