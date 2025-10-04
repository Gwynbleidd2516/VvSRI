import matplotlib.pyplot as plt

def plot_points(points):
    """Рисует точки"""
    plt.figure()
    plt.scatter(*zip(*points), s = 5)
    plt.xlabel = 'X'
    plt.ylabel = 'Y'
    xmin = min(min(x) for x in points)
    xmax = max(max(x) for x in points)
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
    plt.grid()
    plt.show()
