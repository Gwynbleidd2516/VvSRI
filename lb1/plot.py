import matplotlib.pyplot as plt
import numpy as np

from Center import Center


def plot_points(points, title):
    """Рисует точки"""
    plt.figure()
    plt.title(title)
    plt.plot(*zip(*points), 'o', markersize=2)
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


def plot_clasters(centers: list[Center], title: str):
    """Рисует кластеры"""

    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow", "black", "pink"]
    plt.figure()
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    xmax = -np.inf
    xmin = np.inf

    for i, center in enumerate(centers):
        points = center.points
        if len(points) > 0:
            plt.plot(*zip(*points), 'o', markersize=2, color=colors[i])
            xmin_tmp = min(min(x) for x in points)
            xmin = min(xmin_tmp, xmin)
            xmax_tmp = max(max(x) for x in points)
            xmax = max(xmax_tmp, xmax)

        else:
            xmin = min(min(x.pos) for x in centers)
            xmax = max(max(x.pos) for x in centers)

        plt.plot(center.pos[0], center.pos[1], 'x', markersize=10, color=colors[i])

    plt.xlim(xmin - (xmax - xmin) / 4, xmax + (xmax - xmin) / 4)
    plt.ylim(xmin - (xmax - xmin) / 4, xmax + (xmax - xmin) / 4)
    plt.show()


def plot_elbow_graph(inertia: list, best_num_of_clusters, title: str):
    plt.figure()
    plt.title(title)
    plt.plot(range(1, len(inertia) + 1), inertia)
    plt.axvline(best_num_of_clusters, ls = '--')
    plt.xlabel('Количество кластеров')
    plt.ylabel('WCSS')
    plt.xticks(range(0, len(inertia) + 1))
    plt.grid()
    plt.show()
