import matplotlib.pyplot as plt
import numpy as np

def plot_points(points:np.array, title:str, clouds_capasity:int=1):
    """Рисует точки"""
    plt.figure()
    plt.title(title)
    n=int(len(points)/clouds_capasity)
    for i in range(1,clouds_capasity+1):
        plt.plot([x[0] for x in points[(i-1)*n:i*n]],[x[1] for x in points[(i-1)*n:i*n]], 'o', markersize=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()

def plot_with_centers(points:np.array, clusters:np.ndarray, title:str, clouds_capasity:int=1):
    plt.figure()
    plt.title(title)
    n=int(len(points)/clouds_capasity)
    for i in range(1,clouds_capasity+1):
        plt.plot([x[0] for x in points[(i-1)*n:i*n]],[x[1] for x in points[(i-1)*n:i*n]], 'o', markersize=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    for x in clusters:
        plt.plot(x[0], x[1], 'x')
    plt.grid()
    plt.show()

def plot_elbow_graph(arr:list, title:str):
    plt.figure()
    plt.title(title)
    plt.plot(arr[0],arr[1])
    plt.xlabel('Количество кластеров')
    plt.ylabel('Сумма внутрикластерных расстояний')
    plt.grid()
    plt.show()