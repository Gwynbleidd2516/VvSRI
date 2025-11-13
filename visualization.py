import matplotlib.pyplot as plt
import numpy as np

def visualize_train(iteration_errors):
    # Отрисовка графика истории обучения
    plt.figure()
    plt.title("График ошибки нейросети")
    plt.xlabel("Эпохи")
    plt.ylabel("Ошибка нейросети")
    plt.grid()
    plt.plot(iteration_errors)
    plt.show()

def visualize_valid(valid_inp, valid_ans, correct_list, accuracy):
    # Отрисовка валидации
    # Отрисовка точек red - если угадано и она ниже прямой y = -x, blue - если угадано и она выше прямой
    # yellow - неправильно
    plt.figure()
    plt.title(f"Визуализация валидации нейросети\nAccuracy = {accuracy}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    for i in range(len(valid_inp)):
        point_color = "red"
        if valid_ans == 1:
            point_color = "blue"
        if correct_list[i] == 0:
            point_color = "yellow"

        plt.plot(valid_inp[i][0], valid_inp[i][1], 'o', color=point_color)

    # Отрисовка прямой y = -x
    max_x = max(point[0] for point in valid_inp)
    min_x = min(point[0] for point in valid_inp)
    plt.plot([min_x, -min_x], [max_x, -max_x], ls="--")

    plt.show()


def visualize_probability(network):
    # Отрисовка распределения вероятностей
    plt.figure()
    plt.title('Распределение вероятностей точек быть над или под прямой')
    plt.xlabel('X')
    plt.ylabel('Y')

    min_x = -3
    max_x = 3
    x, y = np.meshgrid(np.linspace(min_x, max_x, 100), np.linspace(min_x, max_x, 100))
    grid_points = np.c_[x.ravel(), y.ravel()]
    predictions = np.array([network.predict(point) for point in grid_points])
    predictions = predictions.reshape(x.shape)

    plt.contourf(x, y, predictions, levels=100, cmap='RdYlBu')
    plt.colorbar()
    plt.plot([min_x, -max_x], [-min_x, max_x], ls='--')

    plt.show()