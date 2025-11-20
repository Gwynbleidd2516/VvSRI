from network import Network
from make_points import generate_points_and_ans
from visualization import visualize_valid, visualize_train, visualize_probability

network = Network()

size_of_train = 10000 # Количество точек для обучения
size_of_valid = 100 # Количество точек для валидации
Lambda = 0.17 # Лямбда
tolerance = 0.004 # Допустимая погрешность
batch_size = 32 # Размер батча
train_input, train_answers = generate_points_and_ans(num=size_of_train) # Генерация точек для обучения
valid_input, valid_answers = generate_points_and_ans(num=size_of_valid) # Генерация точек для валидации

print()

# Обучение и визуализация процесса обучения
error_history = network.train(train_input, train_answers, batch_size=batch_size, Lambda=Lambda, tolerance=tolerance)
visualize_train(error_history)

print()

# Валидация и расчёт accuracy
accuracy = network.get_accuracy(valid_input, valid_answers)
print("Accuracy =", accuracy)
correct_list = network.validation(valid_input, valid_answers)
visualize_valid(valid_input, valid_answers, correct_list, accuracy)

# Визуализация вероятностей точек на графике
visualize_probability(network)

print()

# Проверка нейросети вручную, ввод координат пользователем
print("Тест нейросети вручную")
while True:
    print("Введите координаты точки")
    inp = input().strip()
    if inp == "END":
        break

    inp = list(map(float, inp.split()))
    pred_ans = network.predict(inp)
    print(f"Ответ программы: {"выше" if pred_ans > 0.5 else "ниже"} прямой y = -x")
    print(f"Выход нейросети: {pred_ans: .4f}")
    correct = "верно" if (inp[0] > -inp[1] and pred_ans > 0.5) or (inp[0] < -inp[1] and pred_ans < 0.5) else "неверно"
    print(f"Программа ответила {correct}")
    print()





