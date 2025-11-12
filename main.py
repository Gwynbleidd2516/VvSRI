from network import Network
from make_points import generate_points_and_ans
from visualization import visualize_valid, visualize_train, visualize_probability

network = Network()

size_of_train = 10000
size_of_valid = 100
Lambda = 0.17
tolerance = 0.004
batch_size = 32
train_input, train_answers = generate_points_and_ans(num=size_of_train)
valid_input, valid_answers = generate_points_and_ans(num=size_of_valid)

print()

error_history = network.train(train_input, train_answers, batch_size=batch_size, Lambda=Lambda, tolerance=tolerance)
visualize_train(error_history)

print()

accuracy = network.get_accuracy(valid_input, valid_answers)
print("Accuracy =", accuracy)
correct_list = network.validation(valid_input, valid_answers)
visualize_valid(valid_input, valid_answers, correct_list)
visualize_probability(network)

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





