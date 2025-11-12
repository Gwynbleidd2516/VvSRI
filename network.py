import numpy as np

def sigmoida(x):
    return 1 / (1 + np.exp(-x))

class Network:
    def __init__(self):
        self.weights = np.random.randn(4)

    def predict(self, inp):
        input1 = self.weights[0] * inp[0]
        input2 = self.weights[1] * inp[1]
        output = sigmoida(self.weights[2] * input1 + self.weights[3] * input2)
        return output

    def grad(self, inp, ans, my_ans):
        result = []

        for i in range(4):
            w = self.weights[(i + 2) % 4]
            x = inp[i % 2]
            result.append(2 * (my_ans - ans) * my_ans * (1 - my_ans) * w * x)

        return np.array(result)

    def train(self, train_inp, train_ans, batch_size = 32, max_iterations = 1000, Lambda = 0.16, tolerance = 0.01):
        print(f"Начало обучения c λ = {Lambda}, ε = {tolerance}, размером батча = {batch_size}...")
        print(f"Начальные веса: ", end = '')
        for i in range(len(self.weights)):
            ending = '\n'
            if i != len(self.weights) - 1:
                ending = ', '
            print(f"w{i} = {self.weights[i] : .4f}", end = ending)

        iteration_errors = []
        for iteration in range(max_iterations):
            iteration_error = 0

            for i in range(0, len(train_inp), batch_size):
                batch_inp = train_inp[i: i + batch_size]
                batch_ans = train_ans[i: i + batch_size]
                batch_weights = [0]*4
                batch_error = 0
                batch_len = len(batch_inp)

                for j in range(batch_len):
                    pred_ans = self.predict(batch_inp[j])
                    batch_error += (batch_ans[j] - pred_ans) ** 2
                    batch_weights += self.grad(batch_inp[j], batch_ans[j], pred_ans)

                self.weights -= Lambda * (batch_weights / batch_len)
                iteration_error += batch_error / batch_len

            iteration_error /= len(train_inp) // batch_size
            iteration_errors.append(iteration_error)

            if iteration_errors[-1] < tolerance:
                break

        print(f"Обучение закончено за {len(iteration_errors)} эпох, ошибка в последней эпохе = {iteration_errors[-1]}")
        print(f"Конечные веса: ", end = '')
        for i in range(len(self.weights)):
            ending = '\n'
            if i != len(self.weights) - 1:
                ending = ', '
            print(f"w{i} = {self.weights[i] : .4f}", end = ending)

        return iteration_errors

    def get_accuracy(self, valid_inp, valid_ans):
        count_correct = 0

        for i in range(len(valid_inp)):
            pred_ans = self.predict(valid_inp[i])
            if pred_ans > 0.5 and valid_ans[i] == 1 or pred_ans < 0.5 and valid_ans[i] == 0:
                count_correct += 1

        return count_correct / len(valid_inp)

    def validation(self, valid_inp, valid_ans):
        correct_list = []

        for i in range(len(valid_inp)):
            pred_ans = self.predict(valid_inp[i])
            if pred_ans > 0.5 and valid_ans[i] == 1 or pred_ans < 0.5 and valid_ans[i] == 0:
                correct_list.append(1)
            else:
                correct_list.append(0)

        return correct_list