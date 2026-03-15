import numpy as np

def nearestWord(w_idx, W, index_to_word, diff_func_type = "MSE"):
    min_dif = float('inf')
    res = -1
    for i in range(W.shape[0]):
        if i != w_idx:
            if diff_func_type == "MSE":
                cur_dif = MSE(W[w_idx], W[i])
            else:
                cur_dif = binaryCrossEntropy(W[w_idx], W[i])

            if cur_dif < min_dif:
                min_dif = cur_dif
                res = i
    return res, index_to_word[res]

def MSE(word_1, word_2):
    return np.mean((word_1 - word_2) ** 2)

def binaryCrossEntropy(word_1, word_2):
    p = sigmoid(word_1)
    q = sigmoid(word_2)
    return -np.mean(p * np.log(q + 1e-10) + (1 - p) * np.log(1 - q + 1e-10))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dotProduct(word_1, word_2):
    return (word_1 @ word_2) / (np.linalg.norm(word_1) * np.linalg.norm(word_2))
