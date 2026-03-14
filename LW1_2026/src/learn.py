from math import *
import numpy as np

def sigmoid(val):
    return 1/(1+exp(-val))

def makeEmbedding(vocab_size, emb_dim):
    limit=1/sqrt(emb_dim)
    s= np.random.uniform(-limit, limit, (vocab_size, emb_dim))
    return s

def generate_pairs(text : list, vocab : dict, window : int = 3, neg_k : int = 10):
    for i, word in enumerate(text):
        start = max(0, i - 3)
        end = min(len(text) - 1, i + window + 1)
        pos = [vocab[text[j]] for j in range(start,end,1) if i != j]
        neg = []
        while len(neg) < neg_k:
            idx = vocab[text[np.random.randint(0, len(text))]]
            if not idx in pos:
                neg.append(idx)
        yield (vocab[word], pos, neg)

def train(text, vocab, W, C, window=3, neg_k=10, epochs=5, lr=0.01):
    for epoch in range(epochs):
        total_loss = 0

        for w_idx, pos, neg in generate_pairs(text, vocab, window, neg_k):

            grad_w = np.zeros_like(W[w_idx])

            for c_idx in pos:
                s = sigmoid(C[c_idx] @ W[w_idx])
                total_loss += -np.log(s + 1e-10)
                grad_w += (s - 1) * C[c_idx]
                C[c_idx] -= lr * (s - 1) * W[w_idx]

            for c_idx in neg:
                s = sigmoid(C[c_idx] @ W[w_idx])
                total_loss += -np.log(1 - s + 1e-10)
                grad_w += s * C[c_idx]
                C[c_idx] -= lr * s * W[w_idx]

            W[w_idx] -= lr * grad_w

        print(f'Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}')
    return W, C
            




# def train(words, emb_dim, epoch=5, lr=0.01, c_neg=10, window=3):
    