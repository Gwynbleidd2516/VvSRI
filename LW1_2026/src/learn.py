from math import *
import numpy as np
import json

def sigmoid(val):
    return 1/(1+exp(-val))

def makeEmbedding(vocab_size, emb_dim):
    limit=1/sqrt(emb_dim)
    s= np.random.uniform(-limit, limit, (vocab_size, emb_dim))
    return s

def generate_pairs(text : list, vocab : dict, window : int = 3, neg_k : int = 10):
    for i, word in enumerate(text):
        start = max(0, i - window)
        end = min(len(text), i + window + 1)
        pos = [vocab[text[j]] for j in range(start,end,1) if i != j]
        neg = []
        while len(neg) < neg_k:
            idx = vocab[text[np.random.randint(0, len(text))]]
            if not idx in pos and idx != i:
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

def saveVocab(vocab):
    with open(file = 'vocab.json', mode = 'w', encoding='cp1251') as f:
        json.dump(vocab, f, ensure_ascii=False)
    f.close()

def saveEmbeddings(W, C, emb_dim):
    np.save('W_{}.npy'.format(emb_dim), W)
    np.save('C_{}.npy'.format(emb_dim), C)

def loadVocab():
    with open('vocab.json', 'r', encoding='cp1251') as f:
        vocab = json.load(f)
    f.close()
    index_to_word = {int(i): w for w,i in vocab.items()}
    return vocab, index_to_word

def loadEmbeddings(emd_dim):
    W = np.load('W_{}.npy'.format(emd_dim))
    C = np.load('C_{}.npy'.format(emd_dim))
    return W, C

