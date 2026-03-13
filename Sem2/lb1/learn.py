from math import *
import numpy as np

def sigmoid(val):
    return 1/(1+exp(-val))

def makeEmbedding(vocab_size, emb_dim):
    limit=1/sqrt(emb_dim)
    s= np.random.uniform(-limit, limit, (vocab_size, emb_dim))
    return s

# def train(words, emb_dim, epoch=5, lr=0.01, c_neg=10, window=3):
    