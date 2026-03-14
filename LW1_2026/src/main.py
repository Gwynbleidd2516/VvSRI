from parser import readSaga
from learn import *
from collections import *

text=readSaga([str(i) for i in range(1,5)]).split()
print(len(text))

word_freq = Counter(text)

vocab = {w: i for i,w in enumerate(word_freq)}
index_to_word = {i: w for w,i in vocab.items()}

print(f'Размер словаря: {len(vocab)}')
print(f'Длина текста: {len(text)}')

pairs = generate_pairs(text, vocab)

V = len(vocab)
emb_dim = 100
W = makeEmbedding(V, emb_dim)
C = makeEmbedding(V, emb_dim)

for w, pos, neg in generate_pairs(text[:100], vocab):
    print(w, pos, neg)
    print(f'word: {index_to_word[w]}')
    print(f'pos: {[index_to_word[i] for i in pos]}')
    print(f'neg: {[index_to_word[i] for i in neg]}')
    break

text_small = text[:500]
W = makeEmbedding(V, emb_dim)
C = makeEmbedding(V, emb_dim)
train(text, vocab, W, C, epochs=5)