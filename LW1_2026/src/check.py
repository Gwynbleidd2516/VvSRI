from learn import *
from deEmbedding import *

emb_dim = 1000

vocab, index_to_word = loadVocab()
W, C = loadEmbeddings(emb_dim)

for word in ['преступление', 'наказание', 'грех', 'душа', 'бог']:
    _, nearest = nearestWord(vocab[word], W, index_to_word)
    print(f'{word} -> {nearest}')

for word in ['страдание', 'любовь', 'ненависть', 'стыд', 'гордость']:
    _, nearest = nearestWord(vocab[word], W, index_to_word)
    print(f'{word} -> {nearest}')

for word in ['раскольников', 'соня', 'настасья', 'алеша']:
    _, nearest = nearestWord(vocab[word], W, index_to_word)
    print(f'{word} -> {nearest}')

for word in ['деньги', 'бедность', 'власть', 'народ', 'закон']:
    _, nearest = nearestWord(vocab[word], W, index_to_word)
    print(f'{word} -> {nearest}')
