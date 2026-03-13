from parser import readSaga
from learn import *
from collections import *
text=readSaga([('witcher', 4)]).split()
words=Counter(text)
words={x:words.get(x) for x in words}
a=makeEmbedding(len(words), 500)
print(len(words))