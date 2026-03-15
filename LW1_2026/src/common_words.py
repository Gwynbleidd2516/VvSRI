from parser import readSaga
from learn import *
from collections import *

text=readSaga([str(i) for i in range(1,5)]).split()
print(len(text))

word_freq = Counter(text)
top_5 = word_freq.most_common(100)
print(top_5)