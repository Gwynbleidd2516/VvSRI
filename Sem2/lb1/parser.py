import re

def getPatsedText(path:str):
    return ' '.join(list(filter(lambda x: len(x)>2, re.sub(r'[^\w\s]', '', open(path).read().lower()).split())))
    # return ' '.join(list(filter(lambda x: len(x)>2, open(path).read().lower().replace(',.!?/\\()<>:','').split())))

def readSaga(sagas:list[tuple[str, int]])->str:
    ans=''
    for name, cap in sagas:
        for i in range(1, cap+1):
            ans+=getPatsedText(name+'/{}.txt'.format(i))
    return ans