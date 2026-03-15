import re

def getParsedText(path:str):
    with open(path, 'r', encoding='cp1251') as file:
        text = file.read().lower()
    
    return ' '.join(list(filter(lambda x: len(x)>2, re.sub(r'[^а-яё\s]', '', text).split())))

def readSaga(books:list[str])->str:
    ans=''
    for name in books:
        ans+=getParsedText('text/{}.txt'.format(name)) + ' '
    return ans