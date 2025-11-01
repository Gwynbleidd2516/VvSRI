import numpy as np

def plain_func(x):
    return -x

def make_points(cap:int):
    np.random.seed(1)
    dat=[]
    ans=[]
    upper=True
    for _ in range(cap):
        x=np.random.uniform(-100.0,100.0)
        y=0.0
        if upper:
            y=np.random.uniform(plain_func(x), 100.0)
            ans.append(1)
        else:
            y=np.random.uniform(-100.0, plain_func(x))
            ans.append(0)
        dat.append(np.array([x,y]))
        upper=not(upper)
    return dat, ans