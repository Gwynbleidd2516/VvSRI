import numpy as np
import random
from math import *

def gen_point_clouds(ceneter_coords:list, capasity:int, scatter:int):
    """Геренирует облака точек"""
    ans=[]
    for point in ceneter_coords:
        for i in range(capasity):
            inCircPoint=[scatter*2,scatter*2]
            while inCircPoint[0]**2+inCircPoint[1]**2>scatter**2:
                inCircPoint=[random.uniform(-scatter, scatter), random.uniform(-scatter, scatter)]
            x,y=point[0]+inCircPoint[0],point[1]+inCircPoint[1]
            ans.append(np.array([x,y,x+y, log(abs(x))+y,sin(x*y)]))
    return np.array(ans)

def gen_normal_point_clouds(loc:list, capasity:int):
    ans=[]
    for x in loc:
        ans+=[np.random.normal(loc=x, scale=1, size=(capasity,2))]
    ans=np.concatenate(ans).tolist()
    for i in range(len(ans)):
        x1,x2=ans[i]
        ans[i]+=[x1+x2,log(np.abs(x1))+x2, sin(x1*x2)]
    return np.array(ans)