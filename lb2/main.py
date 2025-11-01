from Perceptron import *
from make_points import make_points

inp,ans=make_points(500)
test,test_ans=make_points(1500)

p=Perceptron(0.9, .1)

p.learn(inp, ans, 1000000)

a=[]
for i in range(1500):
    if test_ans[i]==1:
        a.append(p.calc(test[i])>=0.5)
    else:
        a.append(p.calc(test[i])<0.5)
print(100*a.count(True)/1500)