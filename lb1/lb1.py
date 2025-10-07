import matplotlib.pyplot as plt
import numpy as np
import random
from math import *
# from sklearn.cluster import KMeans

n=int(input())
scatter=5
p1=[10,10]
p2=[40,10]
p3=[25,20]

n1=[[],[],[],[],[]]
n2=[[],[],[],[],[]]
n3=[[],[],[],[],[]]

mass=[]
for i in range(n):
    buff=[random.randint(-scatter, scatter), random.randint(-scatter, scatter)]
    x,y=p1[0]+buff[0],p1[1]+buff[1]
    mass.append(np.array([x,y,x+y, log(x)+y,sin(x*y)]))
for i in range(n):
    buff=[random.randint(-scatter, scatter), random.randint(-scatter, scatter)]
    x,y=p2[0]+buff[0],p2[1]+buff[1]
    mass.append(np.array([x,y,x+y, log(x)+y,sin(x*y)]))
for i in range(n):
    buff=[random.randint(-scatter, scatter), random.randint(-scatter, scatter)]
    x,y=p3[0]+buff[0],p3[1]+buff[1]
    mass.append(np.array([x,y,x+y, log(x)+y,sin(x*y)]))
    
# for i in range(n):
#     buff=[random.randint(scatter[0], scatter[1]), random.randint(scatter[0], scatter[1])]
#     n1[0].append(p1[0]+buff[0])
#     n1[1].append(p1[1]+buff[1])
#     n1[2].append(n1[0][-1]+n1[1][-1])
#     n1[3].append(log(n1[0][-1])+n1[1][-1])
#     n1[4].append(sin(n1[0][-1]*n1[1][-1]))
#     buff=[random.randint(scatter[0], scatter[1]), random.randint(scatter[0], scatter[1])]
#     n2[0].append(p2[0]+buff[0])
#     n2[1].append(p2[1]+buff[1])
#     n2[2].append(n2[0][-1]+n2[1][-1])
#     n2[3].append(log(n2[0][-1])+n2[1][-1])
#     n2[4].append(sin(n2[0][-1]*n2[1][-1]))
#     buff=[random.randint(scatter[0], scatter[1]), random.randint(scatter[0], scatter[1])]
#     n3[0].append(p3[0]+buff[0])
#     n3[1].append(p3[1]+buff[1])
#     n3[2].append(n3[0][-1]+n3[1][-1])
#     n3[3].append(log(n3[0][-1])+n3[1][-1])
#     n3[4].append(sin(n3[0][-1]*n3[1][-1]))



# for i in range(len(mass)):
#     mass[i]+=n1[i]+n2[i]+n3[i]
 
mass=np.array(mass)


plt.plot([x[0] for x in mass[:n]],[x[1] for x in mass[:n]], 'o')
plt.plot([x[0] for x in mass[n:2*n]],[x[1] for x in mass[n:2*n]], 'o')
plt.plot([x[0] for x in mass[2*n:3*n]],[x[1] for x in mass[2*n:3*n]], 'o')
# plt.scatter(*zip(*mass), s=5)
plt.show()

ms_std=(mass-mass.mean(axis=0))/mass.std(axis=0)

cov=(1/(ms_std.shape[0]-1))*(ms_std.T @ ms_std)

w,v=np.linalg.eigh(cov)

sorted_ind = np.argsort(w)[::-1]
v_sort = v[:, sorted_ind]
ms_new=ms_std @ v_sort[:,:2]

# mean=[x.mean() for x in mass]

# centered=[mass[i]-mean[i] for i in range(5)]

# covmat=np.cov(centered)

# plt.plot(ms_new[0][:5], ms_new[1][:5], 'o')
# plt.plot(ms_new[0][5:10], ms_new[1][5:10], 'o')
# plt.plot(ms_new[0][10:15], ms_new[1][10:15], 'o')

# kmeans=KMeans(n_clusters=3)
# kmeans.fit(ms_new[1])
# plt.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 'o')

plt.plot([x[0] for x in ms_new[:n]],[x[1] for x in ms_new[:n]], 'o')
plt.plot([x[0] for x in ms_new[n:2*n]],[x[1] for x in ms_new[n:2*n]], 'o')
plt.plot([x[0] for x in ms_new[2*n:3*n]],[x[1] for x in ms_new[2*n:3*n]], 'o')

plt.show()