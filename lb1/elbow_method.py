import numpy as np
from kmeans_alg import *


def do_elbow_with_module(dat:np.array):
    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(dat)
        inertia.append(kmeans.inertia_)
    
    return inertia


def do_elbow(dat:np.array, range_of_sclusters:list):
    inertia = []
    for k in range_of_sclusters:
        clst, wcll=do_kmeans(dat, k)
        inertia.append(wcll)
    
    return inertia