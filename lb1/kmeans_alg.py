from sklearn.cluster import KMeans
import numpy as np
from math import *

EPSILON = 0.1

def do_kmeans_with_module(dat:np.array, cluster_capasity:int):
    km=KMeans(n_clusters=cluster_capasity)
    km.fit(dat)
    return km.cluster_centers_

class Center:
    def __init__(self):
        self.pos=[0.,0.]
        self.sum_of_distances=[0.,0.]
        self.points_capasity=0.
        self.wcss=0.
        self.wcss_temp=0.
        self.prev_pos=[0.,0.]
        pass
    
    def make_new_pos(self):
        self.prev_pos=self.pos
        if self.points_capasity!=0:
            self.pos=[self.sum_of_distances[0]/self.points_capasity, 
                    self.sum_of_distances[1]/self.points_capasity]
            self.sum_of_distances=[0.,0.]
            self.points_capasity=0.
            self.wcss=self.wcss_temp
            self.wcss_temp=0.
        pass    
    
    def enough(self):
        return sqrt((self.prev_pos[0]-self.pos[0])**2 + (self.prev_pos[1]-self.pos[1])**2)<EPSILON
        
    

def do_kmeans(dat:np.array, cluster_capasity:int):
    x_min=min(x[0] for x in dat)
    y_min=min(y[1] for y in dat)
    x_max=max(x[0] for x in dat)
    y_max=max(y[1] for y in dat)
    centers=[]
    for i in range(cluster_capasity):
        centers.append(Center())
    for i in range(cluster_capasity):
        centers[i].pos=[np.random.uniform(x_min,x_max), np.random.uniform(y_min, y_max)]
    stop=False
    while not(stop):
        for point in dat:
            min_ind=0
            min_dist=[999999999999999999999.0, 999999999999999999999.0]
            p=[]
            for i in range(len(centers)):
                dist_diff=[abs(point[0]-centers[i].pos[0]), abs(point[1]-centers[i].pos[1])]
                dist=sqrt(dist_diff[0]**2 + dist_diff[1]**2)
                if dist<sqrt(min_dist[0]**2 + min_dist[1]**2):
                    min_dist=dist_diff
                    p=point
                    min_ind=i
            centers[min_ind].sum_of_distances[0]+=p[0]
            centers[min_ind].sum_of_distances[1]+=p[1]
            centers[min_ind].points_capasity+=1
            centers[min_ind].wcss_temp+=min_dist[0]**2 + min_dist[1]**2
        for i in range(cluster_capasity):
            centers[i].make_new_pos()
        # print([x.pos for x in centers])
        stop=all([x.enough() for x in centers])
    return [x.pos for x in centers], sum([x.wcss for x in centers])
    