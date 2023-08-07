import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

class KMeans:
    def __init__(self, k, X, y, threshold = 100):
        self.k = k
        self.X = X
        self.y = y
        self.threshold = threshold

    def assign_center(self):
        centers = list()
        for i in range(self.k):
            centers.append(self.X[np.random.randint(self.X.shape[0], size = 1)])
        return centers

    def assign_clusters(self, centers):
        clusters = dict()
        for i in range(self.k):
            clusters[i] = list()
            
        for itr, x in enumerate(self.X):
            distance_list = list()
            for j in range(self.k):
                dist = np.linalg.norm(x - centers[j])
                distance_list.append(dist)
            clusters[distance_list.index(min(distance_list))].append(x)
        return clusters
    
    def calculate_centers(self, clusters, centers):
        for i in range(self.k):
            centers[i] = np.average(clusters[i], axis = 0)
        return centers

    def calculate_scalar(self, prev_centers, new_centers):
        total = 0
        for i in range(self.k):
            total += np.linalg.norm(prev_centers[i] - new_centers[i])
        return total/k

    def fit(self):
        #Initializing centroids
        centers = self.assign_center()
        #KMeans 
        scalar_product = 10e7
        while scalar_product > self.threshold:
            previous_centers = centers.copy()
            clusters = self.assign_clusters(centers)
            centers = self.calculate_centers(clusters, centers)
            scalar_product = self.calculate_scalar(previous_centers, centers)
            print(scalar_product)
        return centers