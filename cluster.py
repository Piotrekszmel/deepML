import matplotlib.pyplot as plt 
import numpy as np
import random

random.seed(42)

class KMeans:
    def __init__(self, k, tolerance=0.001, max_iter=1):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter
    
    def fit(self, data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = random.choice(data)

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []
            
            for coordinates in data:
                distances = [np.linalg.norm(coordinates - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(coordinates)
            
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.mean(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                prev_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    print(np.sum((current_centroid-original_centroid) / original_centroid * 100.0))
                    optimized = False  


kmeans = KMeans(k=2)
kmeans.fit([1, 2, 3])
print(kmeans.centroids)
