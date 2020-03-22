import matplotlib.pyplot as plt 
import numpy as np
import random

random.seed(42)

class KMeans:
    def __init__(self, k=2, tolerance=0.001, max_iter=1):
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
                if np.sum((current_centroid - prev_centroid) / prev_centroid * 100.0) > self.tolerance:
                    print(np.sum((current_centroid - prev_centroid) / prev_centroid * 100.0))
                    optimized = False  

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        print("\n classification: ", classification)
        return classification


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])

colors = 10*["g","r","c","b","k"]   

clf = KMeans()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)
        


unknowns = np.array([[1,3],
                     [8,9],
                     [0,3],
                     [5,4],
                     [6,4],])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)


plt.show()
