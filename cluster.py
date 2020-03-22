import matplotlib.pyplot as plt 
import numpy as np
import random

random.seed(42)

class KMeans:
    """
    K-means clustering method implementation

    # Example: 
    
    ```python
        X = np.array([[1, 2],
                [1.5, 1.8],
                [5, 8 ],
                [8, 8],
                [1, 0.6],
                [9,11]])

        colors = 10*["g","r","c","b","k"]   

        unknowns = np.array([[1,3],
                            [8,9],
                            [0,3],
                            [5,4],
                            [6,4],])

        clf = KMeans()
        clf.fit(X)
        clf.visualize(clf.centroids, clf.classifications, unknowns)
    """

    def __init__(self, k: int = 2, tolerance: float = 0.001, max_iter: int = 1) -> None:
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter
    
    def fit(self, data: np.array) -> None:
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
                print(np.sum((current_centroid - prev_centroid) / prev_centroid * 100.0))
                if np.sum((current_centroid - prev_centroid) / prev_centroid * 100.0) > self.tolerance:
                    print(np.sum((current_centroid - prev_centroid) / prev_centroid * 100.0))
                    optimized = False  

    def predict(self, data: np.array) -> int:
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
    def visualize(self, centroids: dict, classifications: dict, unknows: np.array = None) -> None:
        for centroid in centroids:
            plt.scatter(centroids[centroid][0], centroids[centroid][1],
            marker="o", color="k", s=150, linewidths=5)

        for classification in classifications:
            color = colors[classification]
            for featureset in classifications[classification]:
                plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

        if unknowns is not None:
            for unknown in unknowns:
                classification = self.predict(unknown)
                plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)
        
        plt.show()
