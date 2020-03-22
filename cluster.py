import matplotlib.pyplot as plt 
import numpy as np 


class KMeans:
    def __init__(self, k, tolerance=0.001, max_iter=300):
        self.k = k
        self.tolerance = tolerance
        self.max_iter = max_iter
    
    