import matplotlib.pyplot as plt 
import numpy as np 


class SVM:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: "r", -1: "b"}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
        
