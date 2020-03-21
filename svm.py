import matplotlib.pyplot as plt 
import numpy as np 


class SVM:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: "r", -1: "b"}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        opt_dict = {}

        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]

        data = np.array([])

        for xi in self.X:
            data = np.append(data, xi)
        
        self.max_feature_value = max(data)
        self.min_feature_value = min(data)
        data = None

        step_sizes = [self.max_feature_value * 0.01,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]
        
        b_range_multiple = 5
        b_multiple = 5

        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])

            optimized = False
            while not optimized:
                for b in np.arange(-1 * self.max_feature_value * b_range_multiple,
                                   self.max_feature_value * b_range_multiplem,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True

                        for i in self.y:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi)) >= 1:
                                    found_option = False
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                
                if w[0] < 0:
                    optimized = True
                    print("optimized a step!")
                else:
                    w = w - step
        
        norms = sorted([n for n in opt_dict])
        opt_choice = opt_dict[norms[0]]

        self.w = opt_choice[0]
        self.b = opt_choice[1]

        latest_optimum = opt_choice[0][0] + step * 2
    
    def predict(self, features):
        classification = np.sign(np.dot(np.array(features),self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0],features[1],s=200,marker='*',c=self.colors[classification])
        return (classification,np.dot(np.array(features),self.w)+self.b)
