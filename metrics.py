import numpy as np 


class confusion_matrix:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.K = len(np.unique(y_true))
    
    def make_matrix(self):
        self.cm = np.zeros((self.K, self.K))
        for i in range(len(self.y_true)):
            self.cm[self.y_true[i], self.y_pred[i]] += 1
        return self.cm
    
    def precision(self)


confusion_m = confusion_matrix([1,0,0,1], [0,0,0,1])
cm = confusion_m.make_matrix()
print(cm)
