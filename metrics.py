import numpy as np 


class confusion_matrix:
    def __init__(self, y_true, y_pred, labels=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
        self.K = len(labels) if labels else len(np.unique(y_true))
        self.cm = self.make_matrix()
        self.acc = self.accuracy(self.cm)
    
    def make_matrix(self):
        self.cm = np.zeros((self.K, self.K))
        for i in range(len(self.y_true)):
            self.cm[self.y_true[i], self.y_pred[i]] += 1
        return self.cm
    
    def accuracy(self, cm):
        numerator = np.sum([cm[0, 0], cm[1, 1]] )
        denominator = np.sum(cm)
        self.acc = numerator / denominator
        return self.acc
    
    def __repr__(self):
        return f"labels: {self.labels} \n\n {self.cm}  \n\n Accuracy: {self.acc}"

confusion_m = confusion_matrix([0,0,0,0], [0,0,0,1], labels=[0, 1])
print(confusion_m)
