import numpy as np 


class confusion_matrix:
    def __init__(self, y_true, y_pred, labels=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
        self.K = len(np.unique(y_true)) if labels == None else len(labels)
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
    

confusion_m = confusion_matrix([1,0,0,0], [0,0,0,1], labels=[True,False])
print(confusion_m.cm)
print()
print(confusion_m.acc)
print(confusion_m.labels)
