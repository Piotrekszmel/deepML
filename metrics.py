import numpy as np 
from typing import Union

class confusion_matrix:
    """
    Create confusion matrix for given labels

    # Example: 
    
    ```python
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    @param: y_true (np.array) : true labels 
    @param: y_pred (np.array) : predicted labels
    @labels: Union[list, tuple] : all unique values from y_true
    
    @make_matrix: return confusion_matrix
    @accuracy: return accuracy for given confusion_matrix
    @recall: return recall for given confusion_matrix
    @precision: return precision for given confusion_matrix
    """
    
    def __init__(self, y_true, y_pred, labels=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
        self.K = len(labels) if labels else len(np.unique(y_true))
        self.cm = self.make_matrix()
        self.acc = self.accuracy(self.cm)
        self.rec = self.recall(self.cm)
        self.prec = self.precision(self.cm)

    def make_matrix(self):
        self.cm = np.zeros((self.K, self.K))
        for i in range(len(self.y_true)):
            self.cm[int(self.y_true[i]), int(self.y_pred[i])] += 1
        return self.cm
    
    def accuracy(self, cm):
        numerator = np.sum([cm[0, 0], cm[1, 1]] )
        denominator = np.sum(cm)
        self.acc = numerator / denominator
        return self.acc

    def recall(self, cm):
        numerator = cm[1, 1]
        denominator = np.sum([cm[1, 0], cm[1, 1]])
        if denominator == 0:
            self.rec = 0.0
            return self.rec
        self.rec = numerator / denominator
        return self.rec
    
    def precision(self, cm):
        numerator = cm[1, 1]
        denominator = np.sum([cm[0, 1], cm[1, 1]])
        if denominator == 0:
                self.prec = 0.0
                return self.prec
        self.prec = numerator / denominator
        return self.prec
    
    def __repr__(self):
        return (f" labels: {self.labels} \n\n matrix: \n {self.cm} \n\n accuracy: {self.acc} \n" + 
        f" recall: {self.rec} \n precision: {self.prec} \n")
