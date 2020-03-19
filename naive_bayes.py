from math import sqrt, exp, pi
from typing import Union
import numpy as np 


class GaussianNB:
    """
    Gausian Naive Bayes implementation.

    # Example: 
    
    ```python
        from sklearn.datasets import load_iris

        dataset = load_iris()
        X, y = dataset.data, dataset.target  

        model = GaussianNB()

        summaries = model.fit(X, y)
        label = model.predict(summaries, [5.7,2.9,4.2,1.3])
        print(label)

    @seperate_by_class: return dict with labels as keys and Xs associated with label as values
    @fit return dict with labels as keys and (mean, stdev, len(col)) as values
    @calculate_class_probabilities: return probability for each label
    """

    def __init__(self):
        pass

    def mean(self, numbers: Union[list, tuple]) -> float:
        return sum(numbers) / float(len(numbers))
    
    def stdev(self, numbers: Union[list, tuple]) -> float:
        avg = self.mean(numbers)
        variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
        return sqrt(variance)
    
    def separate_by_class(self, X: Union[list, tuple, np.array], y: Union[list, tuple, np.array]) -> dict:
        separated = dict()
        for xs, label in zip(X, y):
            if label not in separated:
                separated[label] = list()
            separated[label].append(xs)
        return separated
    
    def summarize_dataset(self, X: Union[list, tuple]) -> list:
        summaries = [(self.mean(col), self.stdev(col), len(col)) for col in zip(*X)]
        return summaries
    
    def fit(self, X: Union[list, tuple, np.array], y: Union[list, tuple, np.array]) -> dict:
        separated = self.separate_by_class(X, y)
        summaries = dict()
        for label, xs in separated.items():
            summaries[label] = self.summarize_dataset(xs)
        return summaries

    def calculate_probability(self, x: float, mean: float, stdev: float) -> float:
        exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    def calculate_class_probabilities(self, summaries: dict, X: Union[list, tuple]) -> dict:
        n_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for label, class_summaries in summaries.items():
            probabilities[label] = summaries[label][0][2] / float(n_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[label] *= self.calculate_probability(X[i], mean, stdev)
        return probabilities

    def predict(self, summaries: dict, X: list) -> int:
        probabilities = self.calculate_class_probabilities(summaries, X)
        best_label, best_prob = None, -1
        for label, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = label
        return best_label
