import numpy as np 


class GaussianNB:
    def __init__(self):
        pass

    def mean(self, numbers):
        return np.sum(numbers) / float(len(numbers))
    
    def stdev(self, numbers):
        avg = self.mean(numbers)
        variance = np.sum([(x - avg)**2 for x in numbers]) / float(len(numbers - 1))
        return np.sqrt(variance)
    
    def separate_by_class(self, X, y):
        separated = dict()
        for xs, label in zip(X, y):
            if label not in separated:
                separated[label] = list()
            separated[label].append(xs)
        return separated
    
    def summarize_dataset(self, X):
        summaries = [(self.mean(col), self.stdev(col), len(col)) for col in X[:]]
        return summaries
    
    def summarize_by_class(self, X, y):
        separated = self.separate_by_class(X, y)
        summaries = dict()
        for label, xs in separated.items():
            summaries[label] = self.summarize_dataset(xs)
        return summaries
    
gnb = GaussianNB()
X, y = np.array([[1,2], [3,4], [5,6], [7,8]]), [1, 2, 3, 1]
separated = gnb.separate_by_class(X, y)
print(separated)
summ = gnb.summarize_dataset(X)
print("\n", summ)
summaries = gnb.summarize_by_class(X, y)
print("\n", summaries)