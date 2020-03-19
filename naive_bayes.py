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
    