from math import sqrt, exp, pi


class GaussianNB:
    def __init__(self):
        pass

    def mean(self, numbers):
        return sum(numbers) / float(len(numbers))
    
    def stdev(self, numbers):
        avg = self.mean(numbers)
        variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
        return sqrt(variance)
    
    def separate_by_class(self, X, y):
        separated = dict()
        for xs, label in zip(X, y):
            if label not in separated:
                separated[label] = list()
            separated[label].append(xs)
        return separated
    
    def summarize_dataset(self, X):
        summaries = [(self.mean(col), self.stdev(col), len(col)) for col in zip(*X)]
        return summaries
    
    def summarize_by_class(self, X, y):
        separated = self.separate_by_class(X, y)
        summaries = dict()
        for label, xs in separated.items():
            summaries[label] = self.summarize_dataset(xs)
        return summaries

    def calculate_probability(self, x, mean, stdev):
	    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	    return (1 / (sqrt(2 * pi) * stdev)) * exponent

    def calculate_class_probabilities(self, summaries, X):
        n_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for label, class_summaries in summaries.items():
            probabilities[label] = summaries[label][0][2] / float(n_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[label] *= self.calculate_probability(X[i], mean, stdev)
        return probabilities
    


gnb = GaussianNB()

l = [[3.393533211,2.331273381],
	[3.110073483,1.781539638],
	[1.343808831,3.368360954],
	[3.582294042,4.67917911],
	[2.280362439,2.866990263],
	[7.423436942,4.696522875],
	[5.745051997,3.533989803],
	[9.172168622,2.511101045],
	[7.792783481,3.424088941],
	[7.939820817,0.791637231]]
m = [0,0,0,0,0,1,1,1,1,1]
#summ = gnb.summarize_dataset(l)

separated = gnb.summarize_by_class(l, m)
probabilities = gnb.calculate_class_probabilities(separated, l[0])
print(probabilities)

"""
print("\n", summ)
summaries = gnb.summarize_by_class(l, m)
print("\n", summaries)
"""
