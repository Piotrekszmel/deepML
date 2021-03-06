import numpy as np
from typing import Tuple


class Node: 
    """
    Node class used in DecisionTreeClassifier
    
    @param: predicted_class (int) : predicted class for given node
    """

    def __init__(self, predicted_class: int) -> None:
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

    
class DecisionTreeClassifier:
    """
    Decision Tree Classifier used for classification tasks. 

    # Example: 
    
    ```python
        import sys
        from sklearn.datasets import load_iris

        dataset = load_iris()
        X, y = dataset.data, dataset.target  # pylint: disable=no-member
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(X, y)
        print(clf.predict([[5.0, 3.6, 1.3, 0.3]]))

    
    @param: max_depth (int) : max depth of tree

    @predict: return predicted labels for given inputs
    @_best_split: return idx and threshold for the best split
    @_grow_tree: build tree
    """

    def __init__(self, max_depth: int = None, X=None, y=None) -> None:
        """
        X and y are used in RandomForest class. Do not use it for DecisionTreeClassifier
        """

        self.max_depth = max_depth
        self._X = X
        self._y = y

    def fit(self, X: np.array, y: np.array) -> None:
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X: np.array) -> list:
        return [self._predict(inputs) for inputs in X]
    
    def _best_split(self, X: np.array, y: np.array) -> Tuple[int, np.float64]:
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes)
                    )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes)
                    )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
                    
        return best_idx, best_thr
    
    def _grow_tree(self, X: np.array, y: np.array, depth: int = 0) -> Node:
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs: list) -> np.int64:
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
    
    def __repr__(self):
        return (f"Parameters: \n max_depth: {self.max_depth} \n\n")


class RandomForestClassifier:
    """
    Random Forest Classifier using DecisionTreeClassifier. 

    # Example: 
    
    ```python
        from sklearn.datasets import load_iris

        dataset = load_iris()
        X, y = dataset.data, dataset.target  

        forest = RandomForestClassifier(X, y, 10, 2, 50, depth=1)
        print(forest.predict([[5.0, 3.6, 1.3, 0.3]]))

    
    @param: X (np.array) : featuers
    @param: y (np.array) : labels
    @param: n_trees (np.array) : number of trees in forest
    @param: n_features (np.array) : numbers of featuers used in decision tree
    @param: sample_size (np.array) : X and y size for decision tree fit method
    @param: depth (int) : max depth of decision tree

    @_create_tree: return list of DecisionTreeClassifier objects
    @_predict: return predictions based on given X
    """
    
    def __init__(self, X: np.array, y: np.array, n_trees: int, n_features: int, sample_size: int, depth: int = 5) -> None:
        self.X = X
        self.y = y
        self.n_trees = n_trees
        self.n_features = n_features
        self.sample_size = sample_size
        self.depth = depth
        self.trees = [self._create_tree() for i in range(n_trees)]
        self.fit()
        
    def _create_tree(self):
        idxs = list(np.random.permutation(len(self.y))[:self.sample_size])
        feature_idxs = list(np.random.permutation(self.X.shape[1])[:self.n_features])
        return DecisionTreeClassifier(self.depth, self.X[idxs][:, feature_idxs], self.y[idxs])
    
    def fit(self) -> None:
        for tree in self.trees:
            tree.fit(tree._X, tree._y)

    def predict(self, X: list) -> list:
        predictions = []
        for x in X:
            p = [t.predict([x]) for t in self.trees]
            p = [pred for l in p for pred in l]
            predictions.append(np.bincount(p).argmax())
        return predictions
    
    def __repr__(self):
        return (f"Parameters: \n n_trees: {self.n_trees} \n"
                f" n_features: {self.n_features} \n"
                f" sample_size: {self.sample_size} \n"
                f" depth: {self.depth} \n")
