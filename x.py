import numpy as np 


X = np.array([[1,2,3,4], [4,5,6,7], [7,8,9,10]])

idx = [0, 2]
idx_2 = [1, 3]
indices = X[idx][:, idx_2]