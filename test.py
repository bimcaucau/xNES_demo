import numpy as np
x = np.array([1, 2]).reshape(1, -1)
w = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]).reshape(2, 8)
x = np.dot(x, w) + np.array([1,2,3,4,5,6,7,8])
print(x)
print(x.shape)