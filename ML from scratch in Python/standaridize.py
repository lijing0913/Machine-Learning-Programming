# Standardize a matrix in numpy
import numpy as np
import pandas as pd

# Standardize the entire matrix as a whole
# A = (A - np.mean(A)) / np.std(A)
# Standardize each column individually
# A = (A - np.mean(A, axis=0)) / np.std(A, axis=0)
'''
missing data is set to 0.
Steps:
1. replace each 0 with the mean of the non-zero values in its columns.
2. normalize the matrix such that:
    the mean value of each column is 0.
    the standard deviation of the values of each column is 1.
    in other words, change each element e in each column C to e-C_mean/C_std
'''
dataset = [[1, 2, 0],
           [0, 1, 1],
           [5, 6, 5]]
'''
solution:
[[-1.22474487 -0.46291005  0.        ]
 [ 0.         -0.9258201  -1.22474487]
 [ 1.22474487  1.38873015  1.22474487]]
'''
df = np.array(dataset)
# df = np.random.random(size=(3,3))
print('Original Array: ')
print(df)
# calculate the mean of each colum of non-zero elements
col_mean = df.sum(axis=0) / (df != 0).sum(axis=0)
# replace each 0 with the mean of the non-zero values in its columns
# find indices that you need to replace
idx = np.where(df == 0) # idx=(array([0, 1]), array([2, 0])), where the first array is row index and the second array is column index
df[idx] = np.take(col_mean, idx[1])
# standardize each column individually
df = (df - np.mean(df, axis=0)) / np.std(df, axis=0)
print('After standardization: ')
print(df)
