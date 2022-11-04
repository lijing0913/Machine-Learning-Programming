# Normalize a matrix in numpy
# Step 1- Import the library
import numpy as np
import pandas as pd
# Step 2 - Setup the data
df = np.random.random(size=(3,3))
print('Original Array: ')
print(df)
# Step 3 - Perform normalization: s
# substract each element by minimum value of matrix and divide the whole with difference of minimum and maximum of whole matrix
dfmax, dfmin = df.max(), df.min()
df = (df - dfmin) / (dfmax - dfmin)
print('After normalization: ')
print(df)
df = pd.DataFrame(df)
print(df.describe())