"""Sum of two numbers"""
from statistics import linear_regression
from sklearn import linear_model
import numpy as np

# Create training dataset
input_data = np.random.randint(50, size=(20, 2)) # generate random integer in range [0, 50]
input_sum = np.zeros(len(input_data))
for i in range(len(input_data)):
    input_sum[i] = input_data[i][0] + input_data[i][1]

# Build a Linear Regression model
linear_regression_model = linear_model.LinearRegression(fit_intercept=False)
linear_regression_model.fit(input_data, input_sum)

# Test on the the new data
predicted_sum = linear_regression_model.predict([[60, 24]])
print('Predicted sum of 60 and 24 is' + str(predicted_sum))

# Check the regression coefficients
# the model should have 1, 1 as the coefficients
print('Coefficients of linear regression model are' + str(linear_regression_model.coef_))
