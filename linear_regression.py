# Linear Regression: 1D
import numpy as np
import matplotlib.pyplot as plt
class LinearRegressor:
    def __init__(self):
        self.beta0 = 0
        self.beta1 = 0
    def SS(self, X, Y):
        return np.sum((X - Y) ** 2)
    def fit(self, X, Y):
        self.beta1 = np.sum((X - np.mean(X))*(Y - np.mean(Y))) / np.sum((X - np.mean(X)) ** 2)
        self.beta0 = np.mean(Y) - self.beta1 * np.mean(X)
    def Rsquare(self, Y, Y_hat):
        return 1 - self.SS(Y, Y_hat) / self.SS(Y, np.mean(Y))
    def predict(self, X):
        return self.beta1 * X + self.beta0

X = np.linspace(0, 10, 10)
beta1, beta0 = 3, -2
Y = beta1 * X + beta0 + np.random.randn(X.shape[0])

lr = LinearRegressor()
lr.fit(X, Y)
Y_predict = lr.predict(X)
R2 = lr.Rsquare(Y, Y_predict)
print(lr.beta1, lr.beta0, R2)
plt.scatter(Y_predict, Y)
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.show()



# # Linear Regression Example: House Price Prediction
# # build and train a linear regression model
# from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# lr.fit(df_train, y_train)
# plt.scatter(lr.predict(df_train), y_train)
# plt.xlabel('y_pred')
# plt.ylabel('y_true')
# plt.show()