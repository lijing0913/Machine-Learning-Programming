"""Decision Trees"""
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import datasets
[iris_data, iris_target] = datasets.load_iris(return_X_y=True)

dtree = DecisionTreeClassifier()
dtree.fit(iris_data, iris_target)
plt.figure()
plot_tree(decision_tree=dtree)
plt.show()