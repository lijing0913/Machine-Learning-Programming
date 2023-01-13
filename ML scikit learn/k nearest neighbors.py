"""
KNN
given (height, weight) data points to preddict type
"""
from sklearn.neighbors import KNeighborsClassifier
X = [[150, 45], [160, 80], [165, 75], [175, 75], [175, 75]]
Y = ['Normal', 'Obese', 'Overweight', 'Normal', 'Normal']
k = 3

classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(X, Y)
print(classifier.predict([[180, 75]]))