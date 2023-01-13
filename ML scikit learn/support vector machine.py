"""Support Vector Machine model"""
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Use the standard iris dataset
iris = datasets.load_iris()

# Split the dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1)

# Build the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Test on the testing dataset
y_predicted = model.predict(X_test)
print(classification_report(y_test,  y_predicted))
