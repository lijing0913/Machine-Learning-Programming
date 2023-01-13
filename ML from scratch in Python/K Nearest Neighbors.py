## K-Nearest Neighbors

from importlib.metadata import distribution
from pickletools import int4
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import Counter
import time

class KNN:
    def __init__(self):
        self.accurate_predictions = 0
        self.total_predictions = 0
        self.accuracy = 0.0 # self.accuray = self.accurate_predictions / self.total_predictions * 100
    
    def predict(self, training_data: dict, to_predict: list, k=3):
        if len(training_data) >= k:
            print("K cannot be smaller than the number of training data points")
            return

        distributions = []
        for group in training_data:
            for features in training_data[group]:
                euclidean_distance = np.linalg.norm(np.array(features) - np.array(to_predict)) # L2 norm
                distributions.append([euclidean_distance, group])
        
        results = [i[1] for i in sorted(distributions[:k])] # get the closest k points' category
        res = Counter(results).most_common()[0][0] # get the most-frequent category, which is the predicted category for to_predict
        confidence = Counter(results).most_common()[0][1] / k
        return res, confidence
    
    def test(self, test_set: dict, training_set: dict):
        for group in test_set:
            for data in test_set[group]:
                predicted_class, confidence = self.predict(training_set, data, k=3)
                if predicted_class == group: # if predicted category is equal to its real oone
                    self.accurate_predictions += 1
                else:
                    print("Wrong classificatioon with confidence " + str(confidence * 100) + " and class" + str(predicted_class))
                self.total_predictions += 1
        self.accuracy = self.accurate_predictions / self.total_predictions * 100
        print("\nAccuracy: ", str(self.accuracy) + "%")

def mod_data(df: pd.DataFrame):
    df.replace('?', -999999, inplace = True) # inplace=True means replacing is done on the current DataFrame

    df.replace('yes', 4, inplace = True)
    df.replace('no', 2, inplace = True)

    df.replace('notpresent', 4, inplace = True)
    df.replace('present', 2, inplace = True)

    df.replace('abnormal', 4, inplace = True)
    df.replace('normal', 2, inplace = True)

    df.replace('poor', 4, inplace = True)
    df.replace('good', 2, inplace = True)

    df.replace('ckd', 4, inplace = True)
    df.replace('notckd', 2, inplace = True)

if __name__ == "__main__":
    # Read data
    df = pd.read_csv(r"./data/chronic_kidney_disease.csv")
    print(df.head())
    print(df.shape)
    # Encode categorical feature
    mod_data(df)
    dataset = df.astype(float).values.tolist() # convert string to float

    # Normalize the data
    x = df.values # return a numpy array
    x_scaled = preprocessing.MinMaxScaler().fit_transform(x)
    # x_scaled = preprocessing.StandardScaler().fit_transform(x)
    # x_scaled = preprocessing.scale(x)
    df = pd.DataFrame(x_scaled)

    # Shuffle the dataset
    rd.shuffle(dataset)

    # 20% of the available data will be used for testing
    test_size = 0.2

    # The keys of the dict are the classes that the data is classified into
    training_set = {2: [], 4: []}
    test_set = {2: [], 4: []}

    # Split data into training and test data for cross validation
    training_data = dataset[:-int(test_size * len(dataset))]
    test_data = dataset[-int(test_size * len(dataset)):]

    # Insert data into the training set
    for record in training_data:
        training_set[record[-1]].append(record[:-1]) # append the list in the dict with all the elements of the record except the class; the last element is class
    
    # Insert data into the test set
    for record in test_data:
        test_set[record[-1]].append(record[:-1])
    
    start = time.time()
    knn = KNN()
    knn.test(test_set, training_set)
    end = time.time()
    print("Execution Time: ", end - start)


