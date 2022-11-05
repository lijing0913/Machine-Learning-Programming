# Decision Tree 
# Ref: https://towardsdatascience.com/decision-tree-algorithm-in-python-from-scratch-8c43f0e40173

import numpy as np
import pandas as pd
from collections import Counter

# Importing the scikit-learn tree implementation
from sklearn.tree import DecisionTreeClassifier, export_text , DecisionTreeRegressor

class Node:
    """
    Class for creating the nodes for a decision tree
    """
    def __init__(
        self, 
        Y: list, 
        X: pd.DataFrame, 
        min_samples_split=20,
        max_depth=5,
        depth=0,
        node_type='root',
        rule=""
    ):
        # Save the data to node
        self.Y = Y
        self.X = X

        # Save the hyper parameters
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        # Default current depth of node
        self.depth = depth

        # Extract all the features
        self.features = list(self.X.columns)

        # Type of node
        self.node_type = node_type

        # Rule for splitting
        self.rule = rule

        # Calculate counts of Y in the node
        self.counts = Counter(Y)

        # Get the Gini impurity based on the Y distribution
        self.gini_impurity = self.get_gini()

        # Sort the counts and save the final prediction of the node
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        # Get the last item
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]
        
        # Save to object attribute. This node will predict the class with the most frequent class
        self.yhat = yhat

        # Save the number of observations in the node
        self.n = len(Y)

        # Initiate the left and right nodes as empty nodes
        self.left = None
        self.right = None

        # Default values for splits
        self.best_feature = None
        self.best_value = None

    def Gini_impurity(self, y1_count: int, y2_count: int) -> float:
        """
        Given the observations of a binary class, calculate the Gini impurity
        gini = sum(p*(1-p)) = 1 - sum(p**2)
        """

        # Ensure the correct types
        if y1_count is None:
            y1_count = 0
        if y2_count is None:
            y2_count = 0
        
        # Get the total observations
        n = y1_count + y2_count

        # If n is 0 then we return the lowest possiblee gini impurity
        if n == 0:
            return 0.0
        
        # Get the probability of each class
        p1 = y1_count / n
        p2 = y2_count / n

        # Calculate Gini impurity
        gini = 1 - (p1 **2 + p2 ** 2)

        # Return the Gini impurity
        return gini

    def get_gini(self) -> float: 
        """
        Calculate the Gini impurity of a node
        """
        # Get the 0 and 1 counts
        y1_count, y2_count = self.counts.get(0, 0), self.counts.get(1, 0)

        # Get the Gini impurity
        return self.Gini_impurity(y1_count, y2_count)
   
    def ma(self, x: np.array, window: int) -> np.array: #####
        """
        Calculates the moving average of the given list
        """
        return np.convolve(x, np.ones(window), 'valid') / window
    
    def best_split(self) -> tuple:
        """
        Given the X features and Y targets, calculate the best split for a decision tree
        How does the algorithm search for the best split for each feature?
    - For each feature, sort the feature values and get the mean of two neighboring values.
    - Check the Gini gain with each of the values from the above vector. Do this for all features.
    - The final split value and split feature is the one that has the highest Gini gain.
        """
        # Create a dataset for splitting
        df = self.X.copy()
        df['Y'] = self.Y

        # Get the Gini impurity for the base input
        Gini_base = self.get_gini()

        # Find which split yields the best Gini gain
        max_gain = 0

        # Default best feature and split
        best_feature = None
        best_value = None

        for feature in self.features:
            # Drop missing values
            Xdf = df.dropna().sort_values(feature)

            # Sort the values and get the rolling average
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                # Split theee dataset
                left_counts = Counter(Xdf[Xdf[feature] < value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature] >= value]['Y'])

                # Get the Y distribution from the dicts
                y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0, 0), right_counts.get(1, 0)

                # Get the left and right gini impurities
                gini_left = self.Gini_impurity(y0_left, y1_left)
                gini_right = self.Gini_impurity(y0_right, y1_right)

                # Get the number of counts from the left and right data splits
                n_left = y0_left + y1_left
                n_right = y0_right + y1_right

                # Calculate weights for each of the nodes
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                # Calculate the weighted Gini impurity
                wGini = w_left * gini_left + w_right * gini_right
                
                # Calculate the Gini gain
                Gini_gain = Gini_base - wGini 

                # Check if this is the best split so far
                if Gini_gain > max_gain:
                    best_feature = feature
                    best_value = value
                    # Update the best gain so far
                    max_gain = Gini_gain
        return (best_feature, best_value)
    
    def grow_tree(self):
        """
        Recursive method to create teh decision tree
        """
        # Make a df from the data
        df = self.X.copy()
        df['Y'] = self.Y

        # If there is a Gini to be gained, we split further
        if self.depth < self.max_depth and self.n >= self.min_samples_split:
            # Get the best split
            best_feature, best_value = self.best_split()

            if best_feature is not None:
                # Save the best split to the current node
                self.best_feature = best_feature
                self.best_value = best_value

                # Get the left and right nodes
                left_df = df[df[self.best_feature] <= self.best_value].copy()
                right_df = df[df[self.best_feature] > self.best_value].copy()

                # Create the left and right nodes
                left = Node(
                    left_df['Y'].values.tolist(),
                    left_df[self.features],
                    depth = self.depth + 1,
                    max_depth = self.max_depth,
                    min_samples_split = self.min_samples_split,
                    node_type = 'left_node',
                    rule = f"{self.best_feature} <= {round(self.best_value, 3)}"
                )
                self.left = left
                self.left.grow_tree()

                right = Node(
                    right_df['Y'].values.tolist(),
                    right_df[self.features],
                    depth = self.depth + 1,
                    max_depth = self.max_depth,
                    min_samples_split = self.min_samples_split,
                    node_type = 'right_node',
                    rule = f"{self.best_feature} > {round(self.best_value, 3)}"
                )
                self.right = right
                self.right.grow_tree()
    
    def print_info(self, width=4):
        """
        Method to print the information about the tree
        """
        # Define the number of spaces
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const

        if self.node_type == 'root':
            print('Root')
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | Gini impurity of the node: {round(self.gini_impurity, 2)}")
        print(f"{' ' * const}   | Class distribution in the node: {dict(self.counts)}")
    
    def print_tree(self):
        """
        Print the whole tree from the current node to the bottom
        """
        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()

    def predict(self, X:pd.DataFrame):   
        """
        Batch prediction method
        """
        predictions = []
        for _, x in X.iterrows(): # iterate over a pandas Data frame rows in the form of (index, series) pair
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]}) # updates the dictionary with the elements from another dictionary object
            predictions.append(self.predict_class(values))
        return predictions
    
    def predict_class(self, values: dict) -> int: 
        """
        Method to predict the class given a set of features
        """
        curr_node = self
        while curr_node.depth < curr_node.max_depth:
            # Traverse the nodes all the way to the bottom
            best_feature = curr_node.best_feature
            best_value = curr_node.best_value
            
            if curr_node.n < curr_node.min_samples_split:
                break

            if values.get(best_feature) < best_value:
                if self.left is not None:
                    curr_node = curr_node.left
            else:
                if self.right is not None:
                    curr_node = curr_node.right
        return curr_node.yhat

if __name__ == '__main__':
    # Implement
    # Load data
    data = pd.read_csv('data/titanic_survivor/train.csv')

    # Drop missing values
    data = data[['Survived', 'Age', 'Fare']].dropna().copy()

    # Define X and Y
    Y = data['Survived'].values.tolist() # list
    X = data[['Age', 'Fare']]

    # Save the feature list
    features = list(X.columns)

    # Define the dictionary of hyperparameters
    hp = {
        'max_depth': 3,
        'min_samples_split': 100
    }

    # Initiate the root node:
    root = Node(Y, X, **hp)
    
    # Get the best split
    root.grow_tree()

    # Print the tree information
    root.print_tree()

    # Predict
    Xsubset = X.copy()
    Xsubset['yhat'] = root.predict(Xsubset)
    print(Xsubset)


    # # Scikit-learn comparision ----------
    # # Initiate the Node
    # root = Node(Y, X, **hp)

    # # Get the best split
    # root.grow_tree()

    # # Use the ML package
    # clf = DecisionTreeClassifier(**hp)
    # clf.fit(X, Y)

    # # View the results
    # root.print_tree()





        

    