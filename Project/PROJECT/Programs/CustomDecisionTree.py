import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class Node:
    """
    Initialise a node in the decision tree.

    Parameters:
    - predicted_class: The predicted class for the node.
    """
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class CustomDecisionTreeClassifier:
    def __init__(self, max_depth=None):
        """
        Initialise the custom decision tree classifier.

        Parameters:
        - max_depth: THe maximum depth of the tree. If None, the tree is grown until all leaves are pure.
        """
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        Fit method to train the Decision Tree. Counts the unique classes and the number of features.

        Parameters:
        - X: Input features.
        - y: Target labels.
        """
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1] 
        self.tree_ = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursive method to grow the decision tree.

        Parameters:
        - X: Input features.
        - y: Target labels.
        - depth: Current depth of the tree.

        Returns:
        - A node representing the root of the subtree.
        """
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)

        node = Node(predicted_class=predicted_class) # Creates node for predicted class

        if self.max_depth is not None and depth < self.max_depth:
            feature_idxs = np.random.choice(self.n_features_, self.n_features_, replace=False) # Randomly select features
            best_idx, best_thr = self._best_split(X, y, feature_idxs) # Find the best split
            if best_idx is not None:
                indices_left = X[:, best_idx] < best_thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = best_idx
                node.threshold = best_thr
                node.left = self._grow_tree(X_left, y_left, depth + 1) # Build left tree
                node.right = self._grow_tree(X_right, y_right, depth + 1) # Build right tree
        return node

    def _best_split(self, X, y, feature_idxs):
        """
        Method to find the best split based on Gini impurity

        Parameters:
        - X: Input features.
        - y: Target labels.
        - feature_idxs: Indices of features to consider.

        Returns:
        - best_idx: Index of the best feature for splitting.
        - best_thr: Threshold value for the best split
        """
        best_idx, best_thr = None, None
        best_gini = 1.0

        for idx in feature_idxs:
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                indices_left = X[:, idx] < thr
                gini = self._gini_impurity(y[indices_left], y[~indices_left])
                if gini < best_gini:
                    best_idx = idx
                    best_thr = thr
                    best_gini = gini
        return best_idx, best_thr

    def _gini_impurity(self, left_labels, right_labels):
        """
        Gini impurity is a measure in decision tree algorithms to quantify a dataset's impurity level or disorder.
        
        Parameters:
        - left_labels: Labels in the left subset.
        - right_labels: Labels in the right subset.

        Returns:
        - Gini impurity value.
        """
        p_left = len(left_labels) / (len(left_labels) + len(right_labels))
        p_right = len(right_labels) / (len(left_labels) + len(right_labels))
        gini_left = 1.0 - sum((np.sum(left_labels == c) / len(left_labels)) ** 2 for c in np.unique(left_labels))
        gini_right = 1.0 - sum((np.sum(right_labels == c) / len(right_labels)) ** 2 for c in np.unique(right_labels))
        gini = p_left * gini_left + p_right * gini_right
        return gini
    
    def predict(self, X):
        """
        Make predictions using the trained decision tree.

        Parameters:
        - X: Input features.

        Return:
        - Predicted class labels.
        """
        return [self._predict_tree(x, self.tree_) for x in X]
    
    def _predict_tree(self, x, node):
        """
        Recursive method to predict the class for a given input using the decision tree.

        Parameters:
        - x: Input feature.
        - node: Current node in the decision tree.

        Returns:
        - Predicted class label.
        """
        if node.left is None and node.right is None:
            return node.predicted_class
        if x[node.feature_index] < node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)

r'''
# Load and process data
file_path = r'C:\Users\janit\Desktop\Project\PROJECT\heart.csv'
heart_data = pd.read_csv(file_path)

# Separate features from target variable
heart_x = heart_data.drop("HeartDisease", axis=1)
heart_y = heart_data['HeartDisease']

# Replace categorical data with "dummy" variables through one-hot-encoding
heart_x_encoded = pd.get_dummies(heart_x, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(heart_x_encoded, heart_y, test_size=0.3, random_state=42)

# Build and use the custom tree
custom_decision_tree = CustomDecisionTreeClassifier(max_depth=2)
custom_decision_tree.fit(X_train.values, y_train.values)

train_predictions = custom_decision_tree.predict(X_train.values)
test_predictions = custom_decision_tree.predict(X_test.values)
train_acc = accuracy_score(y_train, train_predictions)
test_acc = accuracy_score(y_test, test_predictions)
print("Training accuracy =", train_acc)
print("Test accuracy =", test_acc)

# Evaluate the model and generate classification reports

train_report = classification_report(y_train, train_predictions)
test_report = classification_report(y_test, test_predictions)

# Print classification reports
print("Training Classification Report: ")
print(train_report)

print("\nTest Classification Report: ")
print(test_report)
'''