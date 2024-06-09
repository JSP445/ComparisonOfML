import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class KNNClassifier:
    def __init__(self, k=3):
        """
        Initialise the K-Nearest Neighbours classifier.

        Parameters:
        - k: Number of neighbours to consider during classification.
        """
        self.k = k
    
    def fit(self, X, y):
        """
        Fit method to store the training data.

        Parameters:
        - X: Input features.
        - y: Target labels.
        """
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        """
        Calculate the Euclidean distance between two data points.
        
        Parameters:
        - x1: First data point.
        - x2: Second data point.

        Returns:
        - Euclidean distance between x1 and x2.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """
        Make predictions using the KNN classifier.

        Parameters:
        - X: Input features for prediction.
        
        Returns:
        - Predicted class labels for each input.
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        """
        Predict the class label for a single data point.
        
        Parameters:
        - x: Input data point.

        Returns:
        - Predicted class label based on KNN algorithm.
        """
        # Compute distances between the input data point and all training data
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]

        # Sort by distance and return indices of the first k neighbours
        k_indices = np.argsort(distances)[:self.k]

        # Extract the labels of the k nearest neighbour training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common class label among the k nearest neighbours

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


r'''
# Import data
file_path = r'C:\Users\janit\Desktop\Project\PROJECT\heart.csv'
    
# Test

df = pd.read_csv(file_path)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode categorical variables
label_encoder = LabelEncoder()
for i in range(X.shape[1]):
    if isinstance(X[0, i], str):
        X[:, i] = label_encoder.fit_transform(X[:, i])

# Convert to numeric type
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)

# Training set
y_train_pred = knn.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f'Training Accuracy = {train_accuracy}')

# Generate training classification report
train_class_report = classification_report(y_train, y_train_pred)
print("Training Classification Report: ")
print(train_class_report)

# Test set 
y_test_pred = knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy = {test_accuracy}')

# Generate test classification report
test_class_report = classification_report(y_test, y_test_pred)
print("Test Classification Report: ")
print(test_class_report)
'''