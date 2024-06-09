import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

class NaiveBayesClassifier:

    """
    Train the Naive Bayes classifier.

    Parameters:
    - X: Input features.
    - y: Target labels.
    
    """
    def fit(self, X, y):
        self.classes = np.unique(y) # Extract unique class labels
        # Initialise arrays to store class probabilities, means, and standard deviations
        self.classes_prob = np.zeros(len(self.classes)) 
        self.means = np.zeros((len(self.classes), X.shape[1]))
        self.stds = np.zeros((len(self.classes), X.shape[1]))

        # Calculate class probabilities, means and standard deviations
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.classes_prob[idx] = len(X_c) / len(X)
            self.means[idx, :] = X_c.mean(axis = 0)
            self.stds[idx, :] = X_c.std(axis = 0)
    
    def calcLikelihood(self, x, mean, std):
        """
        Calculate the likelihood of a feature given a class.

        Parameters:
        - x: Input feature.
        - mean: Mean of the feature for a specific class.
        - std: Standard deviation of the feature for a specific class.

        Returns:
        - Likelihood value.
        """
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        return np.prod((1 / (np.sqrt(2 * np.pi) * std)) * exponent)

    def predict(self, X):
        """
        Make predictions using the trained Naive Bayes classifier.

        Parameters:
        - X: Input features.

        Returns:
        - Predicted class labels.
        """
        predictions = []

        for x in X:
            posteriors = []

            # Calculate the posterior probability for each class
            for idx, c in enumerate(self.classes):
                prior = np.log(self.classes_prob[idx])
                likelihood = np.log(self.calcLikelihood(x, self.means[idx, :], self.stds[idx, :]))
                posterior = prior + np.sum(likelihood)
                posteriors.append(posterior)

            # Choose the class with the highest posterior probability
            predictions.append(self.classes[np.argmax(posteriors)])

        return predictions
    
# Load Iris dataset
#iris = load_iris()
#X = iris.data
#y = iris.target

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

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of Naive Bayes
nb = NaiveBayesClassifier()

# Train the classifier
nb.fit(X_train, y_train)

# Make predictions for the training and test sets
train_predictions = nb.predict(X_train)
test_predictions = nb.predict(X_test)

# Calculate accuracy for training and test sets
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Show classification reports for training and test sets
train_report = classification_report(y_train, train_predictions)
test_report = classification_report(y_test, test_predictions)
print("Training Classification Report:")
print(train_report)
print("Test Classification Report:")
print(test_report)
'''