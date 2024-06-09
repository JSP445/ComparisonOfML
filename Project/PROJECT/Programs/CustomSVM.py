import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class SVMClassifier:
    def __init__(self, learning_rate=0.0001, lambda_param=0.01, n_iters=1000):
        """
        Initialize the support vector machine (SVM) classifier.

        Parameters:
        - learning_rate: The step size for updating the model parameters.
        - lambda_param: Regularization parameter for controlling the trade-off between margin maximization and classification error.
        - n_iters: The number of iterations for training the SVM.
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def fit(self, X, y):
        """
        Train the SVM classifier on the provided training data.

        Parameters:
        - X: Input features.
        - y: Target labels.
        """
        n_samples, n_features = X.shape

        # Convert class labels to +1 and -1
        y_ = np.where(y <= 0, -1, 1)

        if np.unique(y_).size == 1:
            print("Only one class present. Skipping training.")
            return
        
        # Initialize weight vector and bias term
        self.w = np.zeros(n_features)
        self.b = 0

        # Check for non-zero samples in each class
        if np.unique(y_).size == 1:
            print("Only one class present. Skipping training.")
            return

        # Training the SVM using the selected number of iterations
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Decision rule for updating parameters
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # Update for correctly classified samples
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Update for misclassified samples
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        """
        Make predictions on new data using the trained SVM classifier.

        Parameters:
        - X: Input features.

        Returns:
        - Predicted class labels.
        """
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

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

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert class labels to +1 and -1 for training and testing sets
y_train_binary = np.where(y_train <= 0, -1, 1)
y_test_binary = np.where(y_test <= 0, -1, 1)

# Create SVM classifier
svm_classifier = SVMClassifier()

# Train SVM classifier
svm_classifier.fit(X_train, y_train_binary)

# Make predictions on the test set
train_predictions = svm_classifier.predict(X_train)
test_predictions = svm_classifier.predict(X_test)

# Calculate accuracy for training and test sets
train_accuracy = accuracy_score(y_train_binary, train_predictions)
test_accuracy = accuracy_score(y_test_binary, test_predictions)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Show classification reports for training and test sets
train_report = classification_report(y_train_binary, train_predictions)
test_report = classification_report(y_test_binary, test_predictions)
print("Training Classification Report:")
print(train_report)
print("Test Classification Report:")
print(test_report)
'''
