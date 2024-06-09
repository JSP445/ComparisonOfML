import numpy as np
from sklearn.preprocessing import StandardScaler

class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialise the Logistic Regression model.

        Parameters:
        - learning_rate: The step size for updating model parameters during gradient descent.
        - num_iterations: The number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()

    def sigmoid(self, z):
        """
        Sigmoid activation function.

        Parameters:
        - z: Input value:

        Returns:
        - Output value after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Fit method to train the Logistic Regression model.

        Parameters:
        - X: Input features
        - y: Target labels (binary).
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)

        # Gradient Descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X_scaled, self.weights) + self.bias
            predictions = self.sigmoid(linear_model)

            # Gradient calculations
            dw = (1 / num_samples) * np.dot(X_scaled.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Make predictions using the trained Logistic Regression model.

        Parameters:
        -X: Input features.
        
        Returns:
        - Predicted class labels (0 or 1).
        """

        # Feature Scaling
        X_scaled = self.scaler.transform(X)

        linear_model = np.dot(X_scaled, self.weights) + self.bias
        predictions = self.sigmoid(linear_model)
        return np.where(predictions > 0.5, 1, 0)

r'''
# Synthetic dataset
# X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
    

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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training model
LRModel = LogisticRegressionClassifier(learning_rate=0.1, num_iterations=1000)
LRModel.fit(X_train, y_train)

# Making predictions on the sets
predictions_train = LRModel.predict(X_train)
predictions_test = LRModel.predict(X_test)

# Evaluating the model
accuracy_train = accuracy_score(y_train, predictions_train)
accuracy_test = accuracy_score(y_test, predictions_test)
print(f"Train accuracy: {accuracy_train: }")
print(f"Test accuracy: {accuracy_test: }")

# Display training classification report
print("Training Set Classification Report: ")
print(classification_report(y_train, predictions_train))

# Display test classification report
print("Test Set Classification Report: ")
print(classification_report(y_test, predictions_test))
'''