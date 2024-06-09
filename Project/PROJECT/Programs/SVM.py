from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create SVM classifier
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')

# Train SVM classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
train_predictions = svm_classifier.predict(X_train)
test_predictions = svm_classifier.predict(X_test)

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