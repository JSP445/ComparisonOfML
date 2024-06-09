import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from KNN import KNNClassifier
from CustomSVM import SVMClassifier
from CustomDecisionTree import CustomDecisionTreeClassifier
from LogisticRegression import LogisticRegressionClassifier
from NaiveBayes import NaiveBayesClassifier
from sklearn.datasets import load_iris

def custom_cross_val_score(classifier, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)
    
    return np.mean(scores)

def plot_roc_curve(y_true, y_score, title='ROC Curve'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

while True:
    # User choice for dataset
    dataset_choice = input("Select dataset: \n 1. Heart Failure dataset \n 2. Heart Attack dataset \n 3. Iris dataset \n 4. Heart Disease dataset \n Type 'exit' to quit \n")

    if dataset_choice.lower() == 'exit':
        print("Exiting program...")
        break

    if dataset_choice == '1':
        file_path = 'heartfailure.csv'
        df = pd.read_csv(file_path)
    elif dataset_choice == '2':
        file_path = 'heartattack.csv'
        df = pd.read_csv(file_path)
    elif dataset_choice == '3':
        # Load the Iris dataset
        iris = load_iris()
        df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
    elif dataset_choice == '4':
        file_path = 'heartdisease.csv'
        df = pd.read_csv(file_path)
    else:
        print("Invalid dataset choice")
        continue

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for i in range(X.shape[1]):
        if isinstance(X[0, i], str):
            X[:, i] = label_encoder.fit_transform(X[:, i])

    # Convert to numeric type
    X = X.astype(float)

    # Convert class labels to +1 and -1
    y_binary = np.where(y <= 0, -1, 1)

    # Split the data with a consistent random state
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # Create a dictionary to map user choices to classifiers
    classifiers = {
        '1': KNNClassifier(k=3),
        '2': SVMClassifier(),
        '3': CustomDecisionTreeClassifier(),
        '4': LogisticRegressionClassifier(),
        '5': NaiveBayesClassifier()
    }

    # Algorithm selection
    algorithm_choice = input("Select algorithm \n 1. K Nearest Neighbours \n 2. SVM \n 3. Decision Trees \n 4. Logistic Regression \n 5. Naive Bayes \n Type 'exit' to quit \n")

    if algorithm_choice.lower() == 'exit':
        print("Exiting program...")
        break

    # Use the dictionary to get the selected algorithm
    algorithm = classifiers.get(algorithm_choice)

    if algorithm is None:
        print("Invalid choice")
        continue

    # Redirect standard output to null device to suppress output
    stdout_original = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    # Fit the selected algorithm
    algorithm.fit(X_train, y_train)

    # Restore standard output
    sys.stdout = stdout_original

    # Training accuracy
    y_train_pred = algorithm.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

    # Test accuracy
    y_test_pred = algorithm.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Generate training classification report
    train_class_report = classification_report(y_train, y_train_pred, zero_division=1)
    print("Training Classification Report:")
    print(train_class_report)

    # Generate test classification report
    test_class_report = classification_report(y_test, y_test_pred, zero_division=1)
    print("Test Classification Report:")
    print(test_class_report)

    # Cross-validation
    cv_accuracy = custom_cross_val_score(algorithm, X, y_binary)
    print(f"Cross validation scores: {cv_accuracy * 100:.2f}%")

    # Plot ROC Curve for the test set
    y_test_score = algorithm.predict(X_test)
    plot_roc_curve(y_test, y_test_score, title='ROC Curve for Test Set')
