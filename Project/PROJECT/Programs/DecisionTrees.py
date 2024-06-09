import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#print(os.getcwd())

# Import data
file_path = r'C:\Users\janit\Desktop\Project\PROJECT\heart.csv'

heart_data = pd.read_csv(file_path)
heart_data.head()

# Separate features from target variable
heart_x = heart_data.drop("HeartDisease", axis = 1)
heart_y = heart_data['HeartDisease']

# Replace categorical data with "dummy" variables thought one-hot-encoding through pandas.get_dummies()

heart_x_encoded = pd.get_dummies(heart_x, drop_first=True)
heart_x_encoded.head()

# Split the data

X_train, X_test, y_train, y_test = train_test_split(heart_x_encoded, heart_y, test_size=0.3)

# Build the tree

DecisionTree = DecisionTreeClassifier(max_depth=2)
DecisionTree.fit(X_train, y_train)
train_predictions = DecisionTree.predict(X_train)
test_predictions = DecisionTree.predict(X_test)
train_acc = accuracy_score(y_train, train_predictions)
test_acc = accuracy_score(y_test, test_predictions)
print("Training accuracy = ", train_acc)
print("Test accuracy", test_acc)


