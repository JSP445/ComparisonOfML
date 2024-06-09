import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from Main import custom_cross_val_score, plot_roc_curve, accuracy_score

class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[1,2], [3,4], [5,6]])
        self.y_train = np.array([1, -1, 1])
        self.X_test = np.array([2,3], [4,5])
        self.y_test = np.array([1, -1])
        self.y_true = np.array([1, 1, -1, -1])
        self.y_score = np.array([0.8, 0.7, 0.3, 0.2])