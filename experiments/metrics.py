# This script contains the metric functions used for all experiments

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def measure_performance(model, X_test, y_true):
    # get predictions on holdout
    y_pred = model.predict(X_test, verbose=0)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    y_pred = y_pred[:, 0]

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_true, y_pred)
    # print('Accuracy: %f' % accuracy)

    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred)
    # print('Precision: %f' % precision)

    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred)
    # print('Recall: %f' % recall)

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, y_pred)
    # print('F1 score: %f' % f1)

   
    return accuracy, precision, recall, f1
