# This script contains the metric functions used for all experiments

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def measure_performance(model, X, y):
    
    # get predictions on holdout
    y_pred = model.predict(X, verbose=0)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    y_pred = y_pred[:, 0]

    # accuracy: (tp + tn) / (tp + tn + fp + fn)
    accuracy = accuracy_score(y_true=y, y_pred=y_pred)
    
    # bal accuracy: average recall for each class, or sensitivity + specificity / 2 (binary)
    bal_accuracy = balanced_accuracy_score(y_true=y, y_pred=y_pred)

    # precision tp / (tp + fp)
    precision = precision_score(y_true=y, y_pred=y_pred)

    # recall / sensitivity: tp / (tp + fn)
    recall = recall_score(y_true=y, y_pred=y_pred)
    
    # get proportions of tn, tp
    tn, fp, fn, tp = confusion_matrix(y_true=y, y_pred=y_pred).ravel()
    
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true=y, y_pred=y_pred)
    # print('F1 score: %f' % f1)

   
    return accuracy, bal_accuracy, precision, recall, f1, tn, fp, fn, tp
