import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def roc_curve(path):
    
    '''        
        Gets the path with the scores from a classifier, computes True Positive Rate and False Positive Rate, 
        AUC score. Also draws ROC curve based on the abovementioned metrics. 
        
        Arguments:
        
        path - a path to the csv file with scores.
    '''   
    
    # Read the csv file
    df = pd.read_csv(path)
    
    # Set thresholds list
    thresholds = list(np.array(list(range(0, 105, 5)))/100)

    # Initialize a list for roc points
    roc_point = []

    for threshold in thresholds:

        tp = 0; fp = 0; fn = 0; tn = 0

        for index, instance in df.iterrows():
            actual = instance["actual"]
            prediction = instance["prediction"]

            if prediction >= threshold:
                prediction_class = 1
            else:
                prediction_class = 0

            if prediction_class == 1 and actual == 1:
                tp = tp + 1
            elif actual == 1 and prediction_class == 0:
                fn = fn + 1
            elif actual == 0 and prediction_class == 1: 
                fp = fp + 1
            elif actual == 0 and prediction_class == 0:
                tn = tn + 1

        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        roc_point.append([tpr, fpr])
    pivot = pd.DataFrame(roc_point, columns = ["tpr", "fpr"])
    pivot["threshold"] = thresholds
    
    auc = round(abs(np.trapz(pivot.tpr, pivot.fpr)), 4)
    plt.scatter(pivot.fpr, pivot.tpr, label=f'AUC Score: {auc:.3f}', c='red', alpha=0.7)
    plt.plot([0, 1], c='blue', alpha=0.7)
    plt.xlabel('FAR (FPR)')
    plt.ylabel('FRR (TPR)')
    plt.legend()

roc_curve('binary-preds.csv')
