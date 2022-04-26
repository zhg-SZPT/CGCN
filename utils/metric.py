import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,auc
from sklearn.preprocessing import label_binarize
from scipy import interp

from scipy.interpolate import interp1d
from scipy.optimize import brentq

def EER(fpr, tpr):
    """计算EER"""
    def func(x):
        return 1. - x - interp1d(fpr, tpr)(x)
    
    eer = brentq(func, 0, 1)
    return eer

def acc_roc(output, target):
    """
    compute fpr、tpr then plot micro-average ROC curve and macro-average ROC cure
    """
    n_classes = output.shape[1]
    labels=target.cpu().numpy()
    
    c=[]
   
    for i in range(n_classes):
        c.append(i)
    b_labels= label_binarize(labels, classes=c)
    scores=output.cpu().numpy()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        a = b_labels[:, i]
        b = scores[:, i]
        fpr[i], tpr[i], _ = roc_curve(a, b) 
        roc_auc[i] = auc(fpr[i], tpr[i]) 

    fpr["micro"], tpr["micro"], th = roc_curve(b_labels.ravel(), scores.ravel(),drop_intermediate=True) 

    # EER
    EER = fpr['micro'][np.nanargmin(np.absolute((1-tpr['micro'] - fpr['micro'])))]
    print('eer:%0.4f' %EER)



    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(roc_auc["micro"])


    plt.figure()
    lw = 2

    marker_style = dict(linestyle='-', color='g', markersize=5,
                    markerfacecolor="tab:red", markeredgecolor="tab:red")

    plt.plot(1-tpr["micro"], fpr["micro"], label='Concat_Fusion', marker='x', linewidth=1.5, markevery=(0.0, 0.1), **marker_style) # FRR = 1 -TPR, FPR = FAR


    plt.plot([0, 1], [0, 1], 'k--', label='EER', lw=lw)

    plt.xlabel('FAR')
    plt.ylabel('FRR')
    plt.xlim([0, 0.2])
    plt.ylim([0, 0.2])
    plt.legend(loc="upper right")
    plt.show()



    return th,fpr,tpr,roc_auc