import numpy as np 
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score



def fpr95_score(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)  # 1 = anomalie
    idx = np.where(tpr >= 0.95)[0][0]
    return fpr[idx]

def cosinus_similarity(emb1,emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def evaluation(y_true, scores, verbose=True):

    auc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    fpr95 = fpr95_score(y_true, scores)

    if verbose:
        print(f"AUC:        {auc:.4f}")
        print(f"Avg Precision: {ap:.4f}")
        print(f"FPR@95:     {fpr95:.4f}")

    return auc, fpr95, ap 