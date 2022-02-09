import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
def compute_metrics_bce(eval_pred):
    logits, labels = eval_pred

    logits[logits>=0.5] = 1
    logits[logits<0.5] = 0
    predictions = logits.reshape(-1)
    labels = labels.reshape(-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, pos_label=1, average='binary')
    precision = precision_score(labels, predictions, pos_label=1, average='binary')
    recall = recall_score(labels, predictions, pos_label=1, average='binary')

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}