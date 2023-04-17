import numpy as np
import matplotlib.pyplot as plt
# it is actually correct like this! need to implement it in learnHeat

# Generate example data
y_true = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 0])
y_score = np.array([0.8, 0.6, 0.3, 0.7, 0.2, 0.4, 0.9, 0.7, 0.5, 0.2])

# Compute precision and recall for different thresholds
thresholds = np.linspace(0, 1, 10)
precisions = []
recalls = []
for threshold in thresholds:
    y_pred = (y_score >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp+fp == 0:
        precision = 1
    if tp+fn == 0:
        recall = 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    precisions.append(precision)
    recalls.append(recall)

# Plot the precision-recall curve
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()