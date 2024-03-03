#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:57:14 2024

@author: hue
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('BIRTH WEIGHT_csv.csv')

# Display the first few rows of the dataframe
print(data.head())


# Cross tabulation of "LOW" with each independent variable
cross_tabulations = {}
independent_vars = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI', 'FTV']

for var in independent_vars:
    cross_tab = pd.crosstab(data['LOW'], data[var], margins=True, margins_name="Total")
    cross_tabulations[var] = cross_tab

# Display each cross-tabulation properly
for var, tab in cross_tabulations.items():
    print(f"\nCross Tabulation of LOW with {var}:")
    print(tab, "\n")



# Selecting independent and dependent variables
X = data[['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI', 'FTV']]
y = data['LOW']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing and fitting the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predicting probabilities
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

# Function to apply different cut-off values and calculate metrics
def apply_cutoff(y_true, y_prob, cutoff):
    y_pred = np.where(y_prob > cutoff, 1, 0)
    sensitivity = np.mean(np.where((y_pred == 1) & (y_true == 1), 1, 0))
    specificity = np.mean(np.where((y_pred == 0) & (y_true == 0), 1, 0))
    misclassification_rate = np.mean(np.where(y_pred != y_true, 1, 0))
    return sensitivity, specificity, misclassification_rate

# Applying different cut-offs
cutoff_values = [0.4, 0.3, 0.55]
metrics = {cutoff: apply_cutoff(y_test, y_pred_prob, cutoff) for cutoff in cutoff_values}

# Display metrics for each cut-off properly
print("Metrics for Different Cut-off Values:")
for cutoff, (sensitivity, specificity, misclassification_rate) in metrics.items():
    print(f"\nCut-off Value: {cutoff}")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.2f}")
    print(f"Specificity (True Negative Rate): {specificity:.2f}")
    print(f"Misclassification Rate: {misclassification_rate:.2f}")
    
# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = roc_auc_score(y_test, y_pred_prob)

print(f"AUC Score: {auc_score}")

# Interpreting AUC Score
if auc_score > 0.75:
    print("The model shows excellent discrimination ability.")
elif auc_score > 0.5:
    print("The model has moderate discrimination ability.")
else:
    print("The model's discrimination ability is poor.")

# Choosing the recommended cut-off
misclassification_rates = {cutoff: metrics[cutoff][2] for cutoff in cutoff_values}
recommended_cutoff = min(misclassification_rates, key=misclassification_rates.get)

print(f"The recommended cut-off value is {recommended_cutoff} based on the lowest misclassification rate.")

# Plotting ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
