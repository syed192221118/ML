import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
np.random.seed(42)
data = {
    'age': np.random.randint(20, 60, 1000),
    'income': np.random.randint(20000, 100000, 1000),
    'credit_score': np.random.randint(300, 850, 1000),
    'loan_amount': np.random.randint(1000, 50000, 1000),
    'approved': np.random.choice([0, 1], 1000)
}
df = pd.DataFrame(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('approved', axis=1), df['approved'], test_size=0.2, random_state=42)

# Calculate means and variances for each class
classes = np.unique(y_train)
mean = {c: X_train[y_train == c].mean() for c in classes}
var = {c: X_train[y_train == c].var() for c in classes}
priors = {c: len(X_train[y_train == c]) / len(X_train) for c in classes}

# Predict on the test set
predictions = []
for x in X_test.to_numpy():
    posteriors = {c: np.sum(np.log(np.exp(- (x - mean[c]) ** 2 / (2 * var[c])) / np.sqrt(2 * np.pi * var[c]))) + np.log(priors[c]) for c in classes}
    predictions.append(max(posteriors, key=posteriors.get))

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test, predictions))
