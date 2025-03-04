import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (Car Evaluation Dataset from UCI Repository)
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
df = pd.read_csv(dataset_url, names=columns)

# Encode categorical features
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

# Split dataset
X = df.drop(columns=["class"])
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to calculate entropy
def calculate_entropy(y):
    counts = np.bincount(y)
    probabilities = counts / np.sum(counts)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# Function to compute information gain
def information_gain(X, y, feature):
    total_entropy = calculate_entropy(y)
    values, counts = np.unique(X[feature], return_counts=True)
    weighted_entropy = sum((counts[i] / sum(counts)) * calculate_entropy(y[X[feature] == values[i]]) for i in range(len(values)))
    return total_entropy - weighted_entropy

# Recursive function to build the decision tree
def build_tree(X, y, depth=0, max_depth=None):
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    if max_depth and depth >= max_depth:
        return Counter(y).most_common(1)[0][0]
    best_feature = max(X.columns, key=lambda feature: information_gain(X, y, feature))
    tree = {best_feature: {}}
    for value in np.unique(X[best_feature]):
        subset = X[X[best_feature] == value]
        tree[best_feature][value] = build_tree(subset.drop(columns=[best_feature]), y[subset.index], depth+1, max_depth)
    return tree

# Function to make predictions
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    value = sample[feature]
    return predict(tree[feature][value], sample) if value in tree[feature] else None

# Train and evaluate the model
decision_tree = build_tree(X_train, y_train, max_depth=5)
y_pred = [predict(decision_tree, X_test.iloc[i]) for i in range(len(X_test))]
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
