import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

print("Feature Names:")
print(iris.feature_names)

print("\nTarget Names:")
print(iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:", accuracy)

plt.figure(figsize=(12, 8))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title("Decision Tree Visualization")
plt.show()

print("""
----------------------------------------
ANALYSIS
----------------------------------------

1. The Iris dataset was used for classification.

2. A Decision Tree Classifier was trained using Scikit-Learn.

3. The dataset was divided into training and testing sets.

4. The model predicts flower species based on sepal and petal measurements.

5. Accuracy score evaluates model performance.

6. The decision tree visualization shows how predictions are made.

7. Decision Trees are easy to understand and interpret for classification problems.

----------------------------------------
PROJECT COMPLETED SUCCESSFULLY
----------------------------------------
""")
