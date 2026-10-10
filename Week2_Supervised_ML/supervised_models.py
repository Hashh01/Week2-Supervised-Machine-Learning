
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report
)

# Step 1: Load the dataset
df = pd.read_csv("student_performance.csv")

print("Student Dataset:")
print(df.head())

# Step 2: Select input features
X = df[["Study_Hours", "Attendance"]]

# Step 3: Linear Regression - Predict marks
y_marks = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_marks, test_size=0.25, random_state=42
)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

marks_predictions = regression_model.predict(X_test)

print("\n--- Linear Regression Results ---")
print("Actual Marks:", list(y_test))
print("Predicted Marks:", marks_predictions.round(2))
print("Mean Absolute Error:", round(
    mean_absolute_error(y_test, marks_predictions), 2
))
print("Root Mean Squared Error:", round(
    mean_squared_error(y_test, marks_predictions) ** 0.5, 2
))
print("R2 Score:", round(r2_score(y_test, marks_predictions), 2))

# Step 4: Logistic Regression - Predict Pass or Fail
y_result = (df["Result"] == "Pass").astype(int)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_result, test_size=0.25, random_state=42, stratify=y_result
)

classification_model = LogisticRegression()
classification_model.fit(X_train_c, y_train_c)

result_predictions = classification_model.predict(X_test_c)

print("\n--- Logistic Regression Results ---")
print("Actual Results:", list(y_test_c))
print("Predicted Results:", list(result_predictions))
print("Accuracy:", round(
    accuracy_score(y_test_c, result_predictions), 2
))
print(classification_report(
    y_test_c,
    result_predictions,
    labels=[0, 1],
    target_names=["Fail", "Pass"],
    zero_division=0
))
