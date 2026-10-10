# Week 2: Supervised Machine Learning Models

## About the Project

This project is part of my machine learning internship task. The main purpose was to learn how supervised machine learning models work and how they can be used to make predictions from existing data.

I used Python and Scikit-learn to build two models: Linear Regression and Logistic Regression. The project uses a small sample dataset containing students' study hours, attendance, marks, and exam results.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Visual Studio Code

## Dataset

The dataset is named `student_performance.csv`. It contains the following columns:

- `Study_Hours`: Number of hours spent studying
- `Attendance`: Attendance percentage
- `Marks`: Marks obtained by the student
- `Result`: Pass or Fail status

This is a sample dataset created for learning and practice.

## Models Implemented

### 1. Linear Regression

Linear Regression is used to predict students' marks based on their study hours and attendance.

The model was evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the R-squared score.

**Results:**
- MAE: 1.31
- RMSE: 1.72
- R-squared score: 0.99

### 2. Logistic Regression

Logistic Regression is used to predict whether a student will pass or fail based on study hours and attendance.

The model was evaluated using accuracy, precision, recall, and F1-score.

**Results:**
- Accuracy: 100%
- Precision: 1.00 for both classes
- Recall: 1.00 for both classes
- F1-score: 1.00 for both classes

These results were obtained from a test set of only four records, so they should not be considered proof of performance on real-world data.

## How to Run the Project

1. Download or clone this repository.
2. Open the project folder in Visual Studio Code.
3. Install the required libraries:

   ```bash
   python -m pip install pandas numpy scikit-learn matplotlib
   ```

4. Run the Python file:

   ```bash
   python supervised_models.py
   ```

Make sure `student_performance.csv` and `supervised_models.py` are in the same folder before running the program.

## What I Learned

While completing this task, I learned how to load a dataset using Pandas, select input features, divide data into training and testing sets, train machine learning models, and evaluate predictions using different performance metrics.

I also learned the difference between regression, which predicts numerical values, and classification, which predicts categories.

## Conclusion

This project gave me practical experience in implementing supervised machine learning using Python. It helped me understand the basic workflow of training, predicting, and evaluating models. I plan to use these concepts as a foundation for learning more machine learning algorithms and working with larger datasets.
