
# Python Developer AI  Task

Build a binary classification model to predict whether a customer will churn (i.e., discontinue their subscription) or not based on various features in the dataset. You are given a dataset containing customer information and whether they churned or not in the past. Your task is to use this dataset to train a machine learning model and evaluate its performance on a test set.

Dataset: You can use the Telco Customer Churn dataset, which is available on Kaggle at https://www.kaggle.com/blastchar/telco-customer-churn.

## Practices and patterns (Must):

- [TDD](https://en.wikipedia.org/wiki/Test-driven_development): A) Processes Data Clean Up B) Train C) Performance Test
- [DDD](https://en.wikipedia.org/wiki/Domain-driven_design): Adjust Relevant Domain
- Clean Architecture
- Clean Code
- Clean git commits that shows your work progress.

## Deliverables:

1. Python script that reads in the dataset, preprocesses it, trains a model, and evaluates its performance on a test set.
2. A report explaining your approach, including details on data preprocessing, feature selection/engineering, choice of algorithm, hyperparameter tuning, and evaluation metrics.
3. A visual representation of your model's performance (e.g., ROC curve, confusion matrix).
4. Please clone this repository in a new github repository in private mode and share with ID: `mason-chase` in private mode on github.com, make sure you do not erase my commits and then create a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) (code review).

## Requirements:

Use Python 3.x.
1. Use scikit-learn for the machine learning tasks.
2. Your code should be well-documented and follow best practices for software engineering.
3. Your report should be clear and concise, with appropriate visualizations to support your claims.
4. Your model should achieve an accuracy of at least 75% on the test set.


## Test Driven

Please improve below test model with clean architecture

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import unittest

class TestModel(unittest.TestCase):

    def setUp(self):
        # Load the data from a CSV file
        self.data = pd.read_csv('data.csv')
        
        # Split the data into features and labels
        self.X = self.data.drop('target', axis=1)
        self.y = self.data['target']
        
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # Preprocess the data by scaling the features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Train the logistic regression model
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        
        # Test the model on the test set
        self.y_pred = self.model.predict(self.X_test)
        
    def test_accuracy(self):
        # Assert
        expected_accuracy = 0.9
        actual_accuracy = accuracy_score(self.y_test, self.y_pred)
        self.assertAlmostEqual(actual_accuracy, expected_accuracy, delta=0.05)

    def test_model_has_coefs(self):
        # Assert
        self.assertIsNotNone(self.model.coef_)

if __name__ == '__main__':
    unittest.main()
```

