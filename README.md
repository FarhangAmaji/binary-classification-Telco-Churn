
# Python Developer AI  Task

Objective: Develop an AI-driven Python application to extract and process key information from a diverse dataset of invoices and receipts.

## Practices and patterns (Must):

- [TDD](https://en.wikipedia.org/wiki/Test-driven_development)
- [DDD](https://en.wikipedia.org/wiki/Domain-driven_design)
- [BDD](https://en.wikipedia.org/wiki/Behavior-driven_development)
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

## Description:

In this challenge, you are tasked with creating a Python-based solution that can accurately extract key data points from a dataset of 50 invoices and receipts originating from various countries. Your solution should be able to identify and process the following key-value pairs:

a. Invoice Date
b. Invoice Amount
c. Tax Amount
d. Array of Invoice Line Items (including Description, Amount, and Quantity)
e. Issuing Company Name

Instructions:

Dataset Preparation: Create a dataset comprising 50 invoices and receipts from a variety of countries. Ensure that the dataset is diverse and includes different formats, languages, and currencies to effectively evaluate the performance of your AI model.

Text Extraction and Key-Value Recognition: Develop an AI model using Python that can extract text from the dataset and accurately identify the key information based on the specified key-value pairs. Consider using Optical Character Recognition (OCR) or other relevant techniques to achieve the desired accuracy.

Model Evaluation: Assess the performance of your AI model, considering factors such as precision, recall, and F1 score. Document any limitations or challenges you faced during the development process and propose potential improvements for future iterations.

Documentation: Provide clear and concise documentation, detailing the steps and processes involved in creating the dataset and developing the AI model. Include any necessary instructions for setting up, training, and testing the model.

Code Submission: Submit your code in a well-structured format, following best practices for coding conventions, commenting, and version control. The code should be easily understandable and maintainable by other developers.

Note: To ensure a fair evaluation, please refrain from using pre-trained models or third-party APIs for the text extraction and key-value recognition tasks. Your solution should be based on your own work and expertise.


## Test Driven

Example below is explaining how to deliver a TDD code base for your result

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
