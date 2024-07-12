# Store Data Machine Learning Project

## Overview

This project aims to predict store performance using machine learning techniques. We utilized logistic regression, decision tree, and neural network models to classify the store performance as "Good" or "Bad". The process involved data cleaning, feature engineering, model training, evaluation, and improvement.

## Dataset

The dataset includes various features about stores, such as:

- Staff
- Floor Space
- Window
- Demographic score
- 40min population
- 30 min population
- 20 min population
- 10 min population
- Store age
- Clearance space
- Competition number
- Competition score
- Car park (Yes/No)
- Location (Retail Park, Shopping Centre, Village)
- Performance (Good/Bad)

## Methodology

### 1. Data Cleaning
- Filled missing values using forward fill method.
- Converted categorical variables to dummy variables.

### 2. Feature Engineering
- Extracted features (X) and target (y).
- Normalized the feature set using MinMaxScaler.

### 3. Train/Test Split
- Split the data into training (80%) and testing (20%) sets.

### 4. Model Training
- Used GridSearchCV for hyperparameter tuning of the following models:
  - Logistic Regression
  - Decision Tree Classifier
  - Neural Network (MLPClassifier)

### 5. Model Evaluation
- Evaluated the models using accuracy, confusion matrix, and classification report.

## Results

### Logistic Regression
- **Accuracy**: 57.1%
- **Confusion Matrix**:
  ```
  [[ 6,  5],
   [ 7, 10]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

           0       0.46      0.55      0.50        11
           1       0.67      0.59      0.62        17

    accuracy                           0.57        28
   macro avg       0.56      0.57      0.56        28
weighted avg       0.59      0.57      0.58        28
  ```

### Decision Tree
- **Accuracy**: 46.4%
- **Confusion Matrix**:
  ```
  [[5, 6],
   [9, 8]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

           0       0.36      0.45      0.40        11
           1       0.57      0.47      0.52        17

    accuracy                           0.46        28
   macro avg       0.46      0.46      0.46        28
weighted avg       0.49      0.46      0.47        28
  ```

### Neural Network
- **Accuracy**: 60.7%
- **Confusion Matrix**:
  ```
  [[ 6,  5],
   [ 6, 11]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

           0       0.50      0.55      0.52        11
           1       0.69      0.65      0.67        17

    accuracy                           0.61        28
   macro avg       0.59      0.60      0.59        28
weighted avg       0.61      0.61      0.61        28
  ```

### Model Comparison
- **Logistic Regression**: Accuracy - 57.1%
- **Decision Tree**: Accuracy - 46.4%
- **Neural Network**: Accuracy - 60.7%

The Neural Network model performed the best among the three models with an accuracy of 60.7%.

## Visualization

We visualized the confusion matrices and compared the accuracies of the models using bar charts.

### Confusion Matrices
![Logistic Regression Confusion Matrix](path_to_logistic_regression_confusion_matrix.png)
![Decision Tree Confusion Matrix](path_to_decision_tree_confusion_matrix.png)
![Neural Network Confusion Matrix](path_to_neural_network_confusion_matrix.png)

### Accuracy Comparison
![Model Accuracy Comparison](path_to_model_accuracy_comparison.png)

## Conclusion

The Neural Network model provided the highest accuracy for predicting store performance. Further improvements can be made by exploring more advanced feature engineering, hyperparameter tuning, and using ensemble methods.
