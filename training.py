## Imports

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report


## Feature selection

# Get the data
df_train = pd.read_csv('training_data_fall2024.csv')

# Create weather index
df_train['weather_index'] = (df_train['temp'] + (100 - df_train['humidity']) + (100 - df_train['precip']) + df_train['dew']) / 4

# Binarize hour of day
df_train['hour_of_day_binary'] = df_train['hour_of_day'].apply(lambda x: 1 if 7 <= x <= 20 else 0)

# Dropping columns: snow
df_train = df_train.drop(columns=['snow'])

random_state = 43
X = df_train.drop(columns=['increase_stock'])
y = df_train['increase_stock']


separator = '-' * 60
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

## Feature Scaling and selection
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

selector = SelectKBest(f_classif, k=9)
x_train_kbest = selector.fit_transform(x_train, y_train)
x_test_kbest = selector.transform(x_test)


## ***** MODELS *****
## KNN
# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Hyperparameter search
param_grid = {
    'n_neighbors': range(1, 30),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
    'p': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)

# Fit the model using GridSearchCV on the training set
grid_search.fit(x_train_kbest, y_train)

# Get the best hyperparameters and model from Grid Search
best_knn = grid_search.best_estimator_

# Predictions on the test set
y_pred = best_knn.predict(x_test_kbest)

# Evaluate the model performance on the test set
accuracy_test = accuracy_score(y_test, y_pred)  # Accuracy on test set
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')

# Train accuracy
y_train_pred = best_knn.predict(x_train_kbest)  # Predictions on the training set
accuracy_train = accuracy_score(y_train, y_train_pred)  # Accuracy on training set

print(f'Best hyperparameters found by GridSearchCV for KNN: {grid_search.best_params_}')
print(f'Accuracy of KNN classifier on test set: {accuracy_test:.2f}')
print(f'F1 Score (weighted) on test set: {f1_weighted:.2f}')
print(f'F1 Score (macro) on test set: {f1_macro:.2f}')
print("\nClassification Report on test set:\n", classification_report(y_test, y_pred))
print(separator)


# Logistic regression
# Define hyperparameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['lbfgs', 'liblinear', 'saga'],  # Solvers
}

# Perform grid search
grid_search = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, cv=5, scoring='f1_weighted',error_score='raise')
grid_search.fit(x_train, y_train)

# Extract best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Test the model on the test set
y_pred = best_model.predict(x_test)
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Print the results
print("Best hyperparameters found by GridSearchCV for Logistic regression: ", best_params)
print("Accuracy of Logistic regression classifier on test set: ", accuracy)
print(f'F1 Score (weighted) on test set: {f1_weighted:.2f}')
print(f'F1 Score (macro) on test set: {f1_macro:.2f}')
print("\nClassification Report:\n", classification_report_result)
print(separator)


## Random Forest

# Define hyperparameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 10],
}

# Perform grid 
randomForest = RandomForestClassifier()
grid_search = GridSearchCV(estimator=randomForest, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(x_train, y_train)

# Extract best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Test the model on the test set
y_pred = best_model.predict(x_test)
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Print the results
print("Best hyperparameters found by GridSearchCV for Random Forest:", grid_search.best_params_)
print("Accuracy of Random Forest classifier on test set: ", accuracy)
print(f'F1 Score (weighted) on test set: {f1_weighted:.2f}')
print(f'F1 Score (macro) on test set: {f1_macro:.2f}')
print("\nClassification Report:\n", classification_report_result)
print(separator)


## LDA
lda = LinearDiscriminantAnalysis()

# Define grid parameters for LDA
grid_params = {
    'solver': ['lsqr', 'eigen'],
    'shrinkage': ['auto', 0.1, 0.2, 0.3, 0.5, 0.7]
}

# Use grid search with cross-validation
grid_search = GridSearchCV(estimator=lda, param_grid=grid_params, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(x_train, y_train)

# Predict using the best estimator found by GridSearchCV
y_pred_lda = grid_search.best_estimator_.predict(x_test)

# Accuracy and F1 scores for LDA
accuracy_lda = accuracy_score(y_test, y_pred_lda)
f1_weighted_lda = f1_score(y_test, y_pred_lda, average='weighted')
f1_macro_lda = f1_score(y_test, y_pred_lda, average='macro')

# Output results for LDA
print("Best hyperparameters found by GridSearchCV for LDA:", grid_search.best_params_)
print("Accuracy of LDA classifier on test set: ", accuracy_lda)
print(f'F1 Score (weighted) on test set: {f1_weighted_lda:.2f}')
print(f'F1 Score (macro) on test set: {f1_macro_lda:.2f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred_lda))
print(separator)

## QDA
qda = QuadraticDiscriminantAnalysis()

# Define grid parameters for QDA
grid_params_qda = {
    'reg_param': [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    'tol': [1e-4, 1e-3, 1e-2, 1e-1]
}

# Grid search with cross-validation for QDA
grid_search_qda = GridSearchCV(estimator=qda, param_grid=grid_params_qda, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search_qda.fit(x_train, y_train)

# Predict using the best estimator found by GridSearchCV for QDA
y_pred_qda = grid_search_qda.best_estimator_.predict(x_test)

# Accuracy and F1 scores for QDA
accuracy_qda = accuracy_score(y_test, y_pred_qda)
f1_weighted_qda = f1_score(y_test, y_pred_qda, average='weighted')
f1_macro_qda = f1_score(y_test, y_pred_qda, average='macro')

# Output results for QDA
print("Best hyperparameters found by GridSearchCV for QDA:", grid_search_qda.best_params_)
print("Accuracy of QDA classifier on test set: ", accuracy_qda)
print(f'F1 Score (weighted) on test set: {f1_weighted_qda:.2f}')
print(f'F1 Score (macro) on test set: {f1_macro_qda:.2f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred_qda))
print(separator)



# ** Naive Classifier
np.random.seed(random_state)

naive_classifier = np.random.rand(len(y_test))
naive_prediction = np.where(naive_classifier > 0.5, 'low_bike_demand', 'high_bike_demand')
accuracy = accuracy_score(y_test, naive_prediction)

f1_weighted = f1_score(y_test, naive_prediction, average='weighted')
f1_macro = f1_score(y_test, naive_prediction, average='macro')


print(f'Accuracy of Naive Classifier: {accuracy:.2f}')
print(f'F1 Score (weighted): {f1_weighted:.2f}')
print(f'F1 Score (macro): {f1_macro:.2f}')

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, naive_prediction))
print(separator)
