import numpy as np
import pandas as pd
import os
import sys
from scipy import stats
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# ------------------------------------ 1. Read Data ---------------------------------------------------


sys.path.append("..")
df = pd.read_pickle("../../data/processed/processed_data.pkl")


# ----------------------------- 2. Split/ Remove Outliers/ Feature Scaling -----------------------------
# Split Data and Remove Outliers from Train Data and Feature Scaling

# 2.1 Split Data to train and test
train, test = train_test_split(df, test_size=0.03, random_state=42)

# 2.2 remove outliers from train data
train = train[train["annual_inc"] <= 300000]
train = train[train["dti"] <= 50]
train = train[train["open_acc"] <= 50]
train = train[train["total_acc"] <= 80]
train = train[train["revol_util"] <= 120]
train = train[train["revol_bal"] <= 300000]

# 2.3  split train and test data into X and y
X_train, y_train = train.drop("loan_status", axis=1), train.loan_status
X_test, y_test = test.drop("loan_status", axis=1), test.loan_status

# 2.4 Normalize Data
# StandardScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------ 3. XGBoost Classifier ---------------------------------------------------


# 3.1 Function to print the scores of the model
def print_score(true, pred, train=True):
    if train:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Train Result:\n")
        print(f"accuracy score: {accuracy_score(true, pred):.4f}\n")
        print(f"classification report: \n\n{classification_report(true, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
        print("------------------------------------------------------")
    elif train == False:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Test Result:\n")
        print(f"accuracy score: {accuracy_score(true, pred):.4f}\n")
        print(f"classification report: \n\n{classification_report(true, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
        print("------------------------------------------------------")


# 3.2 Train Model

"""
# Define the parameter space
param_dist = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(1, 10),
    "learning_rate": uniform(0.01, 0.6),
}

# Create a XGBClassifier instance
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="auc")  # or 'aucpr'

# Create a RandomizedSearchCV instance

xgb_cv = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    cv=3,  # Number of cross-validation folds
    n_iter=50,  # Number of parameter settings to sample
    scoring="roc_auc",  # Scoring metric
    n_jobs=-1,  # Use all CPU cores
    verbose=1,  # Verbosity level
    random_state=42,  # For reproducibility
)

# Fit the RandomizedSearchCV instance to the data
xgb_cv.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", xgb_cv.best_params_)
## Best parameters found:  {'learning_rate': 0.2499165830291533,
##                                'max_depth': 3, 'n_estimators': 413}


"""

# Best parameters found in previous search

best_params = {"learning_rate": 0.2499165830291533, "max_depth": 3, "n_estimators": 413}
# Create a new XGBClassifier instance with the best parameters
best_model = XGBClassifier(use_label_encoder=False, eval_metric="auc", **best_params)
# Train the model on your data
best_model.fit(X_train, y_train)
# Now you can use the trained model to make predictions
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)


#  3.3  Evaluate Model (XGBoost)

print_score(y_train, y_train_pred, train=True)
print_score(y_test, y_test_pred, train=False)


# 3.3.1  Plot roc curve
y_scores = best_model.predict_proba(X_test)[:, 1]
# Compute the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
# Create the RocCurveDisplay object
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
# Plot the ROC curve
roc_display.plot()
# Show the plot
plt.show()

# 3.3.2 Create roc_auc_score dictionary for model comparison
scores_dict = {
    "XGBoost": {
        "Train": roc_auc_score(y_train, best_model.predict(X_train)),
        "Test": roc_auc_score(y_test, best_model.predict(X_test)),
    },
}

# ------------------------------------ 4. Random Forest Classifier ---------------------------------------------------

# 4.1 Train Model(Random Forest)
# Define the parameter space
param_dist = {
    'n_estimators': range(100, 500),
    'max_depth': [None] + list(range(5, 50)),
    'min_samples_split': range(2, 20),
    'min_samples_leaf': range(1, 20),
}

# Create a RandomForestClassifier instance
rf_clf = RandomForestClassifier()

# Create a RandomizedSearchCV instance
rf_random_search = RandomizedSearchCV(estimator=rf_clf,
                                        param_distributions=param_dist,
                                        n_iter=50,
                                        cv=3,
                                        verbose=3,
                                        random_state=42,
                                        n_jobs=-1)

 # Fit the RandomizedSearchCV instance to the data
rf_random_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", rf_random_search.best_params_)
## Best parameters found:  {'n_estimators': {}

# Best parameters found in previous search
best_params = {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 49}
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

#  4.2  Evaluate Model (Random Forest)

print_score(y_train, y_train_pred, train=True)
print_score(y_test, y_test_pred, train=False)

# 4.2.1  Plot roc curve
y_scores = best_model.predict_proba(X_test)[:, 1]
# Compute the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
# Create the RocCurveDisplay object
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
# Plot the ROC curve
roc_display.plot()
# Show the plot
plt.show()

# 4.2.2 add roc_auc_score dictionary for model comparison
scores_dict["Random Forest"] = { Train: roc_auc_score(y_train, best_model.predict(X_train)),
                                Test: roc_auc_score(y_test, best_model.predict(X_test))}

# ------------------------------------ 5. Neural Network Classifier ---------------------------------------------------

# 5.1 Train Model(Neural Network)
