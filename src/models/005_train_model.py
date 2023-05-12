import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# ------------------------------------ 1. Read Data ---------------------------------------------------


sys.path.append("..")
df = pd.read_pickle("../../data/processed/processed_data.pkl")
# ----------------------------- 2. Split/ Remove Outliers/ Feature Scaling -----------------------------
# Split Data and Remove Outliers from Train Data and Feature Scaling

# 2.1 Split Data to train and test
train, test = train_test_split(df, test_size=0.33, random_state=42)

# 2.2 remove outliers from train data
train = train[train['annual_inc'] <= 300000]
train = train[train['dti'] <= 50]
train = train[train['open_acc'] <= 50]
train = train[train['total_acc'] <= 80]
train = train[train['revol_util'] <= 120]
train = train[train['revol_bal'] <= 300000]

# 2.3  split train and test data into X and y
X_train, y_train = train.drop('loan_status', axis=1), train.loan_status
X_test, y_test = test.drop('loan_status', axis=1), test.loan_status

#2.4 Normalize Data
# StandardScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------ 3. Train Model ---------------------------------------------------


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


# 3.2 Conver input and output data to numpy array
X_train = np.array(X_train).astype('float32')
y_train = np.array(y_train).astype
X_test = np.array(X_test).astype
y_test = np.array(y_test).astype


# ------------------------------------ XGBoost Classifier ---------------------------------------------------
xgb_clf = XGBClassifier()

``
