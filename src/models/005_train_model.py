# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt

# Machine learning libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Deep learning libraries
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy

# ------------------------------------ 1. Read Data ---------------------------------------------------


sys.path.append("..")
df = pd.read_pickle("../../data/processed/processed_data.pkl")

len(df.columns)
# ------------------------- 2. Split/ Remove Outliers/ Feature Scaling / Print Score Function------------------------
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

# 2.5 Function to print scores and confusion matrix for Model Evaluation


def print_score(true, pred, dataset_type="Train", model_name="Model"):
    clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
    print(
        f"{model_name} {dataset_type} Result:\n================================================================================"
    )
    print(f"Accuracy Score: {accuracy_score(true, pred):.4f}\n")
    print(f"CLASSIFICATION REPORT: \n{clf_report}\n")
    print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")


def plot_roc_curve(model, X_test, y_test, model_name):
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title(f"ROC Curve for {model_name}")
    plt.show()


# Example usage
# plot_roc_curve(best_model, X_test, y_test, 'Best Model')

# ------------------------------------ 3. XGBoost Classifier ---------------------------------------------------

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

print_score(y_train, y_train_pred, dataset_type="Train", model_name="XGBoost")
print_score(y_test, y_test_pred, dataset_type="Test", model_name="XGBoost")


# 3.3.1  Plot roc curve
plot_roc_curve(best_model, X_test, y_test, "XGBoost")

# 3.3.2 Create roc_auc_score dictionary for model comparison
scores_dict = {
    "XGBoost": {
        "Train": roc_auc_score(y_train, best_model.predict(X_train)),
        "Test": roc_auc_score(y_test, best_model.predict(X_test)),
    },
}

# ------------------------------------ 4. Random Forest Classifier ---------------------------------------------------

# 4.1 Train Model(Random Forest)

"""
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
##Best parameters found:  {'n_estimators': 184, 'min_samples_split': 6, 'min_samples_leaf': 1, 'max_depth': 40}

"""


# Best parameters found in previous search
best_params = {
    "n_estimators": 184,
    "min_samples_split": 6,
    "min_samples_leaf": 1,
    "max_depth": 40,
}
best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

#  4.2  Evaluate Model (Random Forest)

print_score(y_train, y_train_pred, dataset_type="Train", model_name="Random Forest")
print_score(y_test, y_test_pred, dataset_type="Test", model_name="Random Forest")

# 4.2.1  Plot roc curve
plot_roc_curve(best_model, X_test, y_test, "Random Forest")

# 4.2.2 add roc_auc_score dictionary for model comparison
scores_dict["Random Forest"] = {
    Train: roc_auc_score(y_train, best_model.predict(X_train)),
    Test: roc_auc_score(y_test, best_model.predict(X_test)),
}

# ------------------------------------ 5. Neural Network Classifier ---------------------------------------------------


# Function to plot the learning curve of NN_model
def plot_learning_evolution(r):
    plt.figure(figsize=(12, 8))

    if "loss" in r.history:
        plt.subplot(2, 2, 1)
        plt.plot(r.history["loss"], label="Loss")
        if "val_loss" in r.history:
            plt.plot(r.history["val_loss"], label="val_Loss")
        plt.title("Loss evolution during training")
        plt.legend()

    if "AUC" in r.history:
        plt.subplot(2, 2, 2)
        plt.plot(r.history["AUC"], label="AUC")
        if "val_AUC" in r.history:
            plt.plot(r.history["val_AUC"], label="val_AUC")
        plt.title("AUC score evolution during training")
        plt.legend()

    plt.tight_layout()  # adjust subplot parameters for better spacing
    plt.show()


def nn_model(
    num_columns,
    num_labels,
    hidden_units,
    dropout_rates,
    learning_rate,
    activation_function="relu",
    optimizer="adam",
    early_stopping_patience=10,
    model_checkpoint_path="model.h5",
):
    # Define the input layer
    inp = Input(shape=(num_columns,))
    x = BatchNormalization()(inp)
    x = Dropout(dropout_rates[0])(x)

    # Define the hidden layers
    for i in range(len(hidden_units)):
        x = Dense(hidden_units[i], activation=activation_function)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rates[i + 1])(x)

    # Define the output layer
    x = Dense(num_labels, activation="sigmoid")(x)

    # Define the model and compile it
    model = Model(inputs=inp, outputs=x)
    model.compile(
        optimizer=optimizer(learning_rate),
        loss=BinaryCrossentropy(),
        metrics=[AUC(name="AUC")],
    )

    # Define early stopping and model checkpointing
    early_stopping = EarlyStopping(
        monitor="val_AUC",
        patience=early_stopping_patience,
        mode="max",
        restore_best_weights=True,
    )
    model_checkpoint = ModelCheckpoint(
        model_checkpoint_path, monitor="val_AUC", mode="max", save_best_only=True
    )

    return model, [early_stopping, model_checkpoint]


# Example usage
num_columns = X_train.shape[1]
num_labels = 1
hidden_units = [128, 64]
dropout_rates = [0.2, 0.5, 0.5]
learning_rate = 0.001

 model, callbacks = nn_model(num_columns, num_labels, hidden_units, dropout_rates, learning_rate)
# model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=callbacks)
