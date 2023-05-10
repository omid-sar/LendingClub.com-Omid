import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------ 1. Read Data -------------------------------------------

sys.path.append("..")
import utility.plot_settings

df = pd.read_pickle("../../data/interim/df.pkl")

# Correlation between loan_status and other numerical features
df["loan_status"] = df["loan_status"].map({"Fully Paid": 1, "Charged Off": 0})
df.corr()["loan_status"].sort_values(ascending=False)[1:].plot(kind="barh")

# ------------------------------------ 2. Data Preprocessing ------------------------------------

"""
The goal of this step is to prepare the data for the machine learning algorithms.

we will do the following steps:
    1. Remove outliers
    2. Remove and fill missing values
    3. Remove repeated features
    4. Convert categorical features to numerical
"""



# 2.1 Remove outliers


# 2.1.1
# installment: is monthly payment of loan.
# installment column is combination of loan_amnt and term.
# we will remove this column
df.drop("installment", axis=1, inplace=True)


# 2.1.2
# emp_title: is the job title supplied by the Borrower when applying for the loan.
df["emp_title"].nunique()
# there are 173105 unique job titles.
# we can't convert this column to numerical column. so we will remove this column
df.drop("emp_title", axis=1, inplace=True)


# 2.1.3
# emp_length: is the employment length in years.
# groupby emp_length and find the percentage of Fully Paid and Charged Off
df.groupby("emp_length")["loan_status"].value_counts(normalize=True).unstack()
# There is not significant difference between Fully Paid and Charged Off for each emp_length.
# So we will remove this column
df.drop("emp_length", axis=1, inplace=True)


# 2.1.4
# purpose: is the purpose of the loan.
# title: is the loan title provided by the borrower.
df["purpose"].nunique()
# there are 14 unique purpose.
df["title"].nunique()
# there are 63144 unique title.
# There are many unique values in title column. So we will remove this column
df.drop("title", axis=1, inplace=True)


# 2.1.5
# grade: is the loan grade. 
# sub_grade: is the loan subgrade. 
# grade is a part of sub_grade. So we will remove grade column
df.drop("grade", axis=1, inplace=True)


# 2.2 finding columns with missing values and fill/remove them
 

# 2.2.1
# Function to find missing values and percentage in each column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: "Missing Values", 1: "% of Total Values"}
    )
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0
    ].sort_values(
        "% of Total Values", ascending=False
    ).round(1)  # Round percentages
    # Print some summary information
    print(
        "Your selected dataframe has "
        + str(df.shape[1])
        + " columns.\n"
        "There are "
        + str(mis_val_table_ren_columns.shape[0])
        + " columns that have missing values."
    )
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

missing_values_table(df)


# 2.2.2
# fill missing values in mort_acc column
# mort_acc: is the number of mortgage accounts.
# find the correlation between mort_acc and other numerical features
df.corr()["mort_acc"].sort_values(ascending=False)[1:].plot(kind="barh")
# total_acc has highest correlation with mort_acc

# fill missing values in mort_acc column with mean of mort_acc for each total_acc
total_acc_avg = df.groupby("total_acc")["mort_acc"].mean()
# function to fill missing values in mort_acc column
def fill_mort_acc(total_acc, mort_acc):
    """
    total_acc: is the total number of credit lines currently in the borrower's credit file
    mort_acc: is the number of mortgage accounts.
    """
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc
# apply fill_mort_acc function to mort_acc column
df["mort_acc"] = df.apply( lambda x: fill_mort_acc(x["total_acc"], x["mort_acc"]), axis=1)


# 2.2.3
# remove rows with missing values in revol_util and pub_rec_bankruptcies columns
df.dropna(inplace=True)


# 2.3 categorical features and dummy variables
print(df.select_dtypes(["object"]).columns)
# there are 7 categorical features
#['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status',
# 'purpose', 'initial_list_status', 'application_type', 'address']


# 2.3.1
# term: is the number of payments on the loan. Values are in months and can be either 36 or 60.
# convert term column to numerical column
term_values = {' 36 months': 36, ' 60 months': 60}
df['term'] = df.term.map(term_values)
df.term.unique()


# 2.3.2
# home_ownership: is the home ownership status provided by the borrower during registration.
df.home_ownership.unique()
# there are 6 unique values in home_ownership column.
# we will group 'ANY' and 'NONE' values into 'OTHER' category
df["home_ownership"] = df["home_ownership"].replace(["NONE", "ANY"], "OTHER")

# 2.3.3
# dummy variables for sub_grade, home_ownership, verification_status, purpose, initial_list_status, application_type   
dummies_df = pd.get_dummies(df[["sub_grade", "home_ownership", "verification_status", "purpose", "initial_list_status", "application_type"]], drop_first=True)
df = pd.concat([df.drop(["sub_grade", "home_ownership", "verification_status", "purpose", "initial_list_status", "application_type"], axis=1), dummies_df], axis=1)


# 2.3.4
# address: is the address provided by the borrower during registration.
df.address.head()
# extract zip code from address column and create a new column zip_code
df["zip_code"] = df["address"].apply(lambda x: x[-5:])
df["zip_code"].nunique()
# there are 10 unique zip codes!
# its really strange!
# we will create dummy variables for zip_code column
dummy_df = pd.get_dummies(df["zip_code"], drop_first=True)
df = pd.concat([df.drop("zip_code", axis=1), dummy_df], axis=1)
# remove address column
df.drop("address", axis=1, inplace=True)


#2.4 remove duplicate columns and rows

# 2.4.1 find duplicate columns
df_T = df.T
duplicate_columns = df_T.duplicated()
# remove duplicate columns
df_T = df_T[~duplicate_columns]
df = df_T.T

# 2.4.2 find duplicate rows
duplicate_rows = df.duplicated()
# remove duplicate rows
df = df[~duplicate_rows]

# ------------------------------------ 3. Export Proceessed Data ------------------------------------

df.to_pickle("../../data/processed/processed_data.pkl")


# ------------------------------------ 3. Data Preprocessing ------------------------------------


#3.1 Train Test split
X = df.drop("loan_status", axis=1)
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

#3.2 remove outliers
# remove outliers in numerical features
X_train = X_train[X_train['annual_inc'] <= 300000]
X_train = X_train[X_train['dti'] <= 50]
X_train = X_train[X_train['open_acc'] <= 50]
X_train = X_train[X_train['total_acc'] <= 80]
X_train = X_train[X_train['revol_util'] <= 120]
X_train = X_train[X_train['revol_bal'] <= 300000]



