import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


# -------------------------------------------------------------
# 1. Read interim data
# -------------------------------------------------------------

sys.path.append("..")
import utility.plot_settings

df = pd.read_pickle("../../data/interim/df.pkl")

# Correlation between loan_status and other numerical features
df["loan_status"] = df["loan_status"].map({"Fully Paid": 1, "Charged Off": 0})
df.corr()["loan_status"].sort_values(ascending=False)[1:].plot(kind="barh")

# -------------------------------------------------------------
# 2. Data preprocessing
# -------------------------------------------------------------

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
df["mort_acc"].value_counts()
