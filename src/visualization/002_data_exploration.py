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

df
# -------------------------------------------------------------
# 2. Data Visualization
# -------------------------------------------------------------

# Heatmap of Correlation
plt.figure(figsize=(8, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="viridis", cbar=False)
"""
Notice
There is a perfect correlation betweenn "loan_amntand" and  "installment"
which makes sense. Since installment is a friction of loan and the interest rate
"""

# loan_status : Current status of the loan
fig, ax = plt.subplots(figsize=(12, 4))
fig = sns.countplot(data=df, x="loan_status", palette="pastel")
fig.set(xlabel="", ylabel="Counts", title="Loan Status Counts")


"""
loan_amnt & installment
installment: The monthly payment owed by the borrower if the loan originates.
loan_amnt: The listed amount of the loan applied for by the borrower.
"""
# barplot of loan_amnt and installment by loan_status
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig_bar1 = sns.histplot(df, x="loan_amnt", hue="loan_status", ax=axes[0], bins=40)
fig_bar1.set(xlabel="Loan Amount", ylabel="Counts", title="Loan Amount by Loan Status")

fig_bar2 = sns.histplot(df, x="installment", hue="loan_status", ax=axes[1], bins=40)
fig_bar2.set(xlabel="Installment", ylabel="Count", title="Installment by Loan Status")

# boxplot of loan_amnt and installment by loan_status
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig1 = sns.boxplot(
    data=df, x="loan_status", y="loan_amnt", palette="pastel", ax=axes[0]
)
fig1.set(title="Loan Status by Loan Amount", xlabel="Loan Status", ylabel="Loan Amount")

fig2 = sns.boxplot(
    data=df, x="loan_status", y="installment", palette="pastel", ax=axes[1]
)
fig2.set(title="Loan Status by Installment", xlabel="Loan Status", ylabel="Installment")


"""
grade & sub_grade
grade: LC assigned loan grade
sub_grade: LC assigned loan subgrade
"""
# barplot of grade and sub_grade by loan_status
fig, ax = plt.subplots(figsize=(12, 4))

fullyPaid = df.loc[df.loan_status == "Fully Paid", "grade"].value_counts().sort_index()
chargedOff = (
    df.loc[df.loan_status == "Charged Off", "grade"].value_counts().sort_index()
)

ax.bar(x=fullyPaid.index, height=fullyPaid.values, alpha=0.5)
ax.bar(x=chargedOff.index, height=chargedOff.values, alpha=0.6)
ax.set(title="Loan Status by Grade", ylabel="Counts", xlabel="Grades")


# barplot of sub_grade by loan_status
fig, ax = plt.subplots(figsize=(12, 4))

fullyPaid = (
    df.loc[df.loan_status == "Fully Paid", "sub_grade"].value_counts().sort_index()
)
chargedOff = (
    df.loc[df.loan_status == "Charged Off", "sub_grade"].value_counts().sort_index()
)

ax.bar(fullyPaid.index, fullyPaid.values, alpha=0.6)
ax.bar(chargedOff.index, chargedOff.values, alpha=0.7)
ax.set(title="Loan Status by Sub Grade", xlabel="Sub Grade", ylabel="Counts")


"""
int_rate & annual_inc
int_rate: Interest Rate on the loan
annual_inc: The self-reported annual income provided by the borrower during registration
"""
# barplot of int_rate by loan_status
fig, ax = plt.subplots(figsize=(12, 4))

sns.histplot(df, x="int_rate", bins=50, hue="loan_status")
ax.set(title="Loan Status by Interest Rate", xlabel="Interest Rate", ylabel="Counts")

# barplot of annual_inc by loan_status
fig, ax = plt.subplots(figsize=(12, 4))

sns.histplot(df, x="annual_inc", bins=1500, hue="loan_status", alpha=0.7)
ax.set(
    title="Loan Status by Self-reported Annual Income",
    xlabel="Self-reported Annual Income",
    ylabel="Counts",
)
ax.set_xlim(0, 300000)


"""
issue_d & earliest_cr_line
issue_d The month which the loan was funded
earliest_cr_line The month the borrower's earliest reported credit line was opened
"""
# barplot of issue_d and earliest_cr_line by loan_status
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0] = sns.histplot(df, x="issue_d", hue="loan_status", bins=40, ax=axes[0])
axes[0].set(
    title="Loan Status by Loan Issue Date", xlabel="Loan Issue Date", ylabel="Counts"
)

axes[1] = sns.histplot(df, x="earliest_cr_line", hue="loan_status", bins=40, ax=axes[1])
axes[1].set(
    title="Loan Status by The Earliest Reported Credit",
    xlabel="The Earliest Reported Credit",
    ylabel="Counts",
)


"""
dti, total_acc, revol_bal and revol_util
dti A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
total_acc The total number of credit lines currently in the borrower's credit file
revol_bal Total credit revolving balance
revol_util Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
"""

# barplot of dti by loan_status
fig, ax = plt.subplots(figsize=(12, 4))

df_dti = df[df["dti"] <= 50]
sns.histplot(df_dti, x="dti", hue="loan_status", bins=80)

ax.set(
    title="Loan Status by DTI( Debt divided by Self-reported Income)",
    xlabel="DTI( Debt divided by Self-reported Income)",
    ylabel="Counts",
)
ax.set_xlim(0, 50)

# barplot of total_acc by loan_status
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0] = sns.histplot(df, x="revol_util", hue="loan_status", bins=400, ax=axes[0])
axes[0].set_xlim(0, 120)
axes[0].set(title="Loan Status by revol_util")

axes[1] = sns.histplot(df, x="revol_bal", hue="loan_status", bins=1000)
axes[1].set_xlim(0, 80000)
axes[1].set(title="Loan Status by revol_bal")

# barplot of total_acc by loan_status
fig, ax = plt.subplots(figsize=(12, 4))
ax = sns.histplot(df, x="total_acc", hue="loan_status", bins=140)
ax.set_xlim(0, 80)
ax.set_title("Loan Status by The total number of credit lines ")


"""
pub_rec, mort_acc, pub_rec_bankruptcies and open_acc
pub_recThe number of derogatory public records
mort_accThe number of mortgage accounts
open_accThe number of open credit lines in the borrower's credit file.
pub_rec_bankruptciesThe number of public record bankruptcies
"""
# barplot of pub_rec, mort_acc, pub_rec_bankruptcies and open_acc by loan_status
fig, axes = plt.subplots(2, 2, figsize=(12, 10))


fullyPaid1 = df.loc[df["loan_status"] == "Fully Paid", "pub_rec"].value_counts()
chargedOff1 = df.loc[df["loan_status"] == "Charged Off", "pub_rec"].value_counts()

axes[0, 0].bar(x=fullyPaid1.index, height=fullyPaid1.values)
axes[0, 0].bar(x=chargedOff1.index, height=chargedOff1.values)
axes[0, 0].set_xlim(0, 5)
axes[0, 0].set(
    title="Loan Status by Number of derogatory public records",
    xlabel="pub_rec",
    ylabel="Counts",
)


fullyPaid2 = df.loc[df["loan_status"] == "Fully Paid", "mort_acc"].value_counts()
chargedOff2 = df.loc[df["loan_status"] == "Charged Off", "mort_acc"].value_counts()

axes[0, 1].bar(x=fullyPaid2.index, height=fullyPaid2.values)
axes[0, 1].bar(x=chargedOff2.index, height=chargedOff2.values)
axes[0, 1].set(
    title="Loan Status by The number of mortgage accounts",
    xlabel="mort_acc",
    ylabel="Counts",
)


fullyPaid3 = df.loc[df["loan_status"] == "Fully Paid", "open_acc"].value_counts()
chargedOff3 = df.loc[df["loan_status"] == "Charged Off", "open_acc"].value_counts()

axes[1, 0].bar(x=fullyPaid3.index, height=fullyPaid3.values)
axes[1, 0].bar(x=chargedOff3.index, height=chargedOff3.values)
axes[1, 0].set_xlim(0, 60)
axes[1, 0].set(
    title="Loan Status by The number of open credit lines",
    xlabel="open_acc",
    ylabel="Counts",
)


fullyPaid4 = df.loc[
    df["loan_status"] == "Fully Paid", "pub_rec_bankruptcies"
].value_counts()
chargedOff4 = df.loc[
    df["loan_status"] == "Charged Off", "pub_rec_bankruptcies"
].value_counts()

axes[1, 1].bar(x=fullyPaid4.index, height=fullyPaid4.values)
axes[1, 1].bar(x=chargedOff4.index, height=chargedOff4.values)
axes[1, 1].set(
    title="Loan Status by The number of public record bankruptcies",
    xlabel="pub_rec_bankruptcies",
    ylabel="Counts",
)


"""
verification_status, home_ownership, application_type and term
verification_statusIndicates if income was verified by LC, not verified, or if the income source was verified
home_ownershipThe home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER
application_typeIndicates whether the loan is an individual application or a joint application with two co-borrowers
termThe number of payments on the loan. Values are in months and can be either 36 or 60.
"""
# barplot of verification_status, home_ownership, application_type and term by loan_status
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

fullyPaid1 = df.loc[
    df["loan_status"] == "Fully Paid", "verification_status"
].value_counts()
chargedOff1 = df.loc[
    df["loan_status"] == "Charged Off", "verification_status"
].value_counts()

ax[0, 0].bar(x=fullyPaid1.index, height=fullyPaid1.values)
ax[0, 0].bar(x=chargedOff1.index, height=chargedOff1.values)
ax[0, 0].set(
    title="Loan Status by Verification Status",
    ylabel="Counts",
    xlabel="verification_status",
)


fullyPaid2 = df.loc[df["loan_status"] == "Fully Paid", "home_ownership"].value_counts()
chargedOff2 = df.loc[
    df["loan_status"] == "Charged Off", "home_ownership"
].value_counts()

ax[0, 1].bar(x=fullyPaid2.index, height=fullyPaid2.values)
ax[0, 1].bar(x=chargedOff2.index, height=chargedOff2.values)
ax[0, 1].set(
    title="Loan Status by The Home Ownership Status",
    ylabel="Counts",
    xlabel="home_ownership",
)


fullyPaid3 = df.loc[
    df["loan_status"] == "Fully Paid", "application_type"
].value_counts()
chargedOff3 = df.loc[
    df["loan_status"] == "Charged Off", "application_type"
].value_counts()

ax[1, 0].bar(x=fullyPaid3.index, height=fullyPaid3.values)
ax[1, 0].bar(x=chargedOff3.index, height=chargedOff3.values)
ax[1, 0].set(
    title="Loan Status by Application Type", ylabel="Counts", xlabel="application_type"
)


fullyPaid4 = df.loc[df["loan_status"] == "Fully Paid", "term"].value_counts()
chargedOff4 = df.loc[df["loan_status"] == "Charged Off", "term"].value_counts()

ax[1, 1].bar(x=fullyPaid4.index, height=fullyPaid4.values)
ax[1, 1].bar(x=chargedOff4.index, height=chargedOff4.values)
ax[1, 1].set(
    title="Loan Status by The Number of Payments on The Loan",
    ylabel="Counts",
    xlabel="term",
)

"""
emp_title, emp_length and  initial_list_status
emp_title The job title supplied by the Borrower when applying for the loan.*
emp_lengthEmployment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
initial_list_statusThe initial listing status of the loan. Possible values are – W, F
"""
# barplot of emp_title, emp_length and  initial_list_status by loan_status
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

empTitle = df["emp_title"].value_counts()[:10]
axes[0].bar(x=empTitle.index, height=empTitle.values)
axes[0].set(title="The most 10 jobs title afforded a loan")


fullyPaid1 = df.loc[df["loan_status"] == "Fully Paid", "emp_length"].value_counts()
chargedOff1 = df.loc[df["loan_status"] == "Charged Off", "emp_length"].value_counts()

axes[1].bar(x=fullyPaid1.index, height=fullyPaid1.values)
axes[1].bar(x=chargedOff1.index, height=chargedOff1.values)
axes[1].set(
    title="Loan Status by Employment Length", ylabel="Counts", xlabel="emp_length"
)


fullyPaid2 = df.loc[
    df["loan_status"] == "Fully Paid", "initial_list_status"
].value_counts()
chargedOff2 = df.loc[
    df["loan_status"] == "Charged Off", "initial_list_status"
].value_counts()

axes[2].bar(x=fullyPaid2.index, height=fullyPaid2.values)
axes[2].bar(x=chargedOff2.index, height=chargedOff2.values)
axes[2].set(
    title="Loan Status by The initial listing status of the loan",
    ylabel="Counts",
    xlabel="initial_list_status",
)
