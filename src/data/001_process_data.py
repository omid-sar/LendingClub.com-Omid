import numpy as np
import pandas as pd
import os
import sys

# -------------------------------------------------------------
# 1. Define Objectives
# -------------------------------------------------------------

"""
The objective of this script is to process raw data and export interim data.
The interim data will be used for exploratory data analysis and model development.
"""


# -------------------------------------------------------------
# 2. Read raw data
# -------------------------------------------------------------
sys.path.append("..")
df_org = pd.read_csv(
    "../../data/raw/lending_club_loan_two.csv",
    parse_dates=["issue_d", "earliest_cr_line"],
)
df_org.head(3)


# -------------------------------------------------------------
# 3. Explore raw data
# -------------------------------------------------------------

df_org.describe()
df = df_org
df_org.info()


# -------------------------------------------------------------
# 4.Export interim data
# -------------------------------------------------------------

df.to_pickle("../../data/interim/df.pkl")
