# Import necessary packages
import pandas as pd
import logging
import requests
import joblib
from functions.function import (
    clean_education_level,
    num_xs,
    model_output,
    change_categorical,
)


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# Feature Engineering
if __name__ == "__main__":
    # You can edit the below dictionary, when editing please follow the instructions on readme
    df = pd.DataFrame(
        {
            "ID": [3.0, 4.0],
            "LIMIT_BAL": [87700.0, 90000.0],
            "SEX": [2.0, 2.0],
            "EDUCATION": [2.0, 3.0],
            "MARRIAGE": [2.0, 1.0],
            "AGE": [34.0, 26.0],
            "PAY_0": [0.0, 2.0],
            "PAY_2": [0.0, 2.0],
            "PAY_3": [1.0, 0.0],
            "PAY_4": [3.0, 0.0],
            "PAY_5": [3.0, 0.0],
            "PAY_6": [2.0, 0.0],
            "BILL_AMT1": [0.0, 29239.0],
            "BILL_AMT2": [3654.0, 14027.0],
            "BILL_AMT3": [1234.0, 15549.0],
            "BILL_AMT4": [1234.0, 15549.0],
            "BILL_AMT5": [1234.0, 15549.0],
            "BILL_AMT6": [1234.0, 15549.0],
            "PAY_AMT1": [1500.0, 1500.0],
            "PAY_AMT2": [1500.0, 1500.0],
            "PAY_AMT3": [3651.0, 1000.00],
            "PAY_AMT4": [1234.0, 1000.0],
            "PAY_AMT5": [2680.0, 1000.0],
            "PAY_AMT6": [2600.0, 5000.0],
        }
    )

    LOGGER.info("Feature Engineering")
    clean_education_level(df)  # REPLACING 6 with 5 in education
    df["age_group"] = pd.cut(
        df["AGE"],
        bins=[20, 30, 40, 50, 60, 80],
        labels=[
            "20-30",
            "30-40",
            "40-50",
            "50-60",
            "60+",
        ],  # change age into categorical variable
    )
    df["limit_balance_group"] = pd.cut(
        df["LIMIT_BAL"],
        bins=[0, 100000, 200000, 300000, 400000, 500000, 800000],
        labels=["0-100k", "0-200k", "0-300k", "0-400k", "0-500k", "500k+"],
    )  # change limit balance into categorical variable
    x = df.drop(["ID", "AGE", "LIMIT_BAL"], axis=1)  # drop unnecessary columns
    change_categorical(x)
    # one hot coding for xs
    x_encoded = pd.get_dummies(
        x,
        columns=[
            "age_group",
            "limit_balance_group",
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
        ],
        drop_first=True,
    )
    LOGGER.info("Model Predicting...")
    if num_xs(x_encoded) == 88:
        print("Please Wait for the result...")
        print(model_output(x_encoded))
    else:
        print("Check the code, something is wrong with feature engineering!")
