import pytest
import pandas as pd
from functions.function import clean_education_level, model_output, change_categorical
import numpy
import joblib
import requests


data = {
    "ID": [3.0, 4.0],
    "LIMIT_BAL": [87700.0, 90000.0],
    "SEX": [2.0, 2.0],
    "EDUCATION": [2.0, 3.0],
    "MARRIAGE": [2.0, 1.0],
    "AGE": [34.0, 26.0],
    "PAY_0": [0.0, 0.0],
    "PAY_2": [0.0, 0.0],
    "PAY_3": [0.0, 0.0],
    "PAY_4": [0.0, 0.0],
    "PAY_5": [0.0, 1.0],
    "PAY_6": [1.0, 0.0],
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
    "default.payment.next.month": [1.0, 0.0],
}

df = pd.DataFrame(data)


# integration test for both model_output, clean_education_level and change_categorical
def model_output():
    clean_education_level(df)
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
    output = model_output(df)
    assert (
        np.isnan(output) == False
    ), """testing minority observations
        """


def test_changecat():  # unit test to check if we correctly transform these variables into correct categories
    df["age_group"] = pd.cut(
        df["AGE"],
        bins=[20, 30, 40, 50, 60, 80],
        labels=["20-30", "30-40", "40-50", "50-60", "60+"],
    )  # change age into categorical variable )
    df["limit_balance_group"] = pd.cut(
        df["LIMIT_BAL"],
        bins=[0, 100000, 200000, 300000, 400000, 500000, 800000],
        labels=["0-100k", "0-200k", "0-300k", "0-400k", "0-500k", "500k+"],
    )
    df["SEX"].dtype.name == df["limit_balance_group"].dtype.name == df[
        "MARRIAGE"
    ].dtype.name == df["PAY_0"].dtype.name == df["PAY_3"].dtype.name == df[
        "PAY_2"
    ].dtype.name == df[
        "PAY_4"
    ].dtype.name == df[
        "PAY_5"
    ].dtype.name == df[
        "PAY_6"
    ].dtype.name == "category", """checking whether variables are correctly transformed into category
        """
