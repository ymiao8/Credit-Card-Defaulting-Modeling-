# Import necessary packages
import pandas as pd
import logging
import requests
import joblib


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# test functions
def clean_education_level(df):
    df.loc[(df.EDUCATION == 6), "EDUCATION"] = 5  # REPLACING 6 with 5 in education


def num_xs(x_encoded):
    return len(x_encoded.columns)  # gives the number of rows in the dataframe


def change_categorical(
    x,
):  # changes the variables into categorical variables with correct categories
    x["age_group"] = pd.Categorical(
        x["age_group"], categories=["20-30", "30-40", "40-50", "50-60", "60+"]
    )
    x["SEX"] = pd.Categorical(x["age_group"], categories=["1", "2"])
    x["limit_balance_group"] = pd.Categorical(
        x["limit_balance_group"],
        categories=["0-100k", "0-200k", "0-300k", "0-400k", "0-500k", "500k+"],
    )
    x["EDUCATION"] = pd.Categorical(
        x["limit_balance_group"], categories=["0", "1", "2", "3", "4", "5"]
    )
    x["MARRIAGE"] = pd.Categorical(x["MARRIAGE"], categories=["0", "1", "2", "3"])
    x["PAY_0"] = pd.Categorical(
        x["MARRIAGE"],
        categories=["-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    )
    x["PAY_2"] = pd.Categorical(
        x["MARRIAGE"],
        categories=["-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    )
    x["PAY_3"] = pd.Categorical(
        x["MARRIAGE"],
        categories=["-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    )
    x["PAY_4"] = pd.Categorical(
        x["MARRIAGE"],
        categories=["-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    )
    x["PAY_5"] = pd.Categorical(
        x["MARRIAGE"], categories=["-1", "0", "2", "3", "4", "5", "6", "7", "8", "9"]
    )
    x["PAY_6"] = pd.Categorical(
        x["MARRIAGE"], categories=["-1", "0", "2", "3", "4", "5", "6", "7", "8", "9"]
    )


def model_output(data):  # gives the model ouput predicted from input
    model = requests.get(
        url="https://stats404creditproject.s3-us-west-1.amazonaws.com/randomforest.joblib",
        allow_redirects=True,
    )
    open("randomforest.joblib", "wb").write(model.content)
    model = joblib.load("randomforest.joblib")
    df = pd.DataFrame(data)
    return model.predict(df)
