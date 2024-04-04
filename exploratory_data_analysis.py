# imports
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import re


def clean(df):
    # Delete rows 0,1
    df = df.drop(labels=[0, 1])
    # Make header names
    column_names = ["timestamp", "major", "ml_course", "information_retrieval_course", "statistics_course",
                    "database_course", "gender", "used_chatgpt", "birthday", "no_students", "stand_up",
                    "stress_level", "sport_hours", "rand_number", "bedtime", "good_day_1", "good_day_2"]
    df.columns = column_names

    # Clean column major
    ai_master = df.loc[:, "major"].str.contains(r"AI")
    ai_master = ai_master | df.loc[:, "major"].str.match(re.compile("ai", re.IGNORECASE))
    ai_master = ai_master | df.loc[:, "major"].str.contains(
        re.compile("Artificial Intelligence|master ai", re.IGNORECASE))

    cs_master = df.loc[:, "major"].str.contains(r"CS")
    cs_master = cs_master | df.loc[:, "major"].str.contains(re.compile("Computer Science|big data engineering|"
                                                                       "software engineering|green it|"
                                                                       "computing and concurrency|"
                                                                       "computer systems", re.IGNORECASE))
    cls_master = df.loc[:, "major"].str.contains(r"CLS")
    cls_master = cls_master | df.loc[:, "major"].str.contains(re.compile("computational science", re.IGNORECASE))

    bioinformatics = df.loc[:, "major"].str.contains(re.compile("bioinformatics", re.IGNORECASE))
    business_analytics = df.loc[:, "major"].str.contains(r"BA")
    business_analytics = business_analytics | df.loc[:, "major"].str.contains(
        re.compile("business analytics", re.IGNORECASE))

    df["major_cleaned"] = "other"
    assert not any(cls_master & ai_master & cs_master & bioinformatics), "some strings match multiple master categories"
    for i in df.index:
        if ai_master[i]:
            df.loc[i,"major_cleaned"] = "AI"
        if cs_master[i]:
            df.loc[i,"major_cleaned"] = "CS"
        if cls_master[i]:
            df.loc[i,"major_cleaned"] = "CLS"
        if bioinformatics[i]:
            df.loc[i,"major_cleaned"] = "bioinformatics"
        if business_analytics[i]:
            df.loc[i,"major_cleaned"] = "businessAnalytics"

    print("Number of students per major: \n", df["major_cleaned"].value_counts(dropna=False))

    # ignore column birth date, too complicated

    # clean column bed time manually
    # 1 means "0:01-1:00"
    # 2 means "1:01-2:00"
    # 3 means "2:01-3:00"
    # aso.
    # 23 means "22:01 - 23:00"
    # 0 means "23:01 - 00:00"
    # -1 means "not recoverable or meaningful or missing value"
    bedtimes = [1, 1, 0, 2, 0, 23, 0, 3, 1, 1, 23, 5, 0, 2, 18, 0, 3, 2, 0, 0, 1, 2, 0, 1, 22, 4, 0, 0, 2, 23, 4, 23, 1,
                2, 1, 2, 12, 1, 1, 5, 2, 22, 0, 1, 21, 0, 4, 3, 0, 0, 1, 3,2, 1, 0, 1, 0, 0, 2, 2, 2, 0, 2, 1, 1, 12, 4,
                3, 23, 23, 0, 0, 3, 3, 6, 1, 3, 4, 1, 1, 22, 2, 2, 1, 0, 2, 5, 3, 0, 1, 2, 23, 22, 1, 1, 0, 1, 1, 3, 1,
                1, 0, 0, 0, 2, 22, 23, 2, 2, 23, 4, 0, 2, 18, 23, 0, 2, 1, 1, 3, 2, 1, 0, 0, 1, 2, 3, 1, 0, 0, 14, 1, 1,
                1, 3, 7, 0, 19, 2, 23, 2, 2, 2, 2, 3, 1, 0, 2, 0, 23, 22, 0, 3, 1, 0, 2, 22, 22, 0, 4, 1, 3, 0, 1, 1, 0,
                23, 1, 23, 3, 1, 3, 4, 1, 1, 23, 1, 1, 1, 0, 4, 1, 0, 0, 2, 0, 1, 7, 23, 23, 0, 2, 1, 2, 2, 1, 0, 1, 3,
                2, 23, 6, 0, 3, 2, 2, 3, 23, 1, -1, 0, 4, 1, 3, 1, 0, 2, 4, 4, 1, 23, 3, 23, 3, 2, 4, 0, 1, 2, 0, 3, 0,
                3, 0, 0, 0, -1, 2, 2, 23, 1, 0, 23, 0, 3]
    df["bedtimes_cleaned"] = bedtimes
    # remove non-sensical data, anything between 6-19 o'clock makes no sense
    df.loc[np.isin(df["bedtimes_cleaned"], range(6,19)), "bedtimes_cleaned"] = pd.NA
    df.loc[df["bedtimes_cleaned"] == -1, "bedtimes_cleaned"] = pd.NA
    print("Bedtime cleaned: \n", df["bedtimes_cleaned"].value_counts(dropna=False).to_string())

    # Gone to bed late? 1-5 AM
    df["bed_late"] = np.isin(df["bedtimes_cleaned"], range(1,6))
    print("Gone to bed late? \n", df["bed_late"].value_counts(dropna=False).to_string())

    # clean column sports
    #print("Raw number of sport hours mentioned: \n", df["sport_hours"].value_counts())
    df["sport_cleaned"] = df["sport_hours"]
    df.loc[df["sport_cleaned"] == "6 hours", "sport_cleaned"] = "6"
    df.loc[df["sport_cleaned"] == "Whole 50","sport_cleaned"] = "50"
    df.loc[df["sport_cleaned"] == "1/2", "sport_cleaned"] = "0.5"
    df.loc[df["sport_cleaned"] == "geen", "sport_cleaned"] = "0"
    df.loc[df["sport_cleaned"] == "2-4", "sport_cleaned"] = "3"
    df["sport_cleaned"] = pd.to_numeric(df.loc[:, "sport_cleaned"])

    # remove outliers column sport:
    # remove impossible values 632 (90 hours per day) and 57 (8 hours per day) and 50 (7 hours per day)
    df.loc[df["sport_cleaned"] == 632.0, "sport_cleaned"] = pd.NA
    df.loc[df["sport_cleaned"] == 57.0, "sport_cleaned"] = pd.NA
    df.loc[df["sport_cleaned"] == 50.0, "sport_cleaned"] = pd.NA

    print("Distribution of number of sport hours cleaned: \n", df["sport_cleaned"].value_counts(dropna=False))

    # clean column stress level
    #print("Raw Stress level : \n", df["stress_level"].value_counts().to_string())
    # remove impossible values < 0 or > 100
    df["stress_cleaned"] = df["stress_level"]
    df["stress_cleaned"] = pd.to_numeric(df.loc[:,"stress_cleaned"])
    df.loc[(df["stress_cleaned"] < 0) | (df["stress_cleaned"] > 100), "stress_cleaned"] = pd.NA
    print("Stress level cleaned: \n", df["stress_cleaned"].value_counts(dropna=False).to_string())

    # clean column estimate number of students
    print("Raw no students estimate : \n", df["no_students"].value_counts(dropna=False).to_string())
    df["no_students_cleaned"] = df["no_students"]
    df.loc[df["no_students_cleaned"] == "Two hundred fifty", "no_students_cleaned"] = "250"
    df.loc[df["no_students_cleaned"] == "Around200", "no_students_cleaned"] = "200"
    df.loc[df["no_students_cleaned"] == "3thousand", "no_students_cleaned"] = "3000"
    df.loc[df["no_students_cleaned"] == "~280", "no_students_cleaned"] = "280"
    df.loc[df["no_students_cleaned"] == "150-200", "no_students_cleaned"] = "175"
    df.loc[df["no_students_cleaned"] == "over 9000", "no_students_cleaned"] = "9000"
    df.loc[df["no_students_cleaned"] == "Around 200", "no_students_cleaned"] = "200"
    df.loc[df["no_students_cleaned"] == "1 million", "no_students_cleaned"] = "1000000"
    df["no_students_cleaned"] = pd.to_numeric(df.loc[:,"no_students_cleaned"])

    # Remove impossible and silly values (< 20 or > 1000)
    df.loc[(df["no_students_cleaned"] < 20) | (df["no_students_cleaned"] > 1000), "no_students_cleaned"] = pd.NA
    print("Cleaned no students estimate : \n", df["no_students_cleaned"].value_counts(dropna=False).to_string())

    # return new dataframe
    return df

def explore_data(df):
    subdf = df[["stress_cleaned", "sport_cleaned", "bed_late", "no_students_cleaned"]]
    print(subdf.corr(numeric_only=True).to_string())
    return


if __name__ == '__main__':
    # load in data file
    data = pd.read_csv('ODI-2024.csv')
    data = clean(data)
    explore_data(data)
