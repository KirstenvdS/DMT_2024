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
    print(df.loc[:, "major"].to_string())
    ai_master = df.loc[:, "major"].str.contains(r"AI")
    ai_master = ai_master | df.loc[:, "major"].str.match(re.compile("ai", re.IGNORECASE))
    ai_master = ai_master | df.loc[:, "major"].str.contains(
        re.compile("Artificial Intelligence|master ai", re.IGNORECASE))

    cs_master = df.loc[:, "major"].str.contains(r"CS")
    cs_master = cs_master | df.loc[:, "major"].str.contains(re.compile("Computer Science|big data engineering|"
                                                                       "software engineering|green it|computing and concurrency|"
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
            df.major_cleaned[i] = "AI"
        if cs_master[i]:
            df.major_cleaned[i] = "CS"
        if cls_master[i]:
            df.major_cleaned[i] = "CLS"
        if bioinformatics[i]:
            df.major_cleaned[i] = "bioinformatics"
        if business_analytics[i]:
            df.major_cleaned[i] = "businessAnalytics"

    print(df["major_cleaned"].value_counts())

    # ignore column birth date, too complicated

    # clean column bed time manually
    # 1 means "0:01-1:00"
    # 2 means "1:01-2:00"
    # 3 means "2:01-3:00"
    # aso.
    # 23 means "22:01 - 23:00"
    # 0 means "23:01 - 00:00"
    bedtimes = [1, 1, 0, 2, 0, 23, 0, 3, 1, 1, 23, 5, 0, 2, 18, 0, 3, 2, 0, 0, 1, 2, 0, 1, 22, 4, 0, 0, 2, 23, 4, 23, 1,
                2, 1, 2, 12, 1, 1, 5, 2, 22, 0, 1, 21, 0, 4, 3, 0, 0, 1, 3,2, 1, 0, 1, 0, 0, 2, 2, 2, 0, 2, 1, 1, 12, 4,
                3, 23, 23, 0, 0, 3, 3, 6, 1, 3, 4, 1, 1, 22, 2, 2, 1, 0, 2, 5, 3, 0, 1, 2, 23, 22, 1, 1, 0, 1, 1, 3, 1,
                1, 0, 0, 0, 2, 22, 23, 2, 2, 23, 4, 0, 2, 18, 23, 0, 2, 1, 1, 3, 2, 1, 0, 0, 1, 2, 3, 1, 0, 0, 14, 1, 1,
                1, 3, 7, 0, 19, 2, 23, 2, 2, 2, 2, 3, 1, 0, 2, 0, 23, 22, 0, 3, 1, 0, 2, 22, 22, 0, 4, 1, 3, 0, 1, 1, 0,
                23, 1, 23, 3, 1, 3, 4, 1, 1, 23, 1, 1, 1, 0, 4, 1, 0, 0, 2, 0, 1, 7, 23, 23, 0, 2, 1, 2, 2, 1, 0, 1, 3,
                2, 23, 6, 0, 3, 2, 2, 3, 23, 1, -1, 0, 4, 1, 3, 1, 0, 2, 4, 4, 1, 23, 3, 23, 3, 2, 4, 0, 1, 2, 0, 3, 0,
                3, 0, 0, 0, -1, 2, 2, 23, 1, 0, 23, 0, 3]
    df["bedtimes_cleaned"] = bedtimes

    # clean column sports

    # remove outliers


if __name__ == '__main__':
    # load in data file
    data = pd.read_csv('ODI-2024.csv')
    clean(data)
