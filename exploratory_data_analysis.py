# imports
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import re


def clean(df):
    # Delete rows 0,1
    df = df.drop(labels=[0,1])
    # Make header names
    column_names = ["timestamp", "major", "ml_course", "information_retrieval_course", "statistics_course",
                    "database_course", "gender", "used_chatgpt", "birthday", "no_students", "stand_up",
                    "stress_level", "sport_hours", "rand_number", "bedtime", "good_day_1", "good_day_2"]
    df.columns = column_names

    # Clean column major
    print(df.loc[:,"major"].to_string())
    ai_master = df.loc[:,"major"].str.contains(r"AI")
    ai_master = ai_master | df.loc[:,"major"].str.match(re.compile("ai", re.IGNORECASE))
    ai_master = ai_master | df.loc[:, "major"].str.contains(re.compile("Artificial Intelligence|master ai", re.IGNORECASE))

    cs_master = df.loc[:,"major"].str.contains(r"CS")
    cs_master = cs_master | df.loc[:, "major"].str.contains(re.compile("Computer Science|big data engineering|"
                                                                       "software engineering|green it|computing and concurrency|"
                                                                       "computer systems", re.IGNORECASE))
    cls_master = df.loc[:,"major"].str.contains(r"CLS")
    cls_master = cls_master | df.loc[:, "major"].str.contains(re.compile("computational science", re.IGNORECASE))

    bioinformatics = df.loc[:,"major"].str.contains(re.compile("bioinformatics", re.IGNORECASE))
    business_analytics = df.loc[:,"major"].str.contains(r"BA")
    business_analytics = business_analytics | df.loc[:,"major"].str.contains(re.compile("business analytics", re.IGNORECASE))

    df["major_cleaned"] = "other"
    assert not any(cls_master & ai_master & cs_master & bioinformatics), "some strings match multiple master categories"
    for i in df.index:
        if ai_master[i] :
            df.major_cleaned[i] = "AI"
        if cs_master[i] :
            df.major_cleaned[i] = "CS"
        if cls_master[i]:
            df.major_cleaned[i] = "CLS"
        if bioinformatics[i]:
            df.major_cleaned[i] = "bioinformatics"
        if business_analytics[i] :
            df.major_cleaned[i] = "businessAnalytics"


    print(df.loc[:,"major_cleaned"].to_string())
    print(df["major_cleaned"].value_counts())
    df.to_excel("cleaned_df.xlsx")

    # ignore column birth date

    # clean column bed time

    # clean column sports

    # remove outliers


if __name__ == '__main__':
    # load in data file
    data = pd.read_csv('ODI-2024.csv')
    clean(data)
