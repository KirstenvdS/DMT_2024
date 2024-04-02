# imports
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd


def clean(df):
    # Delete rows 0,1
    df = df.drop(labels=[0,1])
    # Make header names
    column_names = ["timestamp", "major", "ml_course", "information_retrieval_course", "statistics_course",
                    "database_course", "gender", "used_chatgpt", "birthday", "no_students", "stand_up",
                    "stress_level", "sport_hours", "rand_number", "bedtime", "good_day_1", "good_day_2"]
    df.columns = column_names

    print(df.shape)
    print(df.describe)
    print(df.head())

if __name__ == '__main__':
    # load in data file
    data = pd.read_csv('ODI-2024.csv')
    clean(data)
