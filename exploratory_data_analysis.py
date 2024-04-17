# imports
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import re
from scipy.stats import chi2_contingency
import seaborn as sns
from dateutil.parser import parse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Colors suitable for color blindness
color_codes = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]


def birthyear(s):
    # get year from fuzzy birthday string
    x = 0
    try:
        x = parse(s, fuzzy=True).year
    except:
        print("Year not found in string!")
    return x


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
            df.loc[i, "major_cleaned"] = "AI"
        if cs_master[i]:
            df.loc[i, "major_cleaned"] = "CS"
        if cls_master[i]:
            df.loc[i, "major_cleaned"] = "CLS"
        if bioinformatics[i]:
            df.loc[i, "major_cleaned"] = "bioInf"
        if business_analytics[i]:

            df.loc[i, "major_cleaned"] = "BA"

    # print("Number of students per major: \n", df["major_cleaned"].value_counts(dropna=False))

    # clean column bed time manually
    # 1 means "0:01-1:00"
    # 2 means "1:01-2:00"
    # 3 means "2:01-3:00"
    # aso.
    # 23 means "22:01 - 23:00"
    # 0 means "23:01 - 00:00"
    # -1 means "not recoverable or meaningful or missing value"
    bedtimes = [1, 1, 0, 2, 0, 23, 0, 3, 1, 1, 23, 5, 0, 2, 18, 0, 3, 2, 0, 0, 1, 2, 0, 1, 22, 4, 0, 0, 2, 23, 4, 23, 1,
                2, 1, 2, 12, 1, 1, 5, 2, 22, 0, 1, 21, 0, 4, 3, 0, 0, 1, 3, 2, 1, 0, 1, 0, 0, 2, 2, 2, 0, 2, 1, 1, 12,
                4,
                3, 23, 23, 0, 0, 3, 3, 6, 1, 3, 4, 1, 1, 22, 2, 2, 1, 0, 2, 5, 3, 0, 1, 2, 23, 22, 1, 1, 0, 1, 1, 3, 1,
                1, 0, 0, 0, 2, 22, 23, 2, 2, 23, 4, 0, 2, 18, 23, 0, 2, 1, 1, 3, 2, 1, 0, 0, 1, 2, 3, 1, 0, 0, 14, 1, 1,
                1, 3, 7, 0, 19, 2, 23, 2, 2, 2, 2, 3, 1, 0, 2, 0, 23, 22, 0, 3, 1, 0, 2, 22, 22, 0, 4, 1, 3, 0, 1, 1, 0,
                23, 1, 23, 3, 1, 3, 4, 1, 1, 23, 1, 1, 1, 0, 4, 1, 0, 0, 2, 0, 1, 7, 23, 23, 0, 2, 1, 2, 2, 1, 0, 1, 3,
                2, 23, 6, 0, 3, 2, 2, 3, 23, 1, -1, 0, 4, 1, 3, 1, 0, 2, 4, 4, 1, 23, 3, 23, 3, 2, 4, 0, 1, 2, 0, 3, 0,
                3, 0, 0, 0, -1, 2, 2, 23, 1, 0, 23, 0, 3]
    df["bedtimes_cleaned"] = bedtimes
    df.loc[df["bedtimes_cleaned"] == -1, "bedtimes_cleaned"] = pd.NA


    # print("Bedtime cleaned: \n", df["bedtimes_cleaned"].value_counts(dropna=False).to_string())

    # Gone to bed late? 1-5 AM
    df["bed_late"] = np.isin(df["bedtimes_cleaned"], range(1, 6))
    # print("Gone to bed late? \n", df["bed_late"].value_counts(dropna=False).to_string())

    # clean column sports
    # print("Raw number of sport hours mentioned: \n", df["sport_hours"].value_counts())
    df["sport_cleaned"] = df["sport_hours"]
    df.loc[df["sport_cleaned"] == "6 hours", "sport_cleaned"] = "6"
    df.loc[df["sport_cleaned"] == "Whole 50", "sport_cleaned"] = "50"
    df.loc[df["sport_cleaned"] == "1/2", "sport_cleaned"] = "0.5"
    df.loc[df["sport_cleaned"] == "geen", "sport_cleaned"] = "0"
    df.loc[df["sport_cleaned"] == "2-4", "sport_cleaned"] = "3"
    df["sport_cleaned"] = pd.to_numeric(df.loc[:, "sport_cleaned"])

    # print("Distribution of number of sport hours cleaned: \n", df["sport_cleaned"].value_counts(dropna=False))

    # clean column stress level
    # print("Raw Stress level : \n", df["stress_level"].value_counts().to_string())
    # remove impossible values < 0 or > 100
    df["stress_cleaned"] = df["stress_level"]

    df["stress_cleaned"] = pd.to_numeric(df.loc[:, "stress_cleaned"])

    # clean column estimate number of students
    # print("Raw no students estimate : \n", df["no_students"].value_counts(dropna=False).to_string())

    df["no_students_cleaned"] = df["no_students"]
    df.loc[df["no_students_cleaned"] == "Two hundred fifty", "no_students_cleaned"] = "250"
    df.loc[df["no_students_cleaned"] == "Around200", "no_students_cleaned"] = "200"
    df.loc[df["no_students_cleaned"] == "3thousand", "no_students_cleaned"] = "3000"
    df.loc[df["no_students_cleaned"] == "~280", "no_students_cleaned"] = "280"
    df.loc[df["no_students_cleaned"] == "150-200", "no_students_cleaned"] = "175"
    df.loc[df["no_students_cleaned"] == "over 9000", "no_students_cleaned"] = "9000"
    df.loc[df["no_students_cleaned"] == "Around 200", "no_students_cleaned"] = "200"
    df.loc[df["no_students_cleaned"] == "1 million", "no_students_cleaned"] = "1000000"
    df["no_students_cleaned"] = pd.to_numeric(df.loc[:, "no_students_cleaned"])
    #print("Cleaned no students estimate : \n", df["no_students_cleaned"].value_counts(dropna=False).to_string())


    # Error of estimation
    true_n_students = len(df)
    df["err_no_students"] = abs(df.loc[:, "no_students_cleaned"] - true_n_students)


    # age
    df["birthyear"] = df['birthday'].apply(birthyear)
    df["age"] = 2024 - df["birthyear"]

    # return new dataframe
    return df


def remove_outliers(df):
    # bed times
    # remove non-sensical data, anything between 6-19 o'clock makes no sense
    df.loc[np.isin(df["bedtimes_cleaned"], range(6, 19)), "bedtimes_cleaned"] = pd.NA

    # sport
    # remove impossible values 632 (90 hours per day) and 57 (8 hours per day) and 50 (7 hours per day)
    df.loc[df["sport_cleaned"] == 632.0, "sport_cleaned"] = pd.NA
    df.loc[df["sport_cleaned"] == 57.0, "sport_cleaned"] = pd.NA
    df.loc[df["sport_cleaned"] == 50.0, "sport_cleaned"] = pd.NA

    # Stress
    # Delete impossible values everywhere because did not answer honestly
    df.loc[(df["stress_cleaned"] < 0) | (df["stress_cleaned"] > 100), "stress_cleaned"] = pd.NA

    # No students estimate
    # Remove impossible and silly values (< 20 or > 600)
    df.loc[(df["no_students_cleaned"] < 20) | (df["no_students_cleaned"] > 600),
        "no_students_cleaned"] = pd.NA

    # re-calculate error of estimation of number of students
    true_n_students = len(df)
    df["err_no_students"] = abs(df.loc[:, "no_students_cleaned"] - true_n_students)

    # Age
    df.loc[(df["age"] < 18) | (df["age"] > 80), "age"] = pd.NA

    # Gender
    df.loc[(df["gender"] == "non-binary") |
           (df["gender"] == "gender fluid") |
           (df["gender"] == "other"), "gender"] = "other"

    return df


def explore_data(df):

    # Number of observations
    N = len(df)
    print("Number of observations: ", N)

    # Number of columns
    M = len(df.columns)
    print("Number of columns: ", M)

    ##################### Numerical values: min, max, mean, median, var, NAs
    # age
    print("Birthyear: ")
    print(df["birthyear"].value_counts(dropna=False))
    print(df["birthyear"].describe())
    print("Age: ")
    print(df["age"].value_counts(dropna=False))
    print(df["age"].describe())
    df.hist(column="age", bins=25)
    # plt.xlabel("Age in years")
    # plt.ylabel("Frequency")
    # plt.title("Age")
    # plt.show()

    # no students estimation
    print("No Students: ")
    print(df["no_students_cleaned"].value_counts(dropna=False))
    print(df["no_students_cleaned"].describe())

    # stress level
    print("Stess level: ")
    print(df["stress_cleaned"].value_counts(dropna=False))
    print(df["stress_cleaned"].describe())

    # sport hours
    print("Sport hours: ")
    print(df["sport_cleaned"].value_counts(dropna=False))
    print(df["sport_cleaned"].describe())

    # bedtime
    print("Bedtime: ")
    print(df["bedtimes_cleaned"].value_counts(dropna=False))
    df["bedtimes_cleaned_t"] = df["bedtimes_cleaned"] - 1
    df.loc[(df["bedtimes_cleaned_t"] < 0), "bedtimes_cleaned_t"] = 23
    df.hist("bedtimes_cleaned_t", bins=24, color=color_codes[0])
    # plt.xlabel("Time in hours")
    # plt.ylabel("Frequency")
    # plt.title("Bedtimes")
    # plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22])
    # plt.savefig("bedtimes_hist.png")
    # plt.show()

    # ignore random number because it doesn't make sense

    #################### Categorical values: counts, NAs
    # major
    print("Major: ")
    print(df["major_cleaned"].value_counts(dropna=False))

    # ml course
    print("ML course: ")
    print(df["ml_course"].value_counts(dropna=False))

    # information retrieval
    print("Information retrieval course: ")
    print(df["information_retrieval_course"].value_counts(dropna=False))

    # statistics course
    print("Statistics course: ")
    print(df["statistics_course"].value_counts(dropna=False))

    # database course
    print("Database course: ")
    print(df["database_course"].value_counts(dropna=False))

    # gender
    print("Gender: ")
    print(df["gender"].value_counts(dropna=False))

    # used chatgpt
    print("Used chatgpt: ")
    print(df["used_chatgpt"].value_counts(dropna=False))

    # stand up
    print("Stand up: ")
    print(df["stand_up"].value_counts(dropna=False))

    # good day categories/wordcloud
    print("Good day: ")
    categories = ["social", "weather", "exercise", "food", "mental_health", "sleep"]
    counts = np.zeros(len(categories))
    for i in range(len(categories)):
        counts[i] = df['happy_' + categories[i]].values.sum()
    counts, categories = zip(*sorted(zip(counts, categories), reverse=True))
    # plt.bar(categories, counts, color=color_codes[0])
    # plt.title("Categories mentioned in \"What makes a good day for you?\"")
    # plt.xlabel("Category")
    # plt.ylabel("Frequency")
    # plt.savefig("good_day_categories_barplot.png")
    # plt.show()

def plot_cleaned_data(df):
    ##################### Interesting properties
    # major and stress: CS more stressed than others
    df.boxplot(column="stress_cleaned", by="major_cleaned")
    plt.title("Boxplot stress grouped by major")
    plt.suptitle('')
    plt.xlabel("Major")
    plt.ylabel("Stress level")
    plt.savefig("stress_major_boxplot.png")
    plt.show()

    df.boxplot(column="stress_cleaned", by="gender")
    plt.title("Boxplot stress grouped by gender")
    plt.suptitle('')
    plt.xlabel("Gender")
    plt.ylabel("Stress level")
    plt.savefig("stress_gender_boxplot.png")
    plt.show()

    # Sport hours by people who mentioned exercise or gym in their categories
    df.boxplot(column="sport_cleaned", by="happy_exercise")
    plt.suptitle('')
    plt.title("Number of hours of exercise per week \nif \"exercise\" "
              "mentioned in \"What makes a good day?\"")
    plt.xlabel("\"Exercise\" mentioned in \"What makes a good day?\"")
    plt.ylabel("Amoung of exercise (# hours/week)")
    plt.savefig("sport_hours_exercise.png")
    plt.show()

    # correlations
    dfsub = df[['stressed', 'bed_late', 'err_no_students','happy_weather', 'happy_food',
                'happy_mental_health', 'happy_exercise', 'happy_sleep', 'happy_social']]
    matrix = dfsub.corr().round(2)
    print(matrix)
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    fig = plt.figure()
    sns.heatmap(matrix, annot=True,vmax=1, vmin=-1, center=0, cmap='vlag', mask=mask)
    plt.title("Correlation Matrix of various numerical outcomes")
    ax = plt.gca()
    ax.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()
    fig.savefig("corr_matrix_heatmap.png")
    plt.show()


def clean_classification(data):
    data_copy = data.copy()

    # create copy with only lowercase letters in strings
    data_copy["good_day_1"] = data_copy["good_day_1"].astype("string")
    data_copy["good_day_2"] = data_copy["good_day_2"].astype("string")
    data_copy["good_day_1"] = data_copy["good_day_1"].str.lower()
    data_copy["good_day_2"] = data_copy["good_day_2"].str.lower()

    # dictionary of all categories and buzzwords
    all_buzzwords = {
        'social': "friend|social|family|sex|party",
        'weather': "weather|sun|sky",
        'exercise': "sports|gym|exercise|working out",
        'food': "brownie|food|coffee|water|bread|pizza|meal|tea|lunch|dinner|breakfast|eat",
        'mental_health': "stress|mental|relax|rest",
        'sleep': 'sleep'
    }

    # code checks whether buzzwords are in the string per column and then adds the booleans
    for category in all_buzzwords.keys():
        data_copy["A"] = data_copy["good_day_1"].str.contains(all_buzzwords[category])
        data_copy["B"] = data_copy["good_day_2"].str.contains(all_buzzwords[category])
        data_copy['happy_' + category] = data_copy["A"] + data_copy["B"]
        data_copy = data_copy.drop(["A", "B"], axis=1)

    # create stressed/not stressed boolean
    data_copy['stressed'] = np.where(data_copy['stress_level'] > 50, True, False)


    #print(data.corr(numeric_only=True).to_string())
    return data_copy

def classification_prep(data):
    data['ml_course_categorical'] = pd.Categorical(data["ml_course"], categories=["yes", "no"], ordered=False)
    data['information_retrieval_course_categorical'] = pd.Categorical(data["information_retrieval_course"],
                                                                           categories=["1", "0"], ordered=False)
    data['statistics_course_categorical'] = pd.Categorical(data["statistics_course"],
                                                                categories=["mu", "sigma"], ordered=False)
    data['database_course'] = pd.Categorical(data["database_course"],
                                                                categories=["ja", "nee"], ordered=False)
    data['used_chatgpt'] = pd.Categorical(data["used_chatgpt"],
                                                                categories=["yes", "no"], ordered=False)

    subdata = data[["major_cleaned", 'ml_course_categorical', 'information_retrieval_course_categorical',
                    'statistics_course_categorical', 'database_course', 'used_chatgpt', 'gender']]

    for i in subdata.columns:
        for j in subdata.columns:
            CrosstabResult = pd.crosstab(index=subdata[i], columns=subdata[j])
            ChiSqResult = chi2_contingency(CrosstabResult)
            print("the p-value of the chisq test between", i, "and", j, "is", ChiSqResult[1])

    # print(data['ml_course_categorical'])
    # print(data['information_retrieval_course_categorical'])
    # print(data['statistics_course_categorical'])
    # print(data["major_cleaned"].value_counts())
    # print(data['ml_course_categorical'].value_counts())
    # print(data['information_retrieval_course_categorical'].value_counts())
    # print(data['statistics_course_categorical'].value_counts())
    # data_encoded = pd.get_dummies(data,
    #                               columns=['ml_course_categorical', 'information_retrieval_course_categorical',
    #                                        'statistics_course_categorical'])
    #
    # subdf = data_encoded['major_cleaned']
    # print(data.corr(numeric_only=False).to_string())
    # print(data_encoded["ml_course_categorical_yes"])
    # print(data_copy[])

def impute_categories_majority(df):
    df["ml_course_categorical"] = df["ml_course_categorical"].fillna("yes")
    df['information_retrieval_course_categorical'] = df['information_retrieval_course_categorical'].fillna("0")
    df["database_course"] = df["database_course"].fillna("ja")
    df["statistics_course"] = df["statistics_course"].fillna("mu")
    df["used_chatgpt"] = df["used_chatgpt"].fillna("yes")

    return df

def majority_class_classifier (df):
    y_test = df["major_cleaned"]
    y_pred = np.full((245, 1), "AI")

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy majority class: ", accuracy)

    return accuracy

def K_nearest_neighbours (data):
    subdf = data[["major_cleaned", 'ml_course_categorical', 'information_retrieval_course_categorical',
                  'statistics_course_categorical', 'database_course', 'used_chatgpt', 'gender']]
    list_features = ['ml_course_categorical', 'information_retrieval_course_categorical',
                  'statistics_course_categorical', 'database_course', 'used_chatgpt']
    subdf[list_features] = subdf[list_features].apply(lambda x: x.cat.codes)
    #train, test = train_test_split(subdf, test_size=0.25)


    #print(list_features)
    X = subdf[list_features]
    #print(x_columns)
    y = subdf["major_cleaned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy KNN: ", accuracy)

    return accuracy

def randomforest(data):
    subdf = data[["major_cleaned", 'ml_course_categorical', 'information_retrieval_course_categorical',
                  'statistics_course_categorical', 'database_course', 'used_chatgpt', 'gender']]
    list_features = ['ml_course_categorical', 'information_retrieval_course_categorical',
                  'statistics_course_categorical', 'database_course', 'used_chatgpt']
    subdf[list_features] = subdf[list_features].apply(lambda x: x.cat.codes)

    X = subdf[list_features]
    y = subdf["major_cleaned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf= RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Random Forest:", accuracy)
    return



if __name__ == '__main__':
    # load in data file
    data = pd.read_csv('ODI-2024.csv')
    data = clean(data)
    data = clean_classification(data)
    explore_data(data)

    classification_prep(data)

    data = remove_outliers(data)

    data_1 = impute_categories_majority(data)

    majority_class_classifier(data_1)
    K_nearest_neighbours(data_1)
    #plot_cleaned_data(data)
    randomforest(data_1)

