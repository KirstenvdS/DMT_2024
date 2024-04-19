from exploratory_data_analysis import clean, clean_classification, remove_outliers
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from sklearn.neighbors import KNeighborsRegressor


def classification_prep(data):
    data['ml_course_categorical'] = pd.Categorical(data["ml_course"], categories=["yes", "no"], ordered=False)
    data['information_retrieval_course_categorical'] = pd.Categorical(data["information_retrieval_course"],
                                                                      categories=["1", "0"], ordered=False)
    data['statistics_course_categorical'] = pd.Categorical(data["statistics_course"],
                                                           categories=["mu", "sigma"], ordered=False)

    subdata = data[["major_cleaned", 'ml_course_categorical', 'information_retrieval_course_categorical',
                    'statistics_course_categorical']]

    for i in subdata.columns:
        for j in subdata.columns:
            CrosstabResult = pd.crosstab(index=subdata[i], columns=subdata[j])
            ChiSqResult = chi2_contingency(CrosstabResult)
            print("the p-value of the chisq test between", i, "and", j, "is", ChiSqResult[1])

    # print(data['ml_course_categorical'])
    # print(data['information_retrieval_course_categorical'])
    # print(data['statistics_course_categorical'])
    print(data["major_cleaned"].value_counts())
    print(data['ml_course_categorical'].value_counts())
    print(data['information_retrieval_course_categorical'].value_counts())
    print(data['statistics_course_categorical'].value_counts())
    # data_encoded = pd.get_dummies(data,
    #                               columns=['ml_course_categorical', 'information_retrieval_course_categorical',
    #                                        'statistics_course_categorical'])
    #
    # subdf = data_encoded['major_cleaned']
    # print(data.corr(numeric_only=False).to_string())
    # print(data_encoded["ml_course_categorical_yes"])
    # print(data_copy[])


def K_nearest_neighbours(data):
    train, test = train_test_split(data, test_size=0.25)
    subdf = train.drop(["major_cleaned", "timestamp"], axis=1)
    x_columns = list(train.columns.values)
    y_column = ["major_cleaned"]

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(train[x_columns], train[y_column])
    knn.predict(test[x_columns])
    return


def randomforest(data):
    train, test = train_test_split(data, test_size=0.25)
    clf = RandomForestClassifier()

    subdf = train.drop(["major_cleaned", "timestamp"], axis=1)
    clf.fit(subdf, train["major_cleaned"])

    return


if __name__ == '__main__':
    # load in data file
    data = pd.read_csv('ODI-2024.csv')
    data = clean(data)
    data = clean_classification(data)
    classification_prep(data)
    data = remove_outliers(data)
    K_nearest_neighbours(data)
    # randomforest(data)
