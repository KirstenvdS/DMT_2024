from exploratory_data_analysis import clean, clean_classification, remove_outliers, impute_missing_values
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from sklearn.neighbors import KNeighborsRegressor
from miceforest import ImputationKernel


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


def majority_class_classifier (df):
    y_test = df["major_cleaned"]
    y_pred = np.full((245, 1), "AI")

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy majority class: ", accuracy)

    return accuracy

def hyperparameterK (x_train, y_train):
    k_values = [i for i in range(1, 16)]
    scores = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, x_train, y_train, cv=10)
        scores.append(np.mean(score))

    best_index = np.argmax(scores)
    best_k = k_values[best_index]

    #print(scores, best_k)

    return best_k


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

    k = hyperparameterK(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy KNN: ", accuracy)

    interval = 1.96 * sqrt((accuracy * (1 - accuracy)) / 245)
    print("Accuracy interval KNN:", interval)

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

    interval = 1.96 * sqrt((accuracy * (1 - accuracy)) / 245)

    print("Accuracy interval RF:", interval)

    return

if __name__ == '__main__':
    # load in data file
    data = pd.read_csv('ODI-2024.csv')
    data = clean(data)
    data = clean_classification(data)
    classification_prep(data)
    data = remove_outliers(data)
    data = impute_missing_values(data)
    K_nearest_neighbours(data)
    # randomforest(data)
