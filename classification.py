from exploratory_data_analysis import clean, clean_classification, remove_outliers, impute_missing_values
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from sklearn.neighbors import KNeighborsRegressor
from miceforest import ImputationKernelfrom exploratory_data_analysis import clean, clean_classification, remove_outliers, impute_missing_values
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, chi2_contingency
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from math import sqrt
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns





def classification_prep (data):
    data['ml_course'] = pd.Categorical(data["ml_course"], categories=["yes", "no"], ordered=False)
    data['information_retrieval_course'] = pd.Categorical(data["information_retrieval_course"],
                                                                           categories=["1", "0"], ordered=False)
    data['statistics_course'] = pd.Categorical(data["statistics_course"],
                                                                categories=["mu", "sigma"], ordered=False)
    data['database_course'] = pd.Categorical(data["database_course"],
                                                                categories=["ja", "nee"], ordered=False)
    data['used_chatgpt'] = pd.Categorical(data["used_chatgpt"],
                                                                categories=["yes", "no"], ordered=False)
    # data['gender'] = pd.Categorical(data["gender"])
    # data['stand_up'] = pd.Categorical(data["stand_up"])

    return data


def classification_chi_sq(data):
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


def majority_class_classifier (df):
    y_test = df["major_cleaned"]
    y_pred = np.full((245, 1), "AI")

    accuracy = accuracy_score(y_test, y_pred)
    #print("Accuracy majority class: ", accuracy)

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
    # subdf = data[["major_cleaned", 'ml_course', 'information_retrieval_course',
    #               'statistics_course', 'database_course', 'used_chatgpt']]
    # list_features = ['ml_course', 'information_retrieval_course',
    #               'statistics_course', 'database_course', 'used_chatgpt']
    # subdf[list_features] = subdf[list_features].apply(lambda x: x.cat.codes)
    # #train, test = train_test_split(subdf, test_size=0.25)
    #
    #
    # #print(list_features)
    subdf = data[["major_cleaned", 'ml_course', 'information_retrieval_course',
                  'statistics_course', 'database_course', 'used_chatgpt', 'gender', 'happy_social', 'happy_weather', 'happy_exercise',
                  'happy_food', 'happy_sleep', 'bed_late', 'age', 'sport_cleaned', 'stress_cleaned', 'stress_level']]
    list_cat_features = ['ml_course', 'information_retrieval_course',
                  'statistics_course', 'database_course', 'used_chatgpt', 'gender', 'happy_social', 'happy_weather', 'happy_exercise',
                  'happy_food', 'happy_sleep', 'bed_late', 'stress_level']
    for c in list_cat_features:
        subdf[c] = pd.Categorical(subdf[c])

    subdf[list_cat_features] = subdf[list_cat_features].apply(lambda x: x.cat.codes)

    list_features = list_cat_features

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

    cm = confusion_matrix(y_test, y_pred)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    plt.show()


    interval = 1.96 * sqrt((accuracy * (1 - accuracy)) / 245)
    #print("Accuracy interval KNN:", interval)

    return accuracy, k

def randomforest(data):
    subdf = data[["major_cleaned", 'ml_course', 'information_retrieval_course',
                  'statistics_course', 'database_course', 'used_chatgpt', 'gender', 'happy_social', 'happy_weather', 'happy_exercise',
                  'happy_food', 'happy_sleep', 'bed_late', 'age', 'sport_cleaned', 'stress_cleaned', 'stress_level']]
    list_cat_features = ['ml_course', 'information_retrieval_course',
                  'statistics_course', 'database_course', 'used_chatgpt', 'gender', 'happy_social', 'happy_weather', 'happy_exercise',
                  'happy_food', 'happy_sleep', 'bed_late', 'stress_level']
    for c in list_cat_features:
        subdf[c] = pd.Categorical(subdf[c])

    subdf[list_cat_features] = subdf[list_cat_features].apply(lambda x: x.cat.codes)

    #list_num_features = ['sport_cleaned', 'age', 'stress_cleaned']
    list_features = list_cat_features

    X = subdf[list_features]
    y = subdf["major_cleaned"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    param_dist = {'n_estimators': randint(50, 500),
                  'max_depth': randint(1, 20)}



    rf= RandomForestClassifier()

    rand_search = RandomizedSearchCV(rf,
                                     param_distributions=param_dist,
                                     n_iter=5,
                                     cv=5)

    rand_search.fit(X_train, y_train)

    best_rf = rand_search.best_estimator_
    # print('Best hyperparameters:', rand_search.best_params_)

    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Random Forest:", accuracy)
    #
    # interval = 1.96 * sqrt((accuracy * (1 - accuracy)) / 245)
    #
    # print("Accuracy interval RF:", interval)
    #
    cm = confusion_matrix(y_test, y_pred)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    plt.show()

    feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    plt_2 = feature_importances.plot.bar()

    #
    #
    plt.show()

    return accuracy, rand_search.best_params_

def bootstrapping(data):
    # configure bootstrap
    n_iterations = 1000
    # run bootstrap
    stats = list()
    depth_stats = list()
    n_estimator_stats = list()
    for i in range(n_iterations):
        # prepare train and test sets
        score, hyperparameterscore = randomforest(data)
        stats.append(score)
        depth_stats.append(hyperparameterscore['max_depth'])
        n_estimator_stats.append(hyperparameterscore['n_estimators'])
        if i % 100 == 0 :
            print("RF:", i)
    # plot scores
    fig = plt.figure()
    rf_hist_acc = plt.hist(stats)
    # rf_hist_acc.set_title("RF Accuracy")
    # rf_hist_acc.set_xlabel("Accuracy scores")
    fig.savefig("rf_hist_acc.png")

    fig = plt.figure()
    rf_hist_depth = plt.hist(depth_stats)
    # rf_hist_depth.set_title("RF Depth Trees")
    # rf_hist_depth.set_xlabel("Depth Trees")
    fig.savefig("rf_hist_depth.png")

    fig = plt.figure()
    rf_hist_ntrees = plt.hist(n_estimator_stats)
    # rf_hist_ntrees.set_title("RF Number of Trees")
    # rf_hist_ntrees.set_xlabel("Number of Trees")
    fig.savefig("rf_hist_ntrees.png")

    # confidence intervals
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    print('random forest: %.1f confidence interval %.1f%% and %.1f%%' % (alpha * 100, lower * 100, upper * 100))

    rf_df = pd.DataFrame(list(zip(stats, depth_stats, n_estimator_stats)), columns=['accuracy', 'depth_of_trees', 'number_of_trees'])

    rf_corr = round(rf_df.corr())
    mask = np.triu(np.ones_like(rf_corr, dtype=bool))
    fig = plt.figure()
    sns.heatmap(rf_corr, annot = True, vmax = 1, vmin = -1, center = 0, cmap = 'vlag', mask=mask, fmt= ".2f")
    fig.savefig("corr_matrix_rf_heatmap.png")
    #plt.show()

    fig = plt.figure()
    rf_depth_acc = sns.scatterplot(x="depth_of_trees", y="accuracy", data=rf_df)
    rf_depth_acc.set_title("RF Accuracy vs. Tree depth")
    rf_depth_acc.set_xlabel("Tree depth")
    fig.savefig("rf_depth_acc.png")
    # plt.show()

    fig = plt.figure()
    rf_ntrees_acc = sns.scatterplot(x="number_of_trees", y="accuracy", data=rf_df)
    rf_ntrees_acc.set_title("RF Accuracy vs. Number of trees")
    rf_ntrees_acc.set_xlabel("Number of trees")
    fig.savefig("rf_ntrees_acc.png")
    # plt.show()

    fig = plt.figure()
    rf_ntrees_depth = sns.scatterplot(x="number_of_trees", y="depth_of_trees", data=rf_df)
    rf_ntrees_depth.set_title("RF Tree depth vs. Number of trees")
    rf_ntrees_depth.set_xlabel("Number of trees")
    rf_ntrees_depth.set_ylabel("Tree depth")
    fig.savefig("rf_ntrees_depth.png")
    # plt.show()


    # configure bootstrap
    n_iterations = 1000
    # run bootstrap
    stats = list()
    stats_k = list()
    for i in range(n_iterations):
        # prepare train and test sets
        score, best_k = K_nearest_neighbours(data)
        stats.append(score)
        stats_k.append(best_k)
        if i % 100 == 0 :
            print("KNN:", i)
    # plot scores
    fig = plt.figure()
    knn_hist_acc = plt.hist(stats)
    # knn_hist_acc.set_title("KNN Accuracy")
    # knn_hist_acc.set_xlabel("Accuracy scores")
    fig.savefig("knn_hist_acc.png")

    fig = plt.figure()
    knn_hist_k = plt.hist(stats_k)
    # title = "KNN K"
    # xlabel = "K"
    # knn_hist_k.set_title(title)
    # knn_hist_k.set_xlabel(xlabel)
    fig.savefig("knn_hist_k.png")

    knn_df = pd.DataFrame(list(zip(stats, stats_k)), columns=['accuracy', 'best_k'])
    knn_corr = round(knn_df.corr(), 2)
    print(knn_corr)

    fig = plt.figure()
    knn_acc_k = sns.scatterplot(x="best_k", y="accuracy", data=knn_df)
    knn_acc_k.set_title("KNN Accuracy vs. K")
    knn_acc_k.set_xlabel("K")
    knn_acc_k.set_ylabel("Accuracy score")
    fig.savefig("knn_acc_k.png")

    #plt.show()
    # confidence intervals
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    print('KNN: %.1f confidence interval %.1f%% and %.1f%%' % (alpha * 100, lower * 100, upper * 100))

    print("DONE BOOTSTRAPPING")

if __name__ == '__main__':
     # load in data file
     data = pd.read_csv('ODI-2024.csv')
     data = clean(data)
     data = clean_classification(data)
     data = classification_prep(data)
     #classification_chi_sq(data)
     data = remove_outliers(data)
     data = impute_missing_values(data)
     majority_class_classifier(data)
     K_nearest_neighbours(data)
     randomforest(data)
     # bootstrapping(data)



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
