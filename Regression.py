# imports
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse



def clean(df):
    
    column_names = ["timestamp", "major", "ml_course", "information_retrieval_course", "statistics_course",
                "database_course", "gender", "used_chatgpt", "birthday", "no_students", "stand_up",
                "stress_level", "sport_hours", "rand_number", "bedtime", "good_day_1", "good_day_2"]

    df.columns = column_names 

    df = df.drop(['timestamp','rand_number','good_day_1','good_day_2','no_students','stand_up','birthday'], axis = 1)

    df = df.dropna()

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

    df["major"] = "other"
    assert not any(cls_master & ai_master & cs_master & bioinformatics), "some strings match multiple master categories"
    for i in df.index:
        if ai_master[i]:
            df.loc[i, "major"] = "AI"
        if cs_master[i]:
            df.loc[i, "major"] = "CS"
        if cls_master[i]:
            df.loc[i, "major"] = "CLS"
        if bioinformatics[i]:
            df.loc[i, "major"] = "bioInf"
        if business_analytics[i]:

            df.loc[i, "major"] = "BA"
            
    bedtimes = [1, 1, 0, 2, 0, -1, 0, 3, 1, 1, -1, 5, 0, 2, 100, 0, 3, 2, 0, 0, 1, 2, 0, 1,-2, 4, 0, 0, 2, -1, 4, -1, 1,
                2, 1, 2, 0, 1, 1, 5, 2, -2, 0, 1, -3, 0, 4, 3, 0, 0, 1, 3, 2, 1, 0, 1, 0, 0, 2, 2, 2, 0, 2, 1, 1, 0,
                4,
                3, -1, -1, 0, 0, 3, 3, 6, 1, 3, 4, 1, 1, -2, 2, 2, 1, 0, 2, 5, 3, 0, 1, 2, -1, -2, 1, 1, 0, 1, 1, 3, 1,
                1, 0, 0, 0, 2, -2, -1, 2, 2, -1, 4, 0, 2, 100, -1, 0, 2, 1, 1, 3, 2, 1, 0, 0, 1, 2, 3, 1, 0, 0, 100, 1, 1,
                1, 3, 7, 0, 100, 2, -1, 2, 2, 2, 2, 3, 1, 0, 2, 0, -1, -2, 0, 3, 1, 0, 2, -2, -2, 0, 4, 1, 3, 0, 1, 1, 0,
                -1, 1, -1, 3, 1, 3, 4, 1, 1, -1, 1, 1, 1, 0, 4, 1, 0, 0, 2, 0, 1, 7, -1, -1, 0, 2, 1, 2, 2, 1, 0, 1, 3,
                2, -1, 6, 0, 3, 2, 2, 3, -1, 1, 100, 0, 4, 1, 3, 1, 0, 2, 4, 4, 1, -1, 3, -1, 3, 2, 4, 0, 1, 2, 0, 3, 0,
                3, 0, 0, 0, 100, 2, 2, -1, 1, 0, -1, 0, 3]
    df["bedtime"] = bedtimes
    df.loc[df["bedtime"] == 100, "bedtime"] = np.nan
    
    df.loc[df["sport_hours"] == "6 hours", "sport_hours"] = "6"
    df.loc[df["sport_hours"] == "Whole 50", "sport_hours"] = "50"
    df.loc[df["sport_hours"] == "1/2", "sport_hours"] = "0.5"
    df.loc[df["sport_hours"] == "geen", "sport_hours"] = "0"
    df.loc[df["sport_hours"] == "2-4", "sport_hours"] = "3"
    df["sport_hours"] = pd.to_numeric(df.loc[:, "sport_hours"])
    
    df.loc[df["sport_hours"] > 49, "sport_hours"] = np.nan

    # return new dataframe
    return df


def preprocessing(df):
    
    class MultiColumnLabelEncoder:
        def __init__(self,columns = None):
            self.columns = columns # array of column names to encode

        def fit(self,X,y=None):
            return self # not relevant here

        def transform(self,X):
            '''
            Transforms columns of X specified in self.columns using
            LabelEncoder(). If no columns specified, transforms all
            columns in X.
            '''
            output = X.copy()
            if self.columns is not None:
                for col in self.columns:
                    output[col] = LabelEncoder().fit_transform(output[col])
            else:
                for colname,col in output.iteritems():
                    output[colname] = LabelEncoder().fit_transform(col)
            return output

        def fit_transform(self,X,y=None):
            return self.fit(X,y).transform(X)
    
    df = MultiColumnLabelEncoder(columns = ['major','ml_course','information_retrieval_course','statistics_course','database_course', 'gender','used_chatgpt']).fit_transform(df)

    for index, row in df.iterrows():
        if row['stress_level'] > 100:
            df.loc[index,'stress_level'] = 100
        
        if row['stress_level'] < 0:
            df.loc[index,'stress_level'] = 0
            
        if row['ml_course'] == 2:
            df.loc[index,'ml_course'] = 1
        
        if row['ml_course'] == 1:
            df.loc[index,'ml_course'] = np.nan
        
        if row['information_retrieval_course'] == 2:
            df.loc[index,'information_retrieval_course'] = np.nan
            
        if row['statistics_course'] == 2:
            df.loc[index,'statistics_course'] = np.nan
        
        if row['database_course'] == 1:
            df.loc[index,'database_course'] = 0
            
        if row['database_course'] == 0:
            df.loc[index,'database_course'] = 1
            
        if row['database_course'] == 2:
            df.loc[index,'database_course'] = np.nan
        
        if row['used_chatgpt'] == 2:
            df.loc[index,'used_chatgpt'] = 1
        
        if row['used_chatgpt'] == 1:
            df.loc[index,'used_chatgpt'] = np.nan
            
        if row['gender'] == 2:
            df.loc[index,'gender'] = 1
        
        if (row['gender'] == 1) or (row['gender'] == 3) or (row['gender'] == 4) or (row['gender'] == 5):
            df.loc[index,'gender'] = 2    
            
        if (row['sport_hours'] == '4-Feb') or (row['sport_hours'] == '2-Jan'):
            df.loc[index,'sport_hours'] = np.nan

    return df
          
def KNN_imputation(df):

    imputer = KNNImputer(n_neighbors=5, missing_values=np.nan)
    imputed_values = imputer.fit_transform(df[['major','ml_course','information_retrieval_course','statistics_course','database_course','used_chatgpt','sport_hours','bedtime','stress_level']])

    imputed_df = pd.DataFrame(imputed_values, columns=['major','ml_course','information_retrieval_course','statistics_course','database_course','used_chatgpt','sport_hours','bedtime','stress_level'])

    for index, row in imputed_df.iterrows():
        if row['ml_course'] <= 0.5 and row['ml_course'] > 0:
            imputed_df.loc[index,'ml_course'] = 0
            
        if row['ml_course'] >= 0.5 and row['ml_course'] < 1:
            imputed_df.loc[index,'ml_course'] = 1
        
        if row['information_retrieval_course'] <= 0.5 and row['information_retrieval_course'] > 0:
            imputed_df.loc[index,'information_retrieval_course'] = 0
            
        if row['information_retrieval_course'] >= 0.5 and row['information_retrieval_course'] < 1:
            imputed_df.loc[index,'information_retrieval_course'] = 1
        
        if row['statistics_course'] <= 0.5 and row['statistics_course'] > 0:
            imputed_df.loc[index,'statistics_course'] = 0
            
        if row['statistics_course'] >= 0.5 and row['statistics_course'] < 1:
            imputed_df.loc[index,'statistics_course'] = 1
            
        if row['database_course'] <= 0.5 and row['database_course'] > 0:
            imputed_df.loc[index,'database_course'] = 0
            
        if row['database_course'] >= 0.5 and row['database_course'] < 1:
            imputed_df.loc[index,'database_course'] = 1
        
        if row['used_chatgpt'] <= 0.5 and row['used_chatgpt'] > 0:
            imputed_df.loc[index,'used_chatgpt'] = 0
            
        if row['used_chatgpt'] >= 0.5 and row['used_chatgpt'] < 1:
            imputed_df.loc[index,'used_chatgpt'] = 1
        
    return imputed_df
    


def multiple_linear_regression(df):
    
    X = df.drop(columns=['stress_level'])
    y = df['stress_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    c = lr.intercept_
    m = lr.coef_
    #print(c)
    #print(m)

    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)
    
    r2 = r2_score(y_test, y_pred_test)
    print("r2: {}".format(r2))

    plt.scatter(y_test, y_pred_test)
    plt.xlabel("Actual stress level")
    plt.ylabel("Predicted stress level")
    plt.show()
    
    mean_absolute_error = mae(y_pred_test, y_test)
    mean_squared_error = mse(y_pred_test, y_test)
    print("mean absolute error: {}".format(mean_absolute_error))
    print("mean squared error: {}".format(mean_squared_error))
    
    return

def simple_regression(data):
    
    X = data.drop(columns=['major','ml_course','information_retrieval_course','statistics_course','database_course','used_chatgpt','sport_hours','stress_level'])
    y = data['stress_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    c = lr.intercept_
    m = lr.coef_
    #print(c)
    #print(m)
    
    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)
    
    r2 = r2_score(y_test, y_pred_test)
    print("r2: {}".format(r2))

    # Prediction on training set
    plt.scatter(X_train, y_train, color = 'lightcoral')
    plt.plot(X_train, y_pred_train, color = 'firebrick')
    plt.title('Bedtime vs Stress level (Training Set)')
    plt.xlabel('Bedtime')
    plt.ylabel('Stress level')
    plt.legend(['X_train/Pred(y_test)', 'X_train/y_train'], title = 'Bedtime/Stress level', loc='best', facecolor='white')
    plt.box(False)
    plt.show()
    
    # Prediction on training set
    plt.scatter(X_test, y_test, color = 'lightcoral')
    plt.plot(X_test, y_pred_test, color = 'firebrick')
    plt.title('Bedtime vs Stress level (Training Set)')
    plt.xlabel('Bedtime')
    plt.ylabel('Stress level')
    plt.legend(['X_train/Pred(y_test)', 'X_train/y_train'], title = 'Bedtime/Stress level', loc='best', facecolor='white')
    plt.box(False)
    plt.show()
    
    mean_absolute_error = mae(y_pred_test, y_test)
    mean_squared_error = mse(y_pred_test, y_test)
    print("mean absolute error: {}".format(mean_absolute_error))
    print("mean squared error: {}".format(mean_squared_error))
    
    return

df = pd.read_csv('ODI-2024.csv')
df = clean(df)
df = preprocessing(df)
df = KNN_imputation(df)
multiple_linear_regression(df)
#simple_regression(df)
