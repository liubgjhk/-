from random import random

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import numpy as np


def method():
    x = pd.read_csv('train.csv')
    df = pd.DataFrame(x)
    z = pd.read_csv('test.csv')
    DF = pd.DataFrame(z)
    x_train = df.drop(['ID', 'CLASS'], axis=1)
    x_test = DF.drop(columns=['ID'])
    y_train = df['CLASS']
    knn = KNeighborsClassifier(n_neighbors=3)
    param_dict={'n_neighbors':[1,5,8,10]}
    knn = GridSearchCV(knn,param_grid=param_dict,cv=5)

    knn.fit(x_train, y_train)
    predict = knn.predict(x_test)
    ID = range(210, 314)
    data = pd.DataFrame({"ID": ID, "CLASS": predict})
    data.to_csv("submission1.csv", index=False)


def method1():
    x = pd.read_csv('train.csv')
    df = pd.DataFrame(x)
    z = pd.read_csv('test.csv')
    DF = pd.DataFrame(z)
    x_train = df.drop(['ID', 'CLASS'], axis=1)
    x_test = DF.drop(columns=['ID'])
    y_train = df['CLASS']
    logistic = LogisticRegression()
    logistic.fit(x_train, y_train)
    predict = logistic.predict(x_test)
    ID=range(210,314)
    data = pd.DataFrame({"ID":ID,"CLASS":predict})
    data.to_csv("submission1.csv",index=False)


def method2():
    x = pd.read_csv('train.csv')
    df = pd.DataFrame(x)
    z = pd.read_csv('test.csv')
    DF = pd.DataFrame(z)
    x_train = df.drop(['ID', 'CLASS'], axis=1)
    x_test = DF.drop(columns=['ID'])
    y_train = df['CLASS']
    knn = DecisionTreeClassifier()
    knn.fit(x_train, y_train)
    predict = knn.predict(x_test)
    ID = range(210, 314)
    data = pd.DataFrame({"ID": ID, "CLASS": predict})
    data.to_csv("submission1.csv", index=False)


def method3():
    x = pd.read_csv('train.csv')
    df = pd.DataFrame(x)
    z = pd.read_csv('test.csv')
    DF = pd.DataFrame(z)
    x_train = df.drop(['ID', 'CLASS'], axis=1)
    x_test = DF.drop(columns=['ID'])
    y_train = df['CLASS']
    s=StandardScaler()
    x_train=s.fit_transform(x_train)
    x_test=s.fit_transform(x_test)
    print('开始训练...')
    # 直接初始化LGBMRegressor
    # 这个LightGBM的Regressor和sklearn中其他Regressor基本是一致的
    gbm = lgb.LGBMRegressor(objective='regression',
                            num_leaves=31,
                            learning_rate=0.05,
                            n_estimators=20)

    # 使用fit函数拟合
    gbm.fit(x_train, y_train)

    # 预测
    print('开始预测...')
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)
    c=[]
    x=0.5
    for m in y_pred:
        result=1 if m>x else 0
        c.append(result)
    print(c)
    ID = range(210, 314)
    data = pd.DataFrame({"ID": ID, "CLASS": c})
    data.to_csv("submission1.csv", index=False)


if __name__ == "__main__":
    method3()
