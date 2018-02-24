#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 23:45:33 2018

@author: Sai Bhavani Chandra Sekhar(1211446116)
A random forest classifier fits a number of decision tree classifiers on samples of the dataset 
#and use averaging to improve the predictive accuracy.
#It also controls control over-fitting. 
#We are using this classifier as it can handle any type of attribute like numerical,cateogroical and ordinal
#we can obtain a high accuracy using this classifier.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("/home/sai/Downloads/processed.cleveland.csv", header=None)
##replacing the missing values "?" with nan
dataset[[10,11,12,13]]=dataset[[10,11,12,13]].replace('?',np.NaN)

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
#replacing Nan by the mode value of the coloumn. 
#since there are missing values in coloumn 12 and 13, and both represent cateogorical data,we replace it with most frequent value of the column             
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = np.NaN, strategy = 'most_frequent', axis = 0)
X = imputer.fit_transform(X)
#split train and test data           
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#fit random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 300, criterion="gini")

rf.fit(X_train, Y_train)
predicted = rf.predict(X_test)
#print accuracy
from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(Y_test,predicted)
print (accuracy)
