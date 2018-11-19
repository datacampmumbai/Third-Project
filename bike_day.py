# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:42:45 2018

@author: Tejas
"""

import os
import pandas as pd
import numpy as np

path="F:/NMIMS Mtech (Sem-III)/Deep learning"
os.chdir(path)

##Read CSV
day_data = pd.read_csv("day.csv")
day_data.info()
day_data.describe()
day_data.head(3)
day_data.tail(3)

import matplotlib.pyplot as plt
day_data.plot("dteday", "cnt")

dummy_fields = ['season', 'weathersit', 'mnth', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(day_data[each], prefix=each, drop_first=False)
    day_data = pd.concat([day_data, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'casual', 'registered']
d_data = day_data.drop(fields_to_drop, axis=1)
d_data.head(3)
d_data.info()


from sklearn.preprocessing import StandardScaler
quant_features = ['cnt']

for each in quant_features:
    d_data[[each]] = StandardScaler().fit_transform(d_data[[each]])


X_day=d_data.iloc[:, d_data.columns != 'cnt'].values
y_day=d_data.iloc[:, 5].values

#Splitting into train and test
from sklearn.cross_validation import train_test_split
X_train_day, X_test_day, y_train_day, y_test_day = train_test_split(X_day, y_day, test_size=0.25, random_state=0)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
classifier = Sequential()

classifier = Sequential()
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 31))
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform'))
classifier.compile(optimizer = 'RMSprop', loss = 'mean_squared_error', metrics=['mse'])

#Fit the model on traiing set
classifier.fit(X_train_day, y_train_day, batch_size=10, epochs=100)
score_day = classifier.evaluate(X_test_day, y_test_day, batch_size=10)
score_day
# Predicting the Test set results
y_pred_day = classifier.predict(X_test_day)

from sklearn.metrics import r2_score
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test_day, y_pred_day))


