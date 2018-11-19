# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 12:20:11 2018

@author: Tejas
"""

import os
import pandas as pd
import numpy as np

path="C:/F/NMIMS/DataScience/Sem-3/DL/data"
os.chdir(path)

##Read CSV
hour_data = pd.read_csv("hour.csv")
hour_data.info()
hour_data.describe()
hour_data.head(3)
hour_data.tail(3)

import matplotlib.pyplot as plt
hour_data.plot("dteday", "cnt")

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(hour_data[each], prefix=each, drop_first=False)
    hour_data = pd.concat([hour_data, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr', 'casual', 'registered']
data = hour_data.drop(fields_to_drop, axis=1)
data.head(3)
data.info()


from sklearn.preprocessing import StandardScaler
quant_features = ['cnt']

for each in quant_features:
    data[[each]] = StandardScaler().fit_transform(data[[each]])


X_hr=data.iloc[:, data.columns != 'cnt'].values
y_hr=data.iloc[:, 5].values

#Splitting into train and test
from sklearn.cross_validation import train_test_split
X_train_hr, X_test_hr, y_train_hr, y_test_hr = train_test_split(X_hr, y_hr, test_size=0.25, random_state=0)


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
classifier = Sequential()

classifier = Sequential()
classifier.add(Dense(units = 29, kernel_initializer = 'uniform', activation = 'relu', input_dim = 56))
classifier.add(Dense(units = 29, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform'))
classifier.compile(optimizer = 'RMSprop', loss = 'mean_squared_error', metrics=['mse'])

#Fit the model on traiing set
classifier.fit(X_train_hr, y_train_hr, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred_hour = classifier.predict(X_test_hr)
score_hr = classifier.evaluate(X_test_hr, y_test_hr, batch_size=10)
score_hr

from sklearn.metrics import r2_score
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test_hr, y_pred_hour))


