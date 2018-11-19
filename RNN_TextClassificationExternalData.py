# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 12:06:43 2018

@author: Sanmoy
"""
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
custom=set(stopwords.words('english')+list(punctuation)+[',', '“', '’'])

import os
os.getcwd()
os.chdir("C:\\F\\NMIMS\\DataScience\\Sem-3\\DL\\data")

reviews = pd.read_csv("reviews.txt", sep="\n", header=None)

###Text Preprocessing
for idx, eachRow in reviews.iterrows():
    newRow=" ".join([word for word in word_tokenize(eachRow[0]) if word not in custom])
    eachRow[0]=newRow

reviews.head()


labels = pd.read_csv("labels.txt", sep="\n", header=None)
labels.head()

from sklearn.model_selection import train_test_split as tts
y = labels[0].replace({'positive':0, 'negative':1})
x_train, x_test, y_train, y_test = tts(reviews[0], y, test_size=0.30, random_state=53)

x_train.shape
x_test.shape

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence, text

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(x_train)
word_index = token.word_index
len(word_index)

### Data Pre-Processing
max_words = len(word_index)
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(x_train)
sequences = tok.texts_to_sequences(x_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)


embedding_vecor_length=32


from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping

### Define RNN Structure
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,embedding_vecor_length,input_length=max_len)(inputs)
    layer = LSTM(32)(layer)
    layer = Dense(128,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()


model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])

##Model Fitting
model.fit(sequences_matrix,y_train,batch_size=16,epochs=3,
validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

##Evaluate Model
accTest = model.evaluate(test_sequences_matrix,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accTest[0],accTest[1]))
